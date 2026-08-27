"""
main.py

Orchestrates all modules in a single event loop.
Run this to start the navigation system.

Usage:
    python main.py --building mybuilding --stream http://192.168.1.5:8080/video

Prerequisites:
    1. Run scan_walk.py once per building to build the map (generates DINOv2 index + osmag.json)
    2. Ollama running locally with moondream and qwen2.5:1.5b pulled
    3. Models downloaded to models/ directory
    4. Phone running IP Webcam app (or use camera index e.g. --stream 1)
"""

import argparse
import time
import threading
import os
import hashlib
import sys
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables before importing modules
load_dotenv()

from modules.m1_user_interaction import listen_for_goal, speak, listen_for_interrupt, SPEECH_RATE
from modules.m2_phone_stream import PhoneStream
from modules.m2_slam import SLAMLocalizer
from modules.m2_midas import DepthEstimator
from modules.m4_path_planner import PathPlanner
from modules.m5_yolo import YOLODetector
from modules.m5_vlm_guidance import GuidanceEngine
from modules.m6_usability import start_web_server, update_state, log_event, get_tts_rate
from modules.m0_image_enhancer import ImageEnhancer

import yaml


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(path: str = "config/settings.yaml") -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found at {path}. Using defaults.")
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Indoor Navigation System — Real-time voice guidance for visually impaired users."
    )
    parser.add_argument(
        "--building", required=True,
        help="Building name (must have maps/<name>/osmag.json — run scan_walk.py first)"
    )
    parser.add_argument(
        "--stream", required=True,
        help="Phone IP Webcam URL e.g. http://192.168.1.5:8080/video or camera index e.g. 1"
    )
    parser.add_argument(
        "--config", default="config/settings.yaml",
        help="Path to settings.yaml (default: config/settings.yaml)"
    )
    parser.add_argument(
        "--use-full-slam", action="store_true",
        help="Use full ORB-SLAM3 (requires Linux build). Defaults to DINOv2 fallback."
    )
    args = parser.parse_args()

    cfg      = load_config(args.config)
    map_dir  = Path("maps") / args.building
    osmag_path = map_dir / "osmag.json"

    # ── Pre-flight validation ─────────────────────────────────────────────────
    if not osmag_path.exists():
        logger.error(f"No map found at {osmag_path}.")
        logger.error(f"Run: python scan_walk.py --building {args.building} --stream {args.stream}")
        sys.exit(1)

    dino_index = map_dir / "dino_index.faiss"
    if not dino_index.exists():
        logger.warning(
            "DINOv2 index not found. Localization will not work correctly. "
            "Re-run scan_walk.py to build the index."
        )

    # Verify Ollama is running
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
    except Exception:
        logger.error("Ollama is not running! Start it first: open the Ollama app or run 'ollama serve'.")
        sys.exit(1)

    # ── Initialize all modules ────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("  Multimodal Indoor Navigation System v2  ")
    logger.info("=" * 55)

    # Module 6: Web UI (start first so caregiver can monitor boot)
    logger.info("[INIT] Starting caregiver web UI...")
    start_web_server(port=cfg.get("web_port", 5050))

    # Module 2a: Phone stream
    logger.info(f"[INIT] Connecting to phone stream: {args.stream}")
    stream = PhoneStream(url=args.stream, max_queue_size=3)
    stream.start()

    # Module 2b: SLAM localizer (uses DINOv2 by default)
    use_fallback = not args.use_full_slam
    if cfg.get("use_slam_fallback", True):
        use_fallback = True
    logger.info(f"[INIT] Loading localizer (mode: {'DINOv2+FAISS' if use_fallback else 'ORB-SLAM3'})...")
    slam = SLAMLocalizer(
        map_dir=str(map_dir),
        use_fallback=use_fallback,
        vocab_path=cfg.get("orb_vocab_path", "ORB_SLAM3/Vocabulary/ORBvoc.txt")
    )

    # Module 2c: MiDaS depth
    logger.info("[INIT] Loading MiDaS depth estimator...")
    midas = DepthEstimator(
        model_path=cfg.get("midas_model", "models/midas_v21_small_256.pt"),
        device="cuda"
    )

    # Module 5a: YOLO detector
    logger.info("[INIT] Loading YOLO-World object detector...")
    yolo = YOLODetector(
        model_path=cfg.get("yolo_model", "models/yolov8s-worldv2.pt"),
        conf_threshold=cfg.get("yolo_conf", 0.35),
        device="cuda"
    )

    # Module 4: Path planner
    logger.info("[INIT] Loading path planner...")
    planner = PathPlanner(osmag_path=str(osmag_path))

    # Module 5b: VLM guidance (with narrative memory)
    logger.info("[INIT] Initializing VLM guidance engine...")
    guidance = GuidanceEngine(max_cache_age_sec=cfg.get("vlm_cache_age", 3.0))

    # Module 0: Image Enhancer
    logger.info("[INIT] Loading Image Enhancer...")
    enhancer = ImageEnhancer(config=cfg)

    logger.info("=" * 55)
    logger.info("  All modules ready. Starting navigation.  ")
    logger.info("=" * 55)
    speak("Navigation system is ready. Please tell me your destination.")

    # ── Navigation state ──────────────────────────────────────────────────────
    goal_node:     str | None   = None
    goal_name:     str | None   = None
    path_nodes:    list[str]    = []
    nl_description: str         = ""
    current_node:  str | None   = None
    prev_node:     str | None   = None   # Track node transitions

    depth_result = {
        "depth_map":     None,
        "prompt_text":   "Depth not yet computed.",
        "center_bucket": "unknown",
        "left_bucket":   "unknown",
        "right_bucket":  "unknown"
    }
    yolo_detections: list[dict] = []
    detection_hash:  str        = ""

    # ── Interrupt listener thread ─────────────────────────────────────────────
    def handle_interrupt(command: str):
        nonlocal goal_node, path_nodes
        if command == "stop":
            speak("Navigation stopped.")
            goal_node  = None
            path_nodes = []
            guidance.narrative.reset()
            update_state(is_navigating=False, goal_node=None, goal_name=None)
            logger.info("[Interrupt] Navigation stopped by user.")

        elif command == "repeat":
            last = guidance._last_instruction
            speak(last if last else "No instruction available.")

        elif command == "where am i":
            if current_node:
                name = planner.nodes.get(current_node, {}).get("name", "unknown location")
                speak(f"You are at {name}.")
            else:
                speak("Your location is not yet determined. Keep walking slowly.")

        elif command == "help":
            speak(
                "Say 'stop' to cancel navigation, "
                "'repeat' to hear the last instruction, "
                "or 'where am I' for your current location."
            )

    interrupt_thread = threading.Thread(
        target=listen_for_interrupt,
        args=(handle_interrupt,),
        daemon=True
    )
    interrupt_thread.start()
    logger.info("[Main] Interrupt listener active.")

    # ── Goal acquisition helper ───────────────────────────────────────────────
    def acquire_goal() -> bool:
        nonlocal goal_node, goal_name, path_nodes, nl_description

        spoken = listen_for_goal()
        if not spoken:
            return False

        resolved = planner.resolve_goal_node(spoken)
        if not resolved:
            speak(f"I couldn't find a location matching '{spoken}'. Please try again.")
            return False

        goal_node  = resolved
        goal_name  = planner.nodes[resolved].get("name", resolved)

        log_event("goal_set", {"spoken": spoken, "resolved_node": resolved, "name": goal_name})
        speak(f"Setting destination to {goal_name}. Building your route now.")
        update_state(goal_node=goal_node, goal_name=goal_name, is_navigating=True)
        guidance.narrative.reset()   # Fresh journey narrative for new goal
        logger.info(f"[Main] Goal set: {goal_name} (node: {resolved})")
        return True

    # ── Performance tuning from config ────────────────────────────────────────
    YOLO_EVERY_N       = cfg.get("yolo_every_n",       5)
    MIDAS_EVERY_N      = cfg.get("midas_every_n",      5)
    MIN_SPEAK_INTERVAL = cfg.get("min_speak_interval",  3.0)

    last_spoken_time = 0.0
    logger.info("[Main] Entering main loop.")

    # ── Main navigation loop ──────────────────────────────────────────────────
    while True:
        # 1. Get latest frame from phone
        frame_id, frame = stream.get_frame(timeout=2.0)
        if frame is None:
            logger.warning("[Main] No frame received. Waiting for stream...")
            time.sleep(0.5)
            continue

        # 2. DINOv2 + FAISS localization
        prev_node = current_node
        node = slam.process_frame(frame, timestamp=time.time())
        node_just_changed = (node is not None and node != current_node)
        if node_just_changed:
            current_node  = node
            node_name     = planner.nodes.get(node, {}).get("name", node)
            log_event("node_change", {"node": node, "name": node_name})
            update_state(current_node=node, current_node_name=node_name)
            logger.info(f"[Main] ▶ At node: {node_name}")
        elif node:
            current_node = node

        # 3. If no goal yet, prompt user and wait
        if not goal_node:
            if not acquire_goal():
                time.sleep(1)
                continue

        # 4. Compute or refresh path when current node is known
        if current_node and (not path_nodes or path_nodes[0] != current_node):
            path_nodes = planner.find_path(current_node, goal_node)
            if path_nodes:
                # Cache NL description keyed on goal (doesn't change as user walks)
                nl_description = planner.get_nl_description(path_nodes[0], goal_node, path_nodes)
                log_event("path_computed", {"path": path_nodes, "goal": goal_name})
                logger.info(f"[Main] Path computed: {len(path_nodes)} nodes")
            else:
                speak(f"I cannot find a route to {goal_name} from your current location.")
                goal_node  = None
                path_nodes = []
                update_state(is_navigating=False)
                continue

        # 5. YOLO detection (every N frames)
        yolo_detections, det_changed = yolo.detect(frame, frame_id, every_n=YOLO_EVERY_N)
        detection_hash = hashlib.md5(
            ",".join(sorted(d["class"] for d in yolo_detections)).encode()
        ).hexdigest()

        if det_changed:
            update_state(detections=yolo_detections)
            if any(d["class"] in ["stairs", "step"] for d in yolo_detections):
                log_event("obstacle", {
                    "type":       "stairs",
                    "detections": [d["class"] for d in yolo_detections]
                })

        # 6. MiDaS depth estimation (every N frames)
        if frame_id % MIDAS_EVERY_N == 0:
            depth_result = midas.estimate(frame)
            update_state(depth={
                "prompt_text":   depth_result["prompt_text"],
                "left_bucket":   depth_result["left_bucket"],
                "center_bucket": depth_result["center_bucket"],
                "right_bucket":  depth_result["right_bucket"],
            })

        # 7. Check if arrived at destination
        if goal_node and current_node == goal_node:
            speak(f"You have arrived at {goal_name}.")
            log_event("arrived", {"node": goal_node, "name": goal_name})
            goal_node  = None
            goal_name  = None
            path_nodes = []
            update_state(is_navigating=False, goal_node=None, goal_name=None)
            continue

        # 8. Generate and speak guidance instruction
        now = time.time()

        # SLAM lost: calm fallback, no VLM call
        if current_node is None:
            if now - last_spoken_time >= MIN_SPEAK_INTERVAL * 2:
                speak("I've lost your position. Please walk slowly and look around.")
                last_spoken_time = now
            continue

        # Trigger VLM on node transition (immediately) OR on timer
        should_speak = node_just_changed or (now - last_spoken_time >= MIN_SPEAK_INTERVAL)

        if should_speak:
            next_step = (
                planner.get_next_step_instruction(current_node, path_nodes)
                if path_nodes else ""
            )
            # Prefer VLM-generated edge description over generic BFS step
            edge_desc = (
                planner.get_edge_description(current_node, path_nodes)
                if path_nodes else ""
            )
            # Depth-aware YOLO prompt with per-object distance estimates
            yolo_text = yolo.format_for_prompt(
                yolo_detections,
                depth_map=depth_result.get("depth_map"),
                frame_shape=frame.shape
            )
            
            enhanced_frame = enhancer.enhance(frame)

            instruction = guidance.generate_instruction(
                frame_bgr             = enhanced_frame,
                current_node          = current_node,
                current_node_name     = planner.nodes.get(current_node, {}).get("name", "unknown"),
                next_step_instruction = next_step,
                path_nl_description   = nl_description,
                yolo_text             = yolo_text,
                depth_text            = depth_result.get("prompt_text", ""),
                detection_hash        = detection_hash,
                node_just_changed     = node_just_changed,
                edge_description      = edge_desc
            )

            if instruction and instruction != guidance._last_instruction:
                rate = get_tts_rate()
                speak(instruction, rate=rate)
                last_spoken_time = now
                update_state(last_instruction=instruction)
                log_event("instruction", {
                    "instruction": instruction,
                    "node":        current_node,
                    "goal":        goal_name
                })


if __name__ == "__main__":
    main()
