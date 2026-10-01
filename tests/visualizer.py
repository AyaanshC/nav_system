"""
tests/visualizer.py

Professional real-time UI visualizer for nav_system_ollama.
Plays the video smoothly in the main thread while processing AI (Depth, YOLO, VLM)
in two background threads (Fast Loop & Slow Loop).

Usage:
    python tests/visualizer.py
    python tests/visualizer.py --video test_videos/house_tour.mp4 --no_tts
"""

import argparse
import time
import sys
import os
import cv2
import numpy as np
import threading
import math
from pathlib import Path
import yaml
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.m4_path_planner   import PathPlanner
from modules.m5_yolo           import YOLODetector
from modules.m2_depth_anything import DepthEstimator
from modules.m5_vlm_guidance   import GuidanceEngine
from modules.m0_image_enhancer import ImageEnhancer
from modules.m8_sonification   import SonificationEngine
from modules.m2_dino_localizer import DINOv2Localizer

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="test_videos/house_tour.mp4",
                    help="Video file, webcam index (0/1), or phone stream URL (http://192.168.x.x:8080/video)")
parser.add_argument("--building", default="housetour")
parser.add_argument("--node", default="node_001")
parser.add_argument("--goal", default="node_163",  help="Target node ID (default: Master Bedroom)")
parser.add_argument("--no_tts", action="store_true", help="Disable text-to-speech audio")
parser.add_argument("--no_localize", action="store_true", help="Disable live DINOv2 localization")
parser.add_argument("--vlm_model", default="qwen2.5vl:3b",
                    help="Ollama VLM model used when no --gemini_key provided")
parser.add_argument("--gemini_key", default=os.environ.get("GEMINI_API_KEY", ""),
                    help="Google Gemini API key. Enables Gemini 2.0 Flash (~0.3s) instead of local Ollama.")
args = parser.parse_args()

# ── Resolve video source (webcam int, phone URL, or file path) ────────────────
_v = args.video.strip()
if _v.isdigit():
    VIDEO_SRC   = int(_v)     # e.g. "0" -> webcam 0
    IS_LIVE_CAM = True
elif _v.startswith("http"):
    VIDEO_SRC   = _v          # IP Webcam / phone stream URL
    IS_LIVE_CAM = True
else:
    VIDEO_SRC   = _v          # Video file path
    IS_LIVE_CAM = False

logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")

logger.info("="*60)
logger.info("  Nav System 2.0 — Glassmorphism UI Visualizer")
logger.info("="*60)
logger.info("Initializing AI models (this will take a moment)...")

# --- Initialize Modules ---
osmag_path = Path("maps") / args.building / "osmag.json"
planner    = PathPlanner(osmag_path=str(osmag_path))
yolo       = YOLODetector("models/yolov8s-worldv2.pt", conf_threshold=0.35, device="cuda")
depth_est  = DepthEstimator(device="cuda")
guidance   = GuidanceEngine(max_cache_age_sec=3.0)
with open("config/settings.yaml") as f: cfg = yaml.safe_load(f)
enhancer   = ImageEnhancer(config=cfg)
sonifier   = SonificationEngine()

# --- DINOv2 Live Localizer ---
# Dynamically detects which map node matches the current video frame.
# This is what makes the VLM instructions context-accurate.
if not args.no_localize:
    map_dir = str(Path("maps") / args.building)
    try:
        localizer = DINOv2Localizer(map_dir=map_dir, device="cuda")
        localizer._smooth_k = 1   # Instant transitions (K=1 = no smoothing needed)
        logger.info(f"[DINOv2] Localizer ready: {len(localizer._node_ids)} nodes indexed.")
    except Exception as e:
        logger.warning(f"[DINOv2] Localizer failed to load, using fixed node: {e}")
        localizer = None
else:
    localizer = None
    logger.info("[DINOv2] Localization disabled (--no_localize). Using fixed node.")

if args.no_tts:
    def speak(t, rate=None): pass
else:
    from modules.m1_user_interaction import speak

path_nodes = planner.find_path(args.node, args.goal)
goal_name  = planner.nodes.get(args.goal, {}).get("name", args.goal)
nl_desc    = planner.get_nl_description(args.node, args.goal, path_nodes)
logger.info(f"[Route] {args.node} -> {args.goal} ({goal_name})  ({len(path_nodes)} nodes)")

# --- Shared State ---
state = {
    "frame": None,
    "detections": [],
    "depth_buckets": {"left": "?", "center": "?", "right": "?"},
    "depth_map": None,
    "instruction": "Waiting for visual context...",  # VLM instruction
    "fast_instruction": "",                           # Rule-based instant guidance
    "hazard": None,
    "running": True,
    "vlm_thinking": False,
    # Live localization state — updated by fast_worker
    "current_node": args.node,
    "current_node_name": planner.nodes.get(args.node, {}).get("name", args.node),
    "localize_similarity": 0.0,
    # Goal & hazard tracking
    "goal_reached": False,
    "hazard_spoken": False,     # Prevents repeated critical-hazard TTS spam
}

def fast_worker():
    """Runs Depth, YOLO, and DINOv2 live localization continuously."""
    global state
    frame_count = 0
    localize_every = 3   # Run localization every N iterations
    loc_log_every  = 30  # Print FAISS score every N localization attempts (for debugging)
    loc_attempts   = 0
    while state["running"]:
        if state["frame"] is None:
            time.sleep(0.01)
            continue
            
        frame = state["frame"].copy()
        frame_count += 1
        
        # 1. Depth
        depth_result = depth_est.estimate(frame)
        d_map = depth_result.get("depth_map")
        
        if d_map is not None:
            md = float(np.percentile(d_map, 95))
            sonifier.set_danger_level(min(max((md-0.4)/0.45, 0.0), 1.0))
            
        # 2. YOLO
        detections, _, hazard = yolo.detect(frame, frame_count, every_n=1, depth_map=d_map, depth_estimator=depth_est)
        
        # 3. DINOv2 Live Localization (every N iterations)
        if localizer is not None and frame_count % localize_every == 0:
            try:
                # --- Direct FAISS query for full transparency ---
                import faiss as _faiss
                emb = localizer.embed(frame).astype("float32").reshape(1, -1)
                _faiss.normalize_L2(emb)
                scores, indices = localizer._index.search(emb, k=1)
                score  = float(scores[0][0])
                best   = localizer._node_ids[int(indices[0][0])]
                loc_attempts += 1
                
                if loc_attempts % loc_log_every == 0:
                    node_name_tmp = planner.nodes.get(best, {}).get("name", best)
                    logger.info(f"[DINOv2] FAISS: best={best} ({node_name_tmp}) score={score:.3f}")
                
                if score >= localizer.MIN_SIMILARITY:
                    node_name = planner.nodes.get(best, {}).get("name", best)
                    if best != state["current_node"]:
                        logger.info(f"[DINOv2] Location: {state['current_node_name']} -> {node_name} (score={score:.3f})")
                    state["current_node"]      = best
                    state["current_node_name"] = node_name
                else:
                    logger.debug(f"[DINOv2] Low similarity ({score:.3f}), holding: {state['current_node_name']}")
            except Exception as e:
                logger.warning(f"[DINOv2] Localize error: {e}")
        
        # Update State safely
        state["depth_map"] = d_map
        state["depth_buckets"] = {
            "left":   depth_result.get("left_bucket",   "?"),
            "center": depth_result.get("center_bucket", "?"),
            "right":  depth_result.get("right_bucket",  "?")
        }
        state["detections"] = detections
        
        if hazard:
            close = [d for d in detections if d.get("direction")=="Center" and (d.get("distance_m") or 9) < 1.5]
            if close:
                c = min(close, key=lambda x: x["distance_m"])
                state["hazard"] = f"HAZARD: {c['class']} {c['distance_m']:.1f}m"
            else:
                state["hazard"] = None
        else:
            state["hazard"] = None

        # ── CRITICAL HAZARD: Immediate TTS (< 0.5m, bypasses VLM 3-4s lag) ──
        if not state["goal_reached"]:
            critical = [
                d for d in detections
                if d.get("direction") == "Center" and (d.get("distance_m") or 99) < 0.5
            ]
            if critical and not state["hazard_spoken"]:
                c = min(critical, key=lambda x: x.get("distance_m", 99))
                alert = f"STOP! {c['class']} {c['distance_m']:.1f} metres ahead!"
                logger.warning(f"[HAZARD] {alert}")
                speak(alert)
                state["hazard_spoken"] = True
                state["instruction"]   = alert
            elif not critical:
                state["hazard_spoken"] = False   # Reset once obstacle clears

        # Yield to let UI breathe
        time.sleep(0.02)

# ── Fast Rule-Based Instruction (no model, instant) ──────────────────────────
def make_fast_instruction(detections: list, depth_buckets: dict, edge_desc: str) -> str:
    """
    Generates an instant guidance string from YOLO + depth, no VLM needed.
    Used to fill the gap between VLM calls so the user always has feedback.
    """
    center  = depth_buckets.get("center", "?")
    left_d  = depth_buckets.get("left",   "?")
    right_d = depth_buckets.get("right",  "?")

    # Immediate hazards from YOLO (< 1.5m center)
    close = [d for d in detections
             if d.get("direction") == "Center" and (d.get("distance_m") or 99) < 1.5]
    if close:
        c = min(close, key=lambda x: x.get("distance_m", 99))
        return f"STOP — {c['class']} {c['distance_m']:.1f}m straight ahead!"

    # Left/right warnings
    left_close  = [d for d in detections if d.get("direction") == "Left"  and (d.get("distance_m") or 99) < 1.0]
    right_close = [d for d in detections if d.get("direction") == "Right" and (d.get("distance_m") or 99) < 1.0]
    if left_close:
        return f"Careful — {left_close[0]['class']} close on left."
    if right_close:
        return f"Careful — {right_close[0]['class']} close on right."

    # Depth-based warnings
    if center == "near":
        return "Slow down — obstacle close ahead."
    if center == "mid" and left_d == "near":
        return "Move right — obstacle on left."
    if center == "mid" and right_d == "near":
        return "Move left — obstacle on right."

    # Fall back to edge description from map
    return edge_desc if edge_desc else "Path clear, continue forward."



def slow_worker():
    """
    Dual-backend VLM worker:
      - Gemini 2.0 Flash  (~0.3-0.5s) when --gemini_key is provided  [RECOMMENDED]
      - Ollama qwen2.5vl:3b (~3-4s)   as local fallback
    Both run continuously in a tight loop.
    """
    global state
    import base64 as _b64
    import PIL.Image

    # ── Select backend ───────────────────────────────────────────────────────
    USE_GEMINI = bool(args.gemini_key)
    if USE_GEMINI:
        import google.generativeai as genai
        genai.configure(api_key=args.gemini_key)
        _gemini = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=genai.GenerationConfig(
                max_output_tokens=40,
                temperature=0.1,
            )
        )
        logger.info("[VLM] Backend: Gemini 2.0 Flash ⚡ (cloud, ~0.3s/call)")
    else:
        from openai import OpenAI as _OAI
        _ollama = _OAI(api_key="ollama", base_url="http://localhost:11434/v1")
        logger.info(f"[VLM] Backend: Ollama {args.vlm_model} (local, ~3-4s/call)")

    last_node_for_path = args.node
    current_path_nodes = path_nodes

    # Shared navigation system prompt
    NAV_SYSTEM = (
        "You are a real-time navigation assistant for a blind person. "
        "Your ONLY job is to give a single, short navigation instruction (max 12 words). "
        "Base it strictly on what you see in the image. "
        "DO NOT describe the image. DO NOT say 'I see'. "
        "Output ONLY the instruction, nothing else.\n\n"
        "Examples of correct output:\n"
        "  Move forward, the door is straight ahead.\n"
        "  Stop — chair 0.6 m directly in front.\n"
        "  Turn left at the sofa.\n"
        "  Continue through the hallway, wall on right."
    )

    BAD_PREFIXES = ("the image shows", "i see", "this is", "in this image",
                    "the room", "there is", "there are", "the photo")

    while state["running"]:
        # ── Skip VLM entirely once the user has arrived ──────────────────────
        if state["goal_reached"]:
            time.sleep(0.5)
            continue

        if state["frame"] is None or state["depth_map"] is None:
            time.sleep(0.05)
            continue

        live_node         = state["current_node"]
        live_node_name    = state["current_node_name"]
        node_just_changed = (live_node != last_node_for_path)

        if node_just_changed:
            # ── GOAL REACHED detection ────────────────────────────────────────
            if live_node == args.goal and not state["goal_reached"]:
                state["goal_reached"] = True
                arrival_msg = f"You have arrived at your destination: {goal_name}!"
                state["instruction"] = arrival_msg
                logger.info(f"[NAV] 🏁 Goal reached: {goal_name}")
                speak(arrival_msg)
                last_node_for_path = live_node
                continue   # Skip VLM for this iteration
            try:
                current_path_nodes = planner.find_path(live_node, args.goal)
                logger.info(f"[Planner] Route: {live_node_name} -> {goal_name} ({len(current_path_nodes)} steps)")
            except Exception as e:
                logger.warning(f"[Planner] Reroute failed: {e}")
            last_node_for_path = live_node

        # Tier-1: instant rule-based (no model, 0ms)
        edge_desc = planner.get_edge_description(live_node, current_path_nodes)
        state["fast_instruction"] = make_fast_instruction(
            state["detections"], state["depth_buckets"], edge_desc
        )

        # Tier-2: VLM ──────────────────────────────────────────────────────────
        frame = state["frame"].copy()
        state["vlm_thinking"] = True

        user_msg = (
            f"Current room: {live_node_name}.\n"
            f"Depth sensors: center={state['depth_buckets']['center']}, "
            f"left={state['depth_buckets']['left']}, "
            f"right={state['depth_buckets']['right']}.\n"
            f"Navigation direction: {edge_desc or 'continue forward'}.\n"
            f"Give the navigation instruction for RIGHT NOW:"
        )

        raw = ""
        try:
            if USE_GEMINI:
                # ── Gemini 2.0 Flash path ─────────────────────────────────────
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = PIL.Image.fromarray(rgb)
                # Resize to 512px — Gemini handles it efficiently
                pil_img.thumbnail((512, 512))

                resp = _gemini.generate_content([NAV_SYSTEM + "\n" + user_msg, pil_img])
                raw  = resp.text.strip().strip('"').strip("'")

            else:
                # ── Ollama path ───────────────────────────────────────────────
                h, w   = frame.shape[:2]
                small  = cv2.resize(frame, (320, int(h * 320 / w)))
                _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                b64    = _b64.b64encode(buf.tobytes()).decode()

                resp = _ollama.chat.completions.create(
                    model=args.vlm_model,
                    max_tokens=30,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": NAV_SYSTEM},
                        {"role": "user", "content": [
                            {"type": "text",      "text": user_msg},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}", "detail": "low"
                            }}
                        ]}
                    ]
                )
                raw = resp.choices[0].message.content.strip().strip('"').strip("'")

            # Strip descriptive prefixes either model might produce
            if any(raw.lower().startswith(p) for p in BAD_PREFIXES):
                raw = edge_desc or state["fast_instruction"]

            if raw and raw != state["instruction"]:
                state["instruction"] = raw
                logger.info(f"[VLM] {raw}")
                speak(raw)

        except Exception as e:
            logger.warning(f"[VLM] Error: {e}")
            fb = state["fast_instruction"]
            if fb and fb != state["instruction"]:
                state["instruction"] = fb

        state["vlm_thinking"] = False

logger.info("Starting Background AI Threads...")
threading.Thread(target=fast_worker, daemon=True).start()
threading.Thread(target=slow_worker, daemon=True).start()

# --- Main Video / Camera Setup ---
cap = cv2.VideoCapture(VIDEO_SRC)
if not cap.isOpened():
    logger.error(f"Could not open video source: {VIDEO_SRC}")
    sys.exit(1)

if IS_LIVE_CAM:
    # Live camera: request 720p, use minimal display delay
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = 1   # 1ms — camera provides frames at its own rate
    logger.info(f"[CAM] Live camera source: {VIDEO_SRC} @ {fps:.0f}fps")
else:
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = int(1000 / fps)
    logger.info(f"[CAM] Video file: {VIDEO_SRC} @ {fps:.0f}fps")

orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

W, H = 1280, 720
font = cv2.FONT_HERSHEY_DUPLEX

cv2.namedWindow("Nav System Visualizer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Nav System Visualizer", W, H)

# Pre-compute glass panels to save CPU
def draw_glass_panel(img, x1, y1, x2, y2, alpha=0.6, blur=15):
    roi = img[y1:y2, x1:x2]
    blurred_roi = cv2.GaussianBlur(roi, (blur, blur), 0)
    dark_roi = cv2.addWeighted(blurred_roi, alpha, np.zeros(blurred_roi.shape, blurred_roi.dtype), 0, 0)
    img[y1:y2, x1:x2] = dark_roi
    # Add subtle border
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return img

def draw_ui(img):
    """Draws a professional Glassmorphism HUD on top of the upscaled video frame."""
    
    t = time.time()
    
    # 1. Bounding Boxes with solid background labels
    for d in state["detections"]:
        if "box" not in d: continue
        x1, y1, x2, y2 = d["box"]
        # Scale coords from original resolution to 1280x720 display size
        sx1, sy1 = int(x1 * W / orig_W), int(y1 * H / orig_H)
        sx2, sy2 = int(x2 * W / orig_W), int(y2 * H / orig_H)
        
        # Color code based on distance: Cyan (Safe), Yellow (Caution), Red (Danger)
        dist = d.get("distance_m", 9.9)
        if dist > 3.0: color = (255, 200, 50)  # Soft Cyan
        elif dist > 1.5: color = (0, 200, 255) # Yellow
        else: color = (50, 50, 255)            # Red
        
        # Draw box corners instead of full box for sleek look
        length = 20
        thick = 2
        cv2.line(img, (sx1, sy1), (sx1+length, sy1), color, thick)
        cv2.line(img, (sx1, sy1), (sx1, sy1+length), color, thick)
        cv2.line(img, (sx2, sy1), (sx2-length, sy1), color, thick)
        cv2.line(img, (sx2, sy1), (sx2, sy1+length), color, thick)
        cv2.line(img, (sx1, sy2), (sx1+length, sy2), color, thick)
        cv2.line(img, (sx1, sy2), (sx1, sy2-length), color, thick)
        cv2.line(img, (sx2, sy2), (sx2-length, sy2), color, thick)
        cv2.line(img, (sx2, sy2), (sx2, sy2-length), color, thick)
        
        # Draw Label Background
        label = f"{d['class']} {dist:.1f}m"
        (lw, lh), _ = cv2.getTextSize(label, font, 0.6, 1)
        cv2.rectangle(img, (sx1, max(0, sy1 - lh - 10)), (sx1 + lw + 10, sy1), color, -1)
        # Draw Label Text
        cv2.putText(img, label, (sx1 + 5, max(15, sy1 - 5)), font, 0.6, (0, 0, 0), 1)

    # 2. Draw Center Crosshair
    cx, cy = W//2, H//2
    cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1)
    cv2.line(img, (cx - 20, cy), (cx - 10, cy), (255, 255, 255), 1)
    cv2.line(img, (cx + 10, cy), (cx + 20, cy), (255, 255, 255), 1)
    cv2.line(img, (cx, cy - 20), (cx, cy - 10), (255, 255, 255), 1)
    cv2.line(img, (cx, cy + 10), (cx, cy + 20), (255, 255, 255), 1)
        
    # 3. Top Glass Panel (System Status & Depth)
    img = draw_glass_panel(img, 20, 20, W-20, 80, alpha=0.5)
    
    cv2.putText(img, "NavSystem 2.0", (40, 55), font, 1.0, (255, 220, 100), 2)
    
    depth = state["depth_buckets"]
    depth_txt = f"DEPTH | L: {depth['left']}  C: {depth['center']}  R: {depth['right']}"
    cv2.putText(img, depth_txt, (W//2 - 220, 55), font, 0.8, (255, 255, 255), 1)
    
    # Hazard Pulsing
    if state["hazard"]:
        pulse = (math.sin(t * 8) + 1) / 2  # 0 to 1
        red_val = int(100 + 155 * pulse)
        cv2.putText(img, state["hazard"], (W - 350, 55), font, 0.9, (50, 50, red_val), 2)

    # --- Live Location Row (DINOv2) ---
    img = draw_glass_panel(img, 20, 85, W-20, 125, alpha=0.45, blur=11)
    node_loc_txt = f"LOCATION: {state['current_node_name']}   [->  {goal_name}]"
    cv2.putText(img, node_loc_txt, (40, 112), font, 0.8, (100, 255, 200), 1)
    # Small indicator showing localization is live
    dot_color = (0, 255, 100) if localizer else (120, 120, 120)
    cv2.circle(img, (W - 50, 105), 6, dot_color, -1)
    loc_label = "LIVE" if localizer else "FIXED"
    cv2.putText(img, loc_label, (W - 90, 112), font, 0.6, dot_color, 1)

    # 4. Bottom Glass Panel (VLM Instruction)
    img = draw_glass_panel(img, 20, H - 140, W-20, H-20, alpha=0.6)
    
    # Animated Status Indicator
    if state["vlm_thinking"]:
        dot_count = int(t * 3) % 4
        dots = "." * dot_count
        cv2.circle(img, (50, H - 100), 8, (0, 165, 255), -1)
        cv2.putText(img, f"VLM ANALYZING{dots}", (70, H - 93), font, 0.7, (0, 165, 255), 1)
    else:
        cv2.circle(img, (50, H - 100), 8, (0, 255, 0), -1)
        cv2.putText(img, "VLM ACTIVE", (70, H - 93), font, 0.7, (0, 255, 0), 1)

    # Wrap Instruction Text
    words = state["instruction"].split(' ')
    lines, current_line = [], ""
    for w in words:
        if len(current_line) + len(w) < 80: current_line += w + " "
        else:
            lines.append(current_line)
            current_line = w + " "
    if current_line: lines.append(current_line)
    
    y = H - 55
    for line in lines[:2]: # Show up to 2 lines
        cv2.putText(img, line.strip(), (50, y), font, 0.85, (255, 255, 255), 2)
        y += 35

    return img

logger.info("Starting Video Playback... Press ESC in the window to stop.")
if IS_LIVE_CAM:
    speak("Live camera started. Navigation system is ready.")
else:
    speak("Visualizer started.")
time.sleep(1)

_dropped_frames = 0   # Track consecutive dropped frames for live-camera reconnect

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        if IS_LIVE_CAM:
            # Live camera: attempt reconnection (phone disconnected, etc.)
            _dropped_frames += 1
            if _dropped_frames % 30 == 1:
                logger.warning(f"[CAM] Frame drop #{_dropped_frames} — reconnecting to {VIDEO_SRC}...")
                cap.release()
                cap = cv2.VideoCapture(VIDEO_SRC)
            time.sleep(0.05)
            continue
        else:
            logger.info("Reached end of video.")
            break
    _dropped_frames = 0

    state["frame"] = frame

    # Scale to display resolution
    display = cv2.resize(frame, (W, H))

    try:
        display = draw_ui(display)
    except Exception as e:
        logger.error(f"UI drawing error: {e}")

    # ── ARRIVAL OVERLAY ──────────────────────────────────────────────────────
    if state["goal_reached"]:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 180, 0), -1)
        display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)
        # Big centred text
        msg1 = "YOU HAVE ARRIVED!"
        msg2 = goal_name
        (tw1, th1), _ = cv2.getTextSize(msg1, font, 2.2, 3)
        (tw2, th2), _ = cv2.getTextSize(msg2, font, 1.1, 2)
        cv2.putText(display, msg1, ((W-tw1)//2, H//2 - 30), font, 2.2, (255,255,255), 3)
        cv2.putText(display, msg2, ((W-tw2)//2, H//2 + 50), font, 1.1, (200,255,200), 2)
        cv2.putText(display, "Press ESC to exit", (W//2 - 130, H//2 + 120), font, 0.8, (200,255,200), 1)

    cv2.imshow("Nav System Visualizer", display)

    # ESC to quit
    if cv2.waitKey(delay) & 0xFF == 27:
        break

state["running"] = False
cap.release()
cv2.destroyAllWindows()
logger.info("Visualization ended gracefully.")
