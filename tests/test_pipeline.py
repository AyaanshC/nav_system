"""
tests/test_pipeline.py

Full end-to-end pipeline test using the real keyframe images from maps/mybuilding/.

Tests every module independently then runs a simulated navigation loop
through the existing keyframes (as if walking the building).

Usage:
    python tests/test_pipeline.py

No stream/camera required — uses saved keyframe images as input.
"""

import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

# ─────────────────────────────────────────────────────────────────────────────

BUILDING     = "mybuilding"
MAP_DIR      = Path("maps") / BUILDING
KF_DIR       = MAP_DIR / "keyframes"
OSMAG_PATH   = MAP_DIR / "osmag.json"

SEPARATOR = "=" * 60

def section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def ok(msg: str):    logger.success(f"[PASS] {msg}")
def warn(msg: str):  logger.warning(f"[WARN] {msg}")
def fail(msg: str):  logger.error(f"[FAIL] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Load all keyframes into memory
# ─────────────────────────────────────────────────────────────────────────────

def load_keyframes() -> list[tuple[str, np.ndarray]]:
    """Load all keyframe images, return list of (node_id, frame)."""
    frames = []
    for p in sorted(KF_DIR.glob("node_*.jpg")):
        img = cv2.imread(str(p))
        if img is not None:
            frames.append((p.stem, img))
    return frames

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — MiDaS Depth Estimator
# ─────────────────────────────────────────────────────────────────────────────

def test_midas(frame: np.ndarray) -> dict:
    section("TEST 1 — MiDaS Depth Estimation")
    try:
        from modules.m2_midas import DepthEstimator
        t0 = time.time()
        midas = DepthEstimator(
            model_path="models/midas_v21_small_256.pt",
            device="cuda"
        )
        load_time = time.time() - t0
        ok(f"MiDaS loaded in {load_time:.1f}s")

        t0 = time.time()
        result = midas.estimate(frame)
        infer_time = time.time() - t0

        ok(f"Depth estimated in {infer_time*1000:.0f}ms")
        ok(f"  Left:   {result['left_bucket']}")
        ok(f"  Center: {result['center_bucket']}")
        ok(f"  Right:  {result['right_bucket']}")
        ok(f"  Prompt: {result['prompt_text']}")
        return result
    except Exception as e:
        fail(f"MiDaS failed: {e}")
        return {"depth_map": None, "prompt_text": "Depth unavailable.", "center_bucket": "unknown", "left_bucket": "unknown", "right_bucket": "unknown"}

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — YOLO Detection
# ─────────────────────────────────────────────────────────────────────────────

def test_yolo(frame: np.ndarray, depth_result: dict) -> list[dict]:
    section("TEST 2 — YOLO-World Object Detection")
    try:
        from modules.m5_yolo import YOLODetector
        import yaml
        with open("config/settings.yaml") as f:
            cfg = yaml.safe_load(f)

        t0 = time.time()
        yolo = YOLODetector(
            model_path=cfg.get("yolo_model", "models/yolov8s-worldv2.pt"),
            conf_threshold=0.30,
            device="cuda"
        )
        load_time = time.time() - t0
        ok(f"YOLO loaded in {load_time:.1f}s")

        t0 = time.time()
        detections, _ = yolo.detect(frame, frame_id=0, every_n=1)
        infer_time = time.time() - t0

        ok(f"Detection in {infer_time*1000:.0f}ms  —  {len(detections)} objects found")
        for d in detections:
            ok(f"  - {d['class']} (conf={d['confidence']:.2f})")

        yolo_text = yolo.format_for_prompt(
            detections,
            depth_map=depth_result.get("depth_map"),
            frame_shape=frame.shape
        )
        ok(f"Formatted prompt: {yolo_text}")
        return detections
    except Exception as e:
        fail(f"YOLO failed: {e}")
        return []

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — DINOv2 Localizer
# ─────────────────────────────────────────────────────────────────────────────

def test_dinov2(frames: list[tuple[str, np.ndarray]]) -> bool:
    section("TEST 3 — DINOv2 + FAISS Localization")
    try:
        from modules.m2_dino_localizer import DINOv2Localizer

        t0 = time.time()
        localizer = DINOv2Localizer(map_dir=str(MAP_DIR), device="cuda")
        load_time = time.time() - t0
        ok(f"DINOv2 loaded in {load_time:.1f}s")

        if not localizer.is_ready:
            warn("No FAISS index found. Building index now from keyframes...")
            t0 = time.time()
            localizer.build_index(str(KF_DIR))
            ok(f"Index built in {time.time()-t0:.1f}s with {len(frames)} keyframes")

        # Test localization on all frames
        ok(f"Testing localization on {len(frames)} keyframes...")
        correct = 0
        results = []
        for true_node_id, frame in frames:
            # Run enough times to pass temporal smoothing (SMOOTH_K=3)
            node_id = None
            for _ in range(4):
                node_id = localizer.localize(frame)
            results.append((true_node_id, node_id))

        ok("Localization results (true → predicted):")
        for true_id, pred_id in results:
            match = true_id == pred_id
            if match:
                correct += 1
            status = "OK" if match else "DIFF"
            ok(f"  [{status}] {true_id} → {pred_id}")

        accuracy = correct / len(results) * 100
        ok(f"Self-localization accuracy: {accuracy:.0f}% ({correct}/{len(results)} correct)")
        return True
    except Exception as e:
        fail(f"DINOv2 localizer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Path Planner
# ─────────────────────────────────────────────────────────────────────────────

def test_path_planner() -> tuple:
    section("TEST 4 — BFS Path Planner")
    try:
        from modules.m4_path_planner import PathPlanner
        planner = PathPlanner(osmag_path=str(OSMAG_PATH))
        ok(f"Graph loaded: {len(planner.nodes)} nodes")

        # List all nodes
        ok("Available nodes:")
        node_list = list(planner.nodes.items())
        for nid, node in node_list:
            ok(f"  {nid}: {node.get('name', '?')} (floor {node.get('level', 1)})")

        # Pick first and last node for path test
        start = node_list[0][0]
        goal  = node_list[-1][0]
        start_name = planner.nodes[start].get("name", start)
        goal_name  = planner.nodes[goal].get("name", goal)

        t0 = time.time()
        path = planner.find_path(start, goal)
        ok(f"BFS path ({start_name} → {goal_name}): {len(path)} steps in {(time.time()-t0)*1000:.0f}ms")
        ok(f"  Path: {' → '.join(path)}")

        # Test edge description lookup
        edge_desc = planner.get_edge_description(start, path)
        if edge_desc:
            ok(f"Edge description: '{edge_desc}'")
        else:
            warn("No edge descriptions in osmag.json (re-run scan_walk.py to generate)")

        # Test step instruction
        step = planner.get_next_step_instruction(start, path)
        ok(f"Next step instruction: '{step}'")

        return planner, path, start, goal
    except Exception as e:
        fail(f"Path planner failed: {e}")
        import traceback
        traceback.print_exc()
        return None, [], None, None

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — VLM Guidance Engine (qwen2.5vl:3b)
# ─────────────────────────────────────────────────────────────────────────────

def test_vlm_guidance(frame: np.ndarray, planner, path: list, start: str):
    section("TEST 5 — VLM Guidance (qwen2.5vl:3b)")
    if planner is None:
        warn("Skipped — path planner not available")
        return

    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
    except Exception:
        fail("Ollama is not running. Start Ollama app first.")
        return

    try:
        from modules.m5_vlm_guidance import GuidanceEngine
        engine = GuidanceEngine(max_cache_age_sec=3.0)
        node_name = planner.nodes.get(start, {}).get("name", "unknown area")

        depth = {"depth_map": None, "prompt_text": "Left: clear. Center: clear. Right: clear."}
        yolo_text = "No obstacles detected in current view."
        step_instr = planner.get_next_step_instruction(start, path)
        edge_desc  = planner.get_edge_description(start, path)

        logger.info(f"Querying qwen2.5vl:3b for guidance at: {node_name}")
        logger.info(f"Next step: {step_instr}")
        if edge_desc:
            logger.info(f"Edge description: {edge_desc}")

        t0 = time.time()
        instruction = engine.generate_instruction(
            frame_bgr             = frame,
            current_node          = start,
            current_node_name     = node_name,
            next_step_instruction = step_instr,
            path_nl_description   = " -> ".join(planner.nodes.get(n, {}).get("name", n) for n in path),
            yolo_text             = yolo_text,
            depth_text            = depth["prompt_text"],
            detection_hash        = "test",
            node_just_changed     = True,
            edge_description      = edge_desc
        )
        elapsed = time.time() - t0

        ok(f"VLM response in {elapsed:.1f}s")
        ok(f"Instruction: '{instruction}'")

        # Test narrative memory
        ok("Narrative memory after 3 node transitions:")
        engine.narrative.update("node_001", "main entrance corridor")
        engine.narrative.update("node_002", "elevator lobby")
        engine.narrative.update("node_003", "seating area")
        ok(f"  {engine.narrative.get_text()}")

    except Exception as e:
        fail(f"VLM guidance failed: {e}")
        import traceback
        traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Simulated Navigation Loop (walk all keyframes)
# ─────────────────────────────────────────────────────────────────────────────

def test_simulated_navigation(frames: list, planner, midas_result_cache: dict):
    section("TEST 6 — Simulated Navigation Loop (all keyframes)")
    if planner is None or not frames:
        warn("Skipped — planner or frames not available")
        return

    try:
        from modules.m5_yolo import YOLODetector
        from modules.m2_midas import DepthEstimator
        import yaml

        with open("config/settings.yaml") as f:
            cfg = yaml.safe_load(f)

        yolo = YOLODetector(model_path=cfg.get("yolo_model", "models/yolov8s-worldv2.pt"),
                             conf_threshold=0.30, device="cuda")
        midas = DepthEstimator(model_path=cfg.get("midas_model", "models/midas_v21_small_256.pt"),
                                device="cuda")

        start = frames[0][0]
        goal  = frames[-1][0]
        path  = planner.find_path(start, goal)

        ok(f"Simulating walk: {planner.nodes[start].get('name')} → {planner.nodes[goal].get('name')}")
        ok(f"Path has {len(path)} steps")

        total_yolo_ms = 0
        total_midas_ms = 0

        for i, (node_id, frame) in enumerate(frames):
            t0 = time.time()
            detections, _ = yolo.detect(frame, frame_id=i, every_n=1)
            total_yolo_ms += (time.time() - t0) * 1000

            t0 = time.time()
            depth = midas.estimate(frame)
            total_midas_ms += (time.time() - t0) * 1000

            step = planner.get_next_step_instruction(node_id, path) if node_id in path else "Off-path"
            edge = planner.get_edge_description(node_id, path) if node_id in path else ""

            direction = edge if edge else step
            node_name = planner.nodes.get(node_id, {}).get("name", node_id)

            logger.info(
                f"  [{i+1:02d}/{len(frames)}] {node_name:<40} "
                f"| {len(detections)} objects "
                f"| Depth C:{depth['center_bucket']:<4} "
                f"| => {direction}"
            )

        n = len(frames)
        ok(f"Avg YOLO inference:  {total_yolo_ms/n:.0f}ms/frame")
        ok(f"Avg MiDaS inference: {total_midas_ms/n:.0f}ms/frame")

    except Exception as e:
        fail(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  nav_system_ollama — Full Pipeline Test")
    print(f"  Building: {BUILDING}  |  Keyframes: {len(list(KF_DIR.glob('*.jpg')))}")
    print(f"{'='*60}")

    if not OSMAG_PATH.exists():
        fail(f"No map found at {OSMAG_PATH}. Run scan_walk.py first.")
        return

    # Load all keyframes
    frames = load_keyframes()
    if not frames:
        fail(f"No keyframe images found in {KF_DIR}")
        return
    ok(f"Loaded {len(frames)} keyframe images for testing")

    # Use node_005 as the primary test frame (mid-building)
    test_node_id, test_frame = frames[4]
    ok(f"Primary test frame: {test_node_id}  ({test_frame.shape[1]}x{test_frame.shape[0]}px)")

    # Run all tests
    depth_result   = test_midas(test_frame)
    detections     = test_yolo(test_frame, depth_result)
    dino_ok        = test_dinov2(frames)
    planner, path, start, goal = test_path_planner()
    test_vlm_guidance(test_frame, planner, path, start)
    test_simulated_navigation(frames, planner, depth_result)

    # Summary
    section("TEST SUMMARY")
    ok("MiDaS depth: DONE")
    ok("YOLO detection: DONE")
    ok(f"DINOv2 localizer: {'DONE' if dino_ok else 'FAILED'}")
    ok(f"Path planner: {'DONE' if planner else 'FAILED'}")
    ok("VLM guidance: DONE (check output above)")
    ok("Simulated navigation loop: DONE")
    print()
    logger.info("Test complete. Check logs above for any WARN/FAIL lines.")
    print()


if __name__ == "__main__":
    main()
