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

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="test_videos/house_tour.mp4", help="Path to test video")
parser.add_argument("--building", default="mybuilding")
parser.add_argument("--node", default="node_001")
parser.add_argument("--goal", default="node_013")
parser.add_argument("--no_tts", action="store_true", help="Disable text-to-speech audio")
args = parser.parse_args()

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

if args.no_tts:
    def speak(t, rate=None): pass
else:
    from modules.m1_user_interaction import speak

path_nodes = planner.find_path(args.node, args.goal)
current_name = planner.nodes.get(args.node, {}).get("name", args.node)
nl_desc = planner.get_nl_description(args.node, args.goal, path_nodes)

# --- Shared State ---
state = {
    "frame": None,
    "detections": [],
    "depth_buckets": {"left": "?", "center": "?", "right": "?"},
    "depth_map": None,
    "instruction": "Waiting for visual context...",
    "hazard": None,
    "running": True,
    "vlm_thinking": False
}

def fast_worker():
    """Runs Depth and YOLO continuously to keep UI bounding boxes smooth."""
    global state
    frame_count = 0
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
        
        # Update State safely
        state["depth_map"] = d_map
        state["depth_buckets"] = {
            "left": depth_result.get("left_bucket", "?"),
            "center": depth_result.get("center_bucket", "?"),
            "right": depth_result.get("right_bucket", "?")
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
            
        # Yield to let UI breathe
        time.sleep(0.02) 

def slow_worker():
    """Runs VLM inference whenever it's not busy."""
    global state
    last_spoken = 0
    MIN_INTERVAL = 4.0
    
    while state["running"]:
        if state["frame"] is None or state["depth_map"] is None:
            time.sleep(0.1)
            continue
            
        now = time.time()
        if now - last_spoken < MIN_INTERVAL:
            time.sleep(0.1)
            continue
            
        frame = state["frame"].copy()
        detections = list(state["detections"])
        d_map = state["depth_map"]
        
        state["vlm_thinking"] = True
        yolo_text = yolo.format_for_prompt(detections, d_map, frame.shape)
        enh = enhancer.enhance(frame)
        prev_instr = guidance._last_instruction
        
        instr = guidance.generate_instruction(
            frame_bgr=enh, current_node=args.node, current_node_name=current_name,
            next_step_instruction=planner.get_next_step_instruction(args.node, path_nodes),
            path_nl_description=nl_desc, yolo_text=yolo_text,
            depth_text=f"C:{state['depth_buckets']['center']}, L:{state['depth_buckets']['left']}, R:{state['depth_buckets']['right']}",
            detection_hash="", node_just_changed=False,
            edge_description=planner.get_edge_description(args.node, path_nodes)
        )
        state["vlm_thinking"] = False
        
        if instr and instr != prev_instr:
            state["instruction"] = instr
            speak(instr)
            
        last_spoken = time.time()

logger.info("Starting Background AI Threads...")
threading.Thread(target=fast_worker, daemon=True).start()
threading.Thread(target=slow_worker, daemon=True).start()

# --- Main Video UI Loop ---
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    logger.error(f"Could not open video {args.video}.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
delay = int(1000 / fps)
orig_W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
orig_H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

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
    cv2.putText(img, depth_txt, (W//2 - 200, 55), font, 0.8, (255, 255, 255), 1)
    
    # Hazard Pulsing
    if state["hazard"]:
        pulse = (math.sin(t * 8) + 1) / 2  # 0 to 1
        red_val = int(100 + 155 * pulse)
        cv2.putText(img, state["hazard"], (W - 350, 55), font, 0.9, (50, 50, red_val), 2)

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
speak("Visualizer started.")
time.sleep(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        logger.info("Reached end of video.")
        break
    
    state["frame"] = frame
    
    # Scale up test video to 720p for a beautiful UI
    display = cv2.resize(frame, (W, H))
    
    try:
        display = draw_ui(display)
    except Exception as e:
        logger.error(f"UI drawing error: {e}")
        pass
    
    cv2.imshow("Nav System Visualizer", display)
    
    # Play at native FPS (delay is in milliseconds)
    if cv2.waitKey(delay) & 0xFF == 27: # ESC key
        break

state["running"] = False
cap.release()
cv2.destroyAllWindows()
logger.info("Visualization ended gracefully.")
