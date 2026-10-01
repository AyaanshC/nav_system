"""
tests/test_with_video.py

Offline video-based system test for nav_system_ollama.
Runs YOLO, Depth Anything V2, image enhancement and VLM guidance
on every frame of a local video file. No phone or WiFi needed.

Usage:
    python tests/test_with_video.py
    python tests/test_with_video.py --video test_videos/indoor_walk.f160.mp4
    python tests/test_with_video.py --max_frames 300 --no_tts
    python tests/test_with_video.py --speed 2.0 --node node_003
"""

import argparse, time, sys, cv2, numpy as np, hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")

parser = argparse.ArgumentParser()
parser.add_argument("--video",       default="test_videos/indoor_walk.f160.mp4")
parser.add_argument("--building",    default="mybuilding")
parser.add_argument("--node",        default="node_001")
parser.add_argument("--goal",        default="node_013")
parser.add_argument("--speed",       type=float, default=1.0)
parser.add_argument("--max_frames",  type=int,   default=0)
parser.add_argument("--no_tts",      action="store_true")
parser.add_argument("--yolo_every",  type=int,   default=5)
parser.add_argument("--depth_every", type=int,   default=5)
args = parser.parse_args()

video_path = Path(args.video)
if not video_path.exists():
    logger.error(f"Video not found: {video_path.resolve()}")
    sys.exit(1)

cap0 = cv2.VideoCapture(str(video_path))
total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap0.get(cv2.CAP_PROP_FPS) or 30.0
W     = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap0.release()

logger.info("="*60)
logger.info("  nav_system_ollama  —  Video Test Mode")
logger.info("="*60)
logger.info(f"  Video   : {video_path.name}  {W}x{H}  {fps:.0f}fps  {total/fps:.0f}s")
logger.info(f"  Building: {args.building}  |  Node: {args.node}  ->  {args.goal}")
logger.info(f"  TTS     : {'OFF' if args.no_tts else 'ON'}  |  Speed: {args.speed}x")
logger.info("="*60)

from modules.m4_path_planner   import PathPlanner
from modules.m5_yolo           import YOLODetector
from modules.m2_depth_anything import DepthEstimator
from modules.m5_vlm_guidance   import GuidanceEngine
from modules.m0_image_enhancer import ImageEnhancer
from modules.m8_sonification   import SonificationEngine
import yaml

if args.no_tts:
    def speak(t, rate=None): logger.info(f"[TTS] {t}")
else:
    from modules.m1_user_interaction import speak

osmag_path = Path("maps") / args.building / "osmag.json"
planner    = PathPlanner(osmag_path=str(osmag_path))
yolo       = YOLODetector("models/yolov8s-worldv2.pt", conf_threshold=0.35, device="cuda")
depth_est  = DepthEstimator(device="cuda")
guidance   = GuidanceEngine(max_cache_age_sec=3.0)
with open("config/settings.yaml") as f: cfg = yaml.safe_load(f)
enhancer   = ImageEnhancer(config=cfg)
sonifier   = SonificationEngine()

path_nodes   = planner.find_path(args.node, args.goal)
current_name = planner.nodes.get(args.node, {}).get("name", args.node)
goal_name    = planner.nodes.get(args.goal, {}).get("name", args.goal)
nl_desc      = planner.get_nl_description(args.node, args.goal, path_nodes)
logger.info(f"[Route] {current_name}  ->  {goal_name}  ({len(path_nodes)} nodes)")
speak(f"Video test started. Route: {current_name} to {goal_name}.")
time.sleep(1.5)

cap          = cv2.VideoCapture(str(video_path))
frame_id     = 0
max_f        = args.max_frames if args.max_frames > 0 else total
frame_delay  = (1.0 / fps) / args.speed
depth_result = {"depth_map": None, "prompt_text": "Depth initialising.", "center_bucket":"?","left_bucket":"?","right_bucket":"?"}
last_spoken  = 0.0
MIN_INTERVAL = 3.0
stats        = {"vlm":0,"yolo":0,"depth":0,"hazard":0,"instructions":[]}

logger.info(f"[Loop] Processing {min(max_f, total)} frames  (Ctrl+C to stop early)\n")

while True:
    ret, frame = cap.read()
    if not ret or frame_id >= max_f: break
    frame_id += 1

    if frame_id % args.depth_every == 0:
        depth_result = depth_est.estimate(frame)
        stats["depth"] += 1
        if depth_result["depth_map"] is not None:
            md = float(np.percentile(depth_result["depth_map"], 95))
            sonifier.set_danger_level(min(max((md-0.4)/0.45, 0.0), 1.0))

    detections, changed, hazard = yolo.detect(
        frame, frame_id, every_n=args.yolo_every,
        depth_map=depth_result.get("depth_map"), depth_estimator=depth_est)
    if frame_id % args.yolo_every == 0: stats["yolo"] += 1

    if hazard:
        close = [d for d in detections if d.get("direction")=="Center" and (d.get("distance_m") or 9) < 1.5]
        if close:
            c = min(close, key=lambda x: x["distance_m"])
            alert = f"Caution! {c['class']} {c['distance_m']:.1f}m ahead."
            if time.time()-last_spoken >= 2.0:
                logger.warning(f"  HAZARD: {alert}")
                speak(alert); last_spoken=time.time(); stats["hazard"]+=1

    now = time.time()
    if now - last_spoken >= MIN_INTERVAL:
        yolo_text = yolo.format_for_prompt(detections, depth_result.get("depth_map"), frame.shape)
        enh       = enhancer.enhance(frame)
        prev_instr = guidance._last_instruction  # save BEFORE call (bug fix)
        instr      = guidance.generate_instruction(
            frame_bgr=enh, current_node=args.node, current_node_name=current_name,
            next_step_instruction=planner.get_next_step_instruction(args.node, path_nodes),
            path_nl_description=nl_desc, yolo_text=yolo_text,
            depth_text=depth_result.get("prompt_text",""), detection_hash="",
            node_just_changed=False,
            edge_description=planner.get_edge_description(args.node, path_nodes))
        if instr:
            if instr != prev_instr:   # genuinely new instruction
                speak(instr); last_spoken=now
                stats["vlm"]+=1; stats["instructions"].append(instr)
                dets = ", ".join(f"{d['class']}@{d.get('direction','?')}({d.get('distance_m') or '?'}m)" for d in detections[:3]) or "none"
                print(f"\r  [{frame_id:5d}/{min(max_f,total)}]  Depth:{depth_result['center_bucket']}  YOLO:[{dets}]")
                print(f"           VLM: \"{instr[:80]}\"")
            else:
                # Cached repeat — update timer so we don't hammer the VLM
                last_spoken = now

    time.sleep(frame_delay)

cap.release()
print()
logger.info("="*60)
logger.info("  Test Complete")
logger.info(f"  Frames: {frame_id}  |  Depth runs: {stats['depth']}  |  YOLO runs: {stats['yolo']}")
logger.info(f"  VLM instructions: {stats['vlm']}  |  Hazard alerts: {stats['hazard']}")
logger.info("  Instructions:")
for i,s in enumerate(stats["instructions"],1): logger.info(f"    {i:2d}. {s}")
logger.info("="*60)
