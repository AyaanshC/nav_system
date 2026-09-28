"""
test_webcam_depth.py

Simple 1-click standalone script to test Metric Depth Perception on your laptop camera.

Usage:
    python test_webcam_depth.py

Controls:
    Press 'q' in the camera window to exit.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from modules.m2_metric_depth import MetricDepthEstimator


def run_webcam_test():
    print("=" * 60)
    print("  1-CLICK LAPTOP WEBCAM TEST — Metric Depth Perception")
    print("=" * 60)

    model_path = "models/metric_video_depth_anything_vits.pth"
    
    # 1. Initialize Depth Estimator
    logger.info("Loading Video-Depth-Anything Metric Model...")
    estimator = MetricDepthEstimator(
        model_path=model_path,
        device="cuda",
        collision_dist_m=0.6,
        warning_dist_m=1.8
    )

    if not estimator.is_ready:
        logger.error("Could not load depth model. Ensure weights exist in models/.")
        return

    # 2. Connect to Laptop Camera (index 0 with DirectShow for instant connection)
    logger.info("Opening laptop webcam (camera 0)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        logger.warning("Camera index 0 failed, trying default backend...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Could not open laptop webcam. Please ensure camera permissions are enabled.")
        return

    logger.success("Webcam connected! Press 'q' in the window to quit.")

    frame_count = 0
    w_disp, h_disp = 480, 360

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed.")
                break

            frame_count += 1
            t_start = time.time()
            
            # Run metric depth estimation (outputs real meters)
            res = estimator.estimate(frame)
            latency_ms = (time.time() - t_start) * 1000

            # Generate color heatmap
            depth_vis = estimator.colorize_depth(res["depth_map"])

            # Resize both to 480x360 and place side-by-side
            rgb_small = cv2.resize(frame, (w_disp, h_disp))
            depth_small = cv2.resize(depth_vis, (w_disp, h_disp))
            combined = np.hstack((rgb_small, depth_small))

            # Top HUD Banner
            cv2.rectangle(combined, (0, 0), (w_disp * 2, 60), (20, 20, 20), -1)
            
            # Line 1: Overall status & Closest hazard
            line1 = f"Latency: {latency_ms:.0f}ms | Closest Hazard: {res['min_distance_m']}m | Danger: {res['danger_level']}"
            # Line 2: Corridor breakdown in meters
            line2 = f"Left: {res['left_distance_m']}m | Center: {res['center_distance_m']}m | Right: {res['right_distance_m']}m"

            cv2.putText(combined, line1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(combined, line2, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)

            # Display the interactive window
            cv2.imshow("Metric Depth Perception (Left: Laptop Camera | Right: Depth in Meters)", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_count % 20 == 0:
                logger.info(f"Frame {frame_count:4d} | Nearest: {res['min_distance_m']}m | Center: {res['center_distance_m']}m | Danger: {res['danger_level']}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam test closed.")


if __name__ == "__main__":
    run_webcam_test()
