"""
tests/test_metric_depth.py

Diagnostic and standalone test script for the Metric Depth Perception Module
using Video Depth Anything Small.

Usage:
    # Test on a dummy synthetic frame:
    python tests/test_metric_depth.py

    # Test on an image file:
    python tests/test_metric_depth.py --image path/to/image.jpg

    # Test on a live camera stream / phone IP webcam:
    python tests/test_metric_depth.py --stream http://192.168.1.5:8080/video
    python tests/test_metric_depth.py --stream 0   (local webcam)
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from modules.m2_metric_depth import MetricDepthEstimator


def main():
    parser = argparse.ArgumentParser(description="Test Metric Depth Estimation Module")
    parser.add_argument("--image", type=str, default="", help="Path to test image file")
    parser.add_argument("--stream", type=str, default="", help="Webcam index (e.g. 0) or IP camera URL")
    parser.add_argument("--model", type=str, default="models/metric_video_depth_anything_vits.pth", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--output", type=str, default="test_depth_output.jpg", help="Path to save colorized depth visualization")
    args = parser.parse_args()

    print("=" * 60)
    print("  Video Depth Anything (Metric Small) — Standalone Test")
    print("=" * 60)

    # 1. Initialize the Metric Depth Estimator
    logger.info(f"Initializing MetricDepthEstimator with weights: {args.model}")
    t0 = time.time()
    estimator = MetricDepthEstimator(
        model_path=args.model,
        device=args.device,
        collision_dist_m=0.6,
        warning_dist_m=1.8,
    )
    load_time = time.time() - t0

    if not estimator.is_ready:
        logger.error("[FAIL] MetricDepthEstimator could not be initialized.")
        logger.error("Ensure PyTorch is installed and the weights file exists at models/metric_video_depth_anything_vits.pth")
        sys.exit(1)

    logger.success(f"[PASS] Model loaded in {load_time:.2f}s on {estimator.device}")

    # 2. Source Selection (Live stream, Image file, or Synthetic frame)
    if args.stream:
        # Camera / Stream mode
        src = int(args.stream) if args.stream.isdigit() else args.stream
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error(f"Cannot open camera stream: {args.stream}")
            sys.exit(1)

        logger.info(f"Connected to stream: {args.stream}. Press 'q' in window to exit.")
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                t_start = time.time()
                res = estimator.estimate(frame)
                dt = (time.time() - t_start) * 1000

                # Colorize for visualization
                color_depth = estimator.colorize_depth(res["depth_map"])
                combined = np.hstack((cv2.resize(frame, (320, 240)), cv2.resize(color_depth, (320, 240))))

                # Display stats on image
                info_text = f"{dt:.0f}ms | Min: {res['min_distance_m']}m | Danger: {res['danger_level']}"
                cv2.putText(combined, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow("Metric Depth (Left: RGB, Right: Depth in Meters)", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if frame_idx % 10 == 0:
                    logger.info(f"Frame {frame_idx:4d} | {info_text} | {res['prompt_text']}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    else:
        # Single image or synthetic frame mode
        if args.image and Path(args.image).exists():
            logger.info(f"Loading image from: {args.image}")
            frame = cv2.imread(args.image)
        else:
            # Check if any existing keyframe exists in maps
            keyframes = list(Path("maps").glob("*/keyframes/node_*.jpg")) if Path("maps").exists() else []
            if keyframes:
                test_kf = str(keyframes[0])
                logger.info(f"Using existing keyframe image: {test_kf}")
                frame = cv2.imread(test_kf)
            else:
                logger.info("Generating synthetic test frame (640x480)...")
                frame = np.full((480, 640, 3), 120, dtype=np.uint8)
                # Draw a mock obstacle in the center
                cv2.rectangle(frame, (260, 180), (380, 420), (40, 40, 200), -1)
                cv2.putText(frame, "Obstacle", (270, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 3. Run Inference
        logger.info("Running metric depth estimation...")
        t_infer = time.time()
        result = estimator.estimate(frame)
        elapsed_ms = (time.time() - t_infer) * 1000

        # 4. Display Results
        print("\n" + "-" * 60)
        print("  INFERENCE RESULTS (Physical Ground-Truth in Meters)")
        print("-" * 60)
        print(f"  Latency:          {elapsed_ms:.1f} ms")
        print(f"  Min Distance:     {result['min_distance_m']} meters")
        print(f"  Center Corridor:  {result['center_distance_m']} meters ({result['center_bucket']})")
        print(f"  Left Corridor:    {result['left_distance_m']} meters ({result['left_bucket']})")
        print(f"  Right Corridor:   {result['right_distance_m']} meters ({result['right_bucket']})")
        print(f"  Danger Level:     {result['danger_level']} (0.0=safe, 1.0=imminent collision)")
        print(f"  VLM Prompt Text:  {result['prompt_text']}")
        print("-" * 60)

        # 5. Save Visualization
        vis = estimator.colorize_depth(result["depth_map"])
        cv2.imwrite(args.output, vis)
        logger.success(f"[PASS] Colorized metric depth visualization saved to: {args.output}")


if __name__ == "__main__":
    main()
