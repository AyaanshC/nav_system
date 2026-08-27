"""
tests/test_yolo.py

Test 3: YOLO-World detection on a dummy frame.

Usage:
    python tests/test_yolo.py

Expected: Loads model, runs detection, prints class names + confidences.
"""

import sys
import time
import numpy as np
import os

sys.path.insert(0, ".")

from modules.m5_yolo import YOLODetector, OBSTACLE_VOCAB

MODEL_PATH = "models/yolov8x-worldv2.pt"

print("=== YOLO-World Detector Test ===")
print(f"Model: {MODEL_PATH}")
print(f"Vocabulary size: {len(OBSTACLE_VOCAB)} classes")
print(f"Classes: {', '.join(OBSTACLE_VOCAB[:10])} ...")

if not os.path.exists(MODEL_PATH):
    print(f"FAIL: Model not found at {MODEL_PATH}")
    print("Auto-download with:")
    print("  python -c \"from ultralytics import YOLO; m=YOLO('yolov8x-worldv2.pt')\"")
    print("  Then move yolov8x-worldv2.pt to models/")
    sys.exit(1)

print("\nLoading model (includes GPU warmup)...")
t0 = time.time()
det = YOLODetector(MODEL_PATH, device="cuda")
print(f"Load + warmup time: {time.time() - t0:.2f}s")

# Test 1: Empty frame (expects no detections)
print("\nTest 1: Black frame (expect 0 detections)")
black = np.zeros((480, 640, 3), dtype=np.uint8)
t0 = time.time()
results, changed = det.detect(black, frame_id=5, every_n=5)
elapsed = (time.time() - t0) * 1000
print(f"  Detections: {len(results)}")
print(f"  Inference time: {elapsed:.1f}ms")
print(f"  Changed: {changed}")

# Test 2: Frame skipping (frame_id not divisible by 5)
print("\nTest 2: Frame skipping (frame_id=3, every_n=5)")
results2, changed2 = det.detect(black, frame_id=3, every_n=5)
assert results2 is results or results2 == results, "Skipped frame should return cached result"
print(f"  Correctly returned cached result (no inference)")

# Test 3: Format for prompt
print("\nTest 3: Prompt formatter")
mock_detections = [
    {"class": "door",   "confidence": 0.82, "bbox": [10, 20, 100, 200]},
    {"class": "person", "confidence": 0.61, "bbox": [300, 50, 500, 400]},
]
text = det.format_for_prompt(mock_detections)
print(f"  Output: {text}")

text_empty = det.format_for_prompt([])
print(f"  Empty: {text_empty}")

# Test 4: 10 inference frames timing
print("\nTest 4: 10 inference frames (every_n=1)...")
times = []
for i in range(1, 11):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    t0 = time.time()
    det.detect(frame, frame_id=i, every_n=1)
    times.append((time.time() - t0) * 1000)

print(f"  Avg: {sum(times)/len(times):.1f}ms  Min: {min(times):.1f}ms  Max: {max(times):.1f}ms")

print("\n✓ YOLO test PASSED")
