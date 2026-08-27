"""
tests/test_midas.py

Test 2: MiDaS depth estimation on a random dummy frame.

Usage:
    python tests/test_midas.py

Expected: Loads model, runs inference, prints near/mid/far buckets.
"""

import sys
import time
import numpy as np
import cv2

sys.path.insert(0, ".")

from modules.m2_midas import DepthEstimator

MODEL_PATH = "models/midas_v21_small_256.pt"

print("=== MiDaS Depth Estimator Test ===")
print(f"Model: {MODEL_PATH}")

# Check model file exists
import os
if not os.path.exists(MODEL_PATH):
    print(f"FAIL: Model not found at {MODEL_PATH}")
    print("Download with:")
    print("  wget -O models/midas_v21_small_256.pt https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt")
    sys.exit(1)

print("Loading model...")
t0 = time.time()
est = DepthEstimator(MODEL_PATH, device="cuda")
print(f"Load time: {time.time() - t0:.2f}s")

# Test 1: Random noise frame
dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
print("\nTest 1: Random noise frame")
t0 = time.time()
result = est.estimate(dummy)
elapsed = (time.time() - t0) * 1000
print(f"  Left:   {result['left_bucket']}")
print(f"  Center: {result['center_bucket']}")
print(f"  Right:  {result['right_bucket']}")
print(f"  Prompt: {result['prompt_text']}")
print(f"  Inference time: {elapsed:.1f}ms")

# Test 2: Run 10 more frames (checks GPU stability)
print("\nTest 2: 10 warmup frames (checks GPU stability)...")
times = []
for i in range(10):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    t0 = time.time()
    est.estimate(frame)
    times.append((time.time() - t0) * 1000)

print(f"  Avg inference: {sum(times)/len(times):.1f}ms")
print(f"  Min: {min(times):.1f}ms  Max: {max(times):.1f}ms")

# Save depth visualization
print("\nSaving depth visualization to depth_vis.jpg...")
vis_frame = np.zeros((480, 640, 3), dtype=np.uint8)
vis_frame[:, :320] = 200   # bright on left (simulates nearby wall)
result = est.estimate(vis_frame)
depth_map = result["depth_map"]
depth_vis = (depth_map * 255).astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
cv2.imwrite("depth_vis.jpg", depth_colored)
print("  Saved depth_vis.jpg")

print("\n✓ MiDaS test PASSED")
