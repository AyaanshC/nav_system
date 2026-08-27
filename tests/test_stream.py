"""
tests/test_stream.py

Test 1: Phone stream connectivity and frame rate.

Usage:
    python tests/test_stream.py --url http://192.168.1.5:8080/video

Expected: Shows live frame in OpenCV window, FPS printed to console.
Press 'q' to quit.
"""

import cv2
import argparse
import time
import sys

parser = argparse.ArgumentParser(description="Test phone stream connectivity.")
parser.add_argument("--url", required=True, help="IP Webcam URL")
parser.add_argument("--frames", type=int, default=60, help="Number of frames to capture (default: 60)")
args = parser.parse_args()

print(f"Connecting to: {args.url}")
cap = cv2.VideoCapture(args.url)

if not cap.isOpened():
    print(f"FAIL: Cannot open stream at {args.url}")
    print("Tips:")
    print("  - Open IP Webcam app on your phone and tap 'Start server'")
    print("  - Check the IP address shown in the app matches this URL")
    print("  - Ensure phone and laptop are on the same WiFi network")
    sys.exit(1)

print("OK: Stream opened. Capturing frames...")
t0 = time.time()
count = 0

while count < args.frames:
    ret, frame = cap.read()
    if not ret:
        print(f"WARN: Frame read failed at frame {count}. Stream dropped.")
        break
    count += 1

    # Overlay FPS on frame
    elapsed = time.time() - t0
    fps = count / max(elapsed, 0.001)
    cv2.putText(frame, f"Frame {count}  |  {fps:.1f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

    cv2.imshow("Stream Test — press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit by user.")
        break

fps = count / max(time.time() - t0, 0.001)
cap.release()
cv2.destroyAllWindows()

print(f"\n=== Stream Test Results ===")
print(f"Frames captured : {count}")
print(f"Average FPS     : {fps:.1f}")
print(f"Resolution      : {frame.shape[1]}x{frame.shape[0]}" if count > 0 else "")
print("PASS" if count > 0 else "FAIL")
