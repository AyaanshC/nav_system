import cv2
import sys

def test_webcam_indices():
    print("Testing local webcam indices (looking for Iriun Webcam)...\n")
    found_any = False
    
    # Check indices 0 through 4
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[OK] Found working camera at index: {i}")
                print(f"   Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                found_any = True
            cap.release()
            
    if not found_any:
        print("[FAIL] No local cameras found. Make sure Iriun Webcam is running and connected to your phone.")
    else:
        print("\nIf Iriun Webcam is your only camera, it is likely index 0.")
        print("You can use this index (e.g. '0') in place of the IP address stream URL.")

if __name__ == "__main__":
    test_webcam_indices()
