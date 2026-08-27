"""
modules/m5_yolo.py

YOLO-World open-vocabulary detector.
- Runs on GPU (RTX 4060) at FP16
- Called every N frames (default N=5)
- Delta cache: only triggers VLM re-query when detected object set changes
- format_for_prompt() accepts optional depth_map for per-object distance estimates
"""

import torch
import numpy as np
import hashlib
from ultralytics import YOLO
from loguru import logger


# 32-item obstacle vocabulary for visually impaired indoor navigation
# Covers all common collision hazards a blind person might encounter
OBSTACLE_VOCAB = [
    "door", "door handle", "stairs", "step", "chair",
    "table", "desk", "pillar", "column", "wall",
    "person", "wheelchair", "shopping cart", "bicycle", "scooter",
    "bag", "luggage", "box", "trash can", "fire extinguisher",
    "sign", "board", "bench", "counter", "railing",
    # Added: critical indoor hazards
    "wet floor sign", "cable", "escalator", "elevator door",
    "dog", "cat", "speed bump"
]


class YOLODetector:
    """
    YOLO-World open-vocabulary object detector.
    Runs FP16 on CUDA for maximum throughput.
    Uses frame-skip and delta caching to minimize unnecessary VLM calls.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        device: str = "cuda"
    ):
        """
        Args:
            model_path:      Path to yolov8s-worldv2.pt (small variant for speed)
            conf_threshold:  Minimum confidence to report a detection
            device:          "cuda" or "cpu"
        """
        self.conf = conf_threshold
        self.device = device if torch.cuda.is_available() else "cpu"
        self._last_hash = ""
        self._last_results: list[dict] = []

        logger.info(f"[YOLO] Loading {model_path} on {self.device}")
        self.model = YOLO(model_path)
        self.model.set_classes(OBSTACLE_VOCAB)

        # Warm up GPU with a dummy frame (avoids first-frame latency spike)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False, device=self.device, half=(self.device == "cuda"))
        logger.info("[YOLO] Model loaded and warmed up.")

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        every_n: int = 5
    ) -> tuple[list[dict], bool]:
        """
        Run detection on frame. Returns (detections, changed).
        Skips inference if frame_id % every_n != 0.

        Args:
            frame_bgr: OpenCV BGR frame
            frame_id:  Monotonically increasing frame number from PhoneStream
            every_n:   Run inference every N frames

        Returns:
            detections: list of {\"class\": str, \"confidence\": float, \"bbox\": [x1,y1,x2,y2]}
            changed:    True if detection set differs from last run (triggers VLM re-query)
        """
        if frame_id % every_n != 0:
            return self._last_results, False

        use_half = self.device == "cuda"
        results = self.model(
            frame_bgr,
            verbose=False,
            device=self.device,
            half=use_half,
            conf=self.conf,
            iou=0.45
        )

        detections: list[dict] = []
        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                cls_name = OBSTACLE_VOCAB[cls_idx] if cls_idx < len(OBSTACLE_VOCAB) else "unknown"
                conf_val = float(box.conf[0])
                # Filter out low-confidence detections — they pollute the VLM prompt
                if conf_val < 0.45:
                    continue
                detections.append({
                    "class":      cls_name,
                    "confidence": conf_val,
                    "bbox":       [round(v, 1) for v in box.xyxy[0].tolist()]
                })

        # Delta check — hash the sorted class names only
        # (bounding box positions change constantly even for static objects)
        new_hash = hashlib.md5(
            ",".join(sorted(d["class"] for d in detections)).encode()
        ).hexdigest()

        changed = new_hash != self._last_hash
        self._last_hash = new_hash
        self._last_results = detections

        if detections:
            logger.debug(f"[YOLO] Frame {frame_id}: {[d['class'] for d in detections]}")

        return detections, changed

    def format_for_prompt(
        self,
        detections: list[dict],
        depth_map: np.ndarray | None = None,
        frame_shape: tuple | None = None
    ) -> str:
        """
        Convert detections to concise text for VLM prompt.
        If depth_map is provided, estimates per-object distance using the
        MiDaS depth at the center of each bounding box.

        Args:
            detections:  List of detection dicts from detect()
            depth_map:   Optional MiDaS depth output (H x W float32, normalized 0-1)
            frame_shape: (H, W) of original frame (needed to scale bboxes to depth_map)

        Returns:
            e.g. "Detected objects: door (1.2m, very close), chair (2.8m, ahead)."
        """
        if not detections:
            return "No obstacles detected in current view."

        parts = []
        for d in detections:
            label = d["class"]

            if depth_map is not None and frame_shape is not None:
                # Sample depth at bbox center, scaled to depth_map resolution
                fh, fw = frame_shape[:2]
                dh, dw = depth_map.shape[:2]
                x1, y1, x2, y2 = d["bbox"]
                cx = int(((x1 + x2) / 2) / fw * dw)
                cy = int(((y1 + y2) / 2) / fh * dh)
                cx = max(0, min(cx, dw - 1))
                cy = max(0, min(cy, dh - 1))
                depth_val = float(depth_map[cy, cx])

                # Inverse depth: higher = closer
                if depth_val > 0.72:
                    dist_label = "very close (<1.5m)"
                elif depth_val > 0.45:
                    dist_label = "ahead (1.5-3m)"
                else:
                    dist_label = "far (>3m)"
                parts.append(f"{label} [{dist_label}]")
            else:
                conf = d["confidence"]
                conf_label = "high" if conf > 0.7 else "medium"
                parts.append(f"{label} ({conf_label} confidence)")

        return "Detected objects: " + ", ".join(parts) + "."

    @property
    def last_results(self) -> list[dict]:
        """Most recent detection results (may be from a cached frame)."""
        return self._last_results
