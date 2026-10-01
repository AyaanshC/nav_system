"""
modules/m5_yolo.py

YOLO-World open-vocabulary detector.
- Runs on GPU (RTX 4060) at FP16
- Called every N frames (default N=5)
- Delta cache: only triggers VLM re-query when detected object set changes
- format_for_prompt() uses Depth Anything V2 for per-object meter distances
- Immediate hazard flag: any center obstacle < 1.5m sets immediate_hazard=True

Changes vs original:
  detect() now returns (detections, changed, immediate_hazard) — 3-tuple.
  Each detection dict gains 'direction' and 'distance_m' fields.
"""

import torch
import numpy as np
import hashlib
from collections import deque
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
        self._last_hash    = ""
        self._last_results: list[dict] = []
        self._last_hazard:  bool       = False

        # ByteTrack history: track ID -> deque of (cx, cy, area)
        self._history = {}

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
        every_n: int = 5,
        depth_map: np.ndarray | None = None,
        depth_estimator=None
    ) -> tuple[list[dict], bool, bool]:
        """
        Run detection on frame. Returns (detections, changed, immediate_hazard).
        Skips inference if frame_id % every_n != 0.

        Args:
            frame_bgr:        OpenCV BGR frame
            frame_id:         Monotonically increasing frame number from PhoneStream
            every_n:          Run inference every N frames
            depth_map:        Optional depth map from DepthEstimator.estimate()
            depth_estimator:  Optional DepthEstimator instance for get_bounding_box_distance()

        Returns:
            detections:       list of dicts with class, confidence, bbox, motion,
                              direction (Left/Center/Right), distance_m (float)
            changed:          True if detection set differs from last run
            immediate_hazard: True if any center obstacle is estimated < 1.5m away
        """
        if frame_id % every_n != 0:
            return self._last_results, False, self._last_hazard

        use_half = self.device == "cuda"
        results = self.model.track(
            frame_bgr,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
            device=self.device,
            half=use_half,
            conf=self.conf,
            iou=0.45
        )

        H, W = frame_bgr.shape[:2]
        detections: list[dict] = []
        immediate_hazard = False

        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue

            for box in r.boxes:
                cls_idx  = int(box.cls[0])
                cls_name = OBSTACLE_VOCAB[cls_idx] if cls_idx < len(OBSTACLE_VOCAB) else "unknown"
                conf_val = float(box.conf[0])
                track_id = int(box.id[0])

                if conf_val < 0.45:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cx   = (x1 + x2) / 2
                cy   = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)

                # ── Horizontal direction (Left / Center / Right) ──────────────
                cx_norm = cx / W
                if cx_norm < 0.35:
                    direction = "Left"
                elif cx_norm <= 0.65:
                    direction = "Center"
                else:
                    direction = "Right"

                # ── Per-object distance in meters ─────────────────────────────
                # Uses Depth Anything V2's get_bounding_box_distance() when available.
                # Falls back to depth_map center-pixel sampling, then confidence.
                distance_m = None
                if depth_estimator is not None and depth_map is not None:
                    try:
                        distance_m = depth_estimator.get_bounding_box_distance(
                            [x1, y1, x2, y2], depth_map
                        )
                    except Exception:
                        distance_m = None

                if distance_m is None and depth_map is not None:
                    # Fallback: sample depth at bbox center
                    dh, dw = depth_map.shape[:2]
                    dcx = int(cx / W * dw)
                    dcy = int(cy / H * dh)
                    dcx = max(0, min(dcx, dw - 1))
                    dcy = max(0, min(dcy, dh - 1))
                    depth_val = float(depth_map[dcy, dcx])
                    distance_m = (1.0 - depth_val) * 5.0

                # ── Immediate hazard check ────────────────────────────────────
                if direction == "Center" and distance_m is not None and distance_m < 1.5:
                    immediate_hazard = True
                    logger.warning(
                        f"[YOLO] ⚠ IMMEDIATE HAZARD: {cls_name} "
                        f"{distance_m:.1f}m directly ahead!"
                    )

                # ── Tracking history + motion ──────────────────────────────────
                if track_id not in self._history:
                    self._history[track_id] = deque(maxlen=5)
                self._history[track_id].append((cx, cy, area))

                motion = "static"
                hist = self._history[track_id]
                if len(hist) >= 3:
                    old_cx, old_cy, old_area = hist[0]
                    dx     = cx   - old_cx
                    d_area = area - old_area
                    if d_area > (old_area * 0.15):
                        motion = "approaching"
                    elif dx < -20:
                        motion = "moving left"
                    elif dx > 20:
                        motion = "moving right"

                det = {
                    "class":      cls_name,
                    "confidence": conf_val,
                    "bbox":       [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "motion":     motion,
                    "direction":  direction,
                    "distance_m": round(distance_m, 2) if distance_m is not None else None,
                }
                detections.append(det)

        # Delta check (hash class names — bbox changes every frame)
        new_hash = hashlib.md5(
            ",".join(sorted(d["class"] for d in detections)).encode()
        ).hexdigest()

        changed = new_hash != self._last_hash
        self._last_hash    = new_hash
        self._last_results = detections
        self._last_hazard  = immediate_hazard

        if detections:
            logger.debug(f"[YOLO] Frame {frame_id}: {[d['class'] for d in detections]}")

        return detections, changed, immediate_hazard

    def format_for_prompt(
        self,
        detections: list[dict],
        depth_map: np.ndarray | None = None,
        frame_shape: tuple | None = None
    ) -> str:
        """
        Convert detections to concise text for VLM prompt.
        Prefers distance_m from detect() (Depth Anything V2 bbox distance).
        Falls back to depth_map center-pixel sampling if distance_m is missing.

        Args:
            detections:  List of detection dicts from detect()
            depth_map:   Depth map (H x W float32) — used only as fallback
            frame_shape: Original frame shape — used only as fallback

        Returns:
            e.g. "Detected objects: door [Center, 1.2m, very close], chair [Left, 2.8m, approaching]."
        """
        if not detections:
            return "No obstacles detected in current view."

        parts = []
        for d in detections:
            label     = d["class"]
            direction = d.get("direction", "")
            dist_m    = d.get("distance_m")   # Set by detect() via Depth Anything V2

            # Build distance label
            if dist_m is not None:
                if dist_m < 1.5:
                    dist_label = f"{dist_m:.1f}m, very close"
                elif dist_m < 3.0:
                    dist_label = f"{dist_m:.1f}m, ahead"
                else:
                    dist_label = f"{dist_m:.1f}m, far"
            elif depth_map is not None and frame_shape is not None:
                # Fallback: sample depth at bbox center
                fh, fw = frame_shape[:2]
                dh, dw = depth_map.shape[:2]
                x1, y1, x2, y2 = d["bbox"]
                cx = max(0, min(int(((x1 + x2) / 2) / fw * dw), dw - 1))
                cy = max(0, min(int(((y1 + y2) / 2) / fh * dh), dh - 1))
                depth_val = float(depth_map[cy, cx])
                if depth_val > 0.72:
                    dist_label = "very close (<1.5m)"
                elif depth_val > 0.45:
                    dist_label = "ahead (1.5-3m)"
                else:
                    dist_label = "far (>3m)"
            else:
                conf_label = "high" if d["confidence"] > 0.7 else "medium"
                dist_label = f"{conf_label} confidence"

            motion_str = f", {d['motion']}" if d.get("motion") != "static" else ""
            dir_str    = f"{direction}, " if direction else ""
            parts.append(f"{label} [{dir_str}{dist_label}{motion_str}]")

        return "Detected objects: " + ", ".join(parts) + "."

    @property
    def last_results(self) -> list[dict]:
        """Most recent detection results (may be from a cached frame)."""
        return self._last_results
