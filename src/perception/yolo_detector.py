import torch
import numpy as np
from ultralytics import YOLO
from loguru import logger
import hashlib
from typing import List, Dict, Tuple

# Pre-set obstacle labels (based on paper constraints)
OBSTACLE_VOCAB = [
    "door", "stairs", "chair", "table", "bin", "person", "elevator", "obstacle"
]

class YOLOWorldDetector:
    """
    Open-Vocabulary Obstacle Detector fusing YOLO-World and Depth Anything V2.
    """
    def __init__(self, model_path: str = "yolov8s-world.pt", conf_threshold: float = 0.35, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.conf = conf_threshold
        logger.info(f"[YOLO] Loading YOLO-World model on {self.device}")

        try:
            self.model = YOLO(model_path)
            self.model.set_classes(OBSTACLE_VOCAB)
        except Exception as e:
            logger.error(f"[YOLO] Failed to load YOLO model: {e}")
            logger.warning("[YOLO] Falling back to dummy YOLO model for testing.")
            self.model = None

        self._last_hash = ""
        self._last_results = []

    def detect(self, frame_bgr: np.ndarray, depth_map: np.ndarray, depth_estimator) -> Tuple[List[Dict], bool, bool]:
        """
        Runs object detection and fuses with depth map.

        Args:
            frame_bgr: BGR frame
            depth_map: Depth map from Depth Estimator
            depth_estimator: Instance of DepthEstimator to calculate bbox distance

        Returns:
            detections: List of detection dictionaries
            changed: Boolean indicating if detection hash changed
            immediate_hazard: Boolean indicating if an obstacle is < 1.5m in the Center
        """
        if self.model is None:
            # Dummy detection for testing if model failed
            return [], False, False

        H, W = frame_bgr.shape[:2]

        results = self.model.predict(
            frame_bgr,
            verbose=False,
            device=self.device,
            half=(self.device == "cuda"),
            conf=self.conf
        )

        detections = []
        immediate_hazard = False

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_idx = int(box.cls[0])
                cls_name = OBSTACLE_VOCAB[cls_idx] if cls_idx < len(OBSTACLE_VOCAB) else "obstacle"
                conf_val = float(box.conf[0])

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

                # Classify horizontal direction
                x_center_norm = ((x1 + x2) / 2.0) / W
                if x_center_norm < 0.35:
                    direction = "Left"
                elif x_center_norm <= 0.65:
                    direction = "Center"
                else:
                    direction = "Right"

                # Extract distance in meters
                bbox = [x1, y1, x2, y2]
                distance = depth_estimator.get_bounding_box_distance(bbox, depth_map)

                # Immediate Hazard Trigger
                if direction == "Center" and distance < 1.5:
                    immediate_hazard = True
                    logger.warning(f"[YOLO] IMMEDIATE HAZARD: {cls_name} directly ahead at {distance:.2f}m!")

                detections.append({
                    "class": cls_name,
                    "confidence": conf_val,
                    "bbox": [x1, y1, x2, y2],
                    "direction": direction,
                    "distance": distance
                })

        # Hash check to see if we need to reprompt VLM
        # Hash based on sorted classes and their directions
        detection_strings = [f"{d['class']}_{d['direction']}" for d in detections]
        new_hash = hashlib.md5(",".join(sorted(detection_strings)).encode()).hexdigest()

        changed = new_hash != self._last_hash
        self._last_hash = new_hash
        self._last_results = detections

        return detections, changed, immediate_hazard
