"""
modules/m2_metric_depth.py

State-of-the-art Metric Depth Perception Module using Video-Depth-Anything Small.

Replaces the old MiDaS relative inverse depth estimation with true physical meters:
  - Video-Depth-Anything Small (28.4M params)
  - Uses streaming temporal attention to eliminate frame-to-frame depth jitter/flickering
  - Outputs physical distance in meters (e.g. 0.2m to 20.0m)
  - Seamless drop-in replacement for m2_midas.py:
      result = depth_estimator.estimate(frame_bgr)
      result["depth_map"]          -> 2D float32 array in real meters
      result["min_distance_m"]     -> closest obstacle in walking path
      result["center_distance_m"]  -> median forward corridor distance
      result["danger_level"]       -> 0.0 to 1.0 (used directly by m8_sonification.py)
      result["prompt_text"]        -> descriptive spatial text for Qwen2.5-VL
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
from loguru import logger

# Ensure Video-Depth-Anything submodule is in Python sys.path
VDA_DIR = Path(__file__).parent.parent / "Video-Depth-Anything"
if VDA_DIR.exists() and str(VDA_DIR) not in sys.path:
    sys.path.insert(0, str(VDA_DIR))

try:
    import torch
    from video_depth_anything.video_depth_stream import VideoDepthAnything
    VDA_AVAILABLE = True
except ImportError as e:
    VDA_AVAILABLE = False
    logger.warning(f"[MetricDepth] Video-Depth-Anything could not be imported: {e}")


class MetricDepthEstimator:
    """
    Real-time Metric Depth Perception Engine.
    Uses Video Depth Anything Small in streaming mode for continuous, temporally consistent depth.
    """

    def __init__(
        self,
        model_path: str = "models/metric_video_depth_anything_vits.pth",
        device: str = "cuda",
        input_size: int = 518,
        collision_dist_m: float = 0.6,
        warning_dist_m: float = 1.8,
    ):
        """
        Args:
            model_path:       Path to metric_video_depth_anything_vits.pth checkpoint
            device:           'cuda' or 'cpu' (falls back to cpu if cuda unavailable)
            input_size:       Inference resolution dimension (multiple of 14, default 518)
            collision_dist_m: Imminent collision distance threshold in meters (<0.6m -> danger=1.0)
            warning_dist_m:   Caution warning threshold in meters (>1.8m -> danger=0.0)
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.collision_dist_m = collision_dist_m
        self.warning_dist_m = warning_dist_m
        self.model = None

        if not VDA_AVAILABLE:
            logger.error("[MetricDepth] Missing dependencies. Ensure PyTorch and Video-Depth-Anything are available.")
            return

        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        if self.device == "cpu":
            logger.warning("[MetricDepth] Running on CPU. GPU recommended for lower latency.")
        else:
            logger.info(f"[MetricDepth] Initializing on {self.device}")

        self._load_model()

    def _load_model(self):
        """Initializes Video-Depth-Anything-Small and loads the metric checkpoint."""
        if not self.model_path.exists():
            logger.error(f"[MetricDepth] Weights not found at {self.model_path}. Please check the models/ directory.")
            return

        try:
            logger.info(f"[MetricDepth] Loading Video-Depth-Anything Small weights: {self.model_path.name}")
            # Small configuration: features=64, out_channels=[48, 96, 192, 384]
            model_config = {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            }
            self.model = VideoDepthAnything(**model_config)

            state_dict = torch.load(str(self.model_path), map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
            self.model = self.model.to(self.device).eval()
            logger.success("[MetricDepth] Model loaded and ready for streaming inference.")
        except Exception as e:
            logger.error(f"[MetricDepth] Failed to initialize model: {e}")
            self.model = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def estimate(self, frame_bgr: np.ndarray) -> dict:
        """
        Run streaming metric depth estimation on a single camera frame.

        Args:
            frame_bgr: OpenCV BGR frame (H x W x 3, uint8)

        Returns:
            dict containing:
              - "depth_map":          2D np.ndarray float32 (values in physical METERS)
              - "is_metric":          True
              - "min_distance_m":     closest obstacle in forward corridor (meters)
              - "center_distance_m":  median distance in center path (meters)
              - "left_distance_m":    median distance in left third (meters)
              - "right_distance_m":   median distance in right third (meters)
              - "danger_level":       0.0 (safe) to 1.0 (imminent collision) for m8_sonification
              - "center_bucket":      "near" | "mid" | "far" (legacy compatibility)
              - "left_bucket":        "near" | "mid" | "far"
              - "right_bucket":       "near" | "mid" | "far"
              - "prompt_text":        Formatted text for Qwen2.5-VL guidance
        """
        if self.model is None or frame_bgr is None:
            return self._fallback_result()

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        try:
            # infer_video_depth_one maintains streaming temporal attention cache
            depth = self.model.infer_video_depth_one(
                frame_rgb,
                input_size=self.input_size,
                device=self.device,
                fp32=(self.device == "cpu")
            )
            # Ensure output is a 2D float32 array in meters
            depth = np.clip(depth.astype(np.float32), 0.05, 30.0)
        except Exception as e:
            logger.error(f"[MetricDepth] Inference failed: {e}")
            return self._fallback_result()

        # Split frame into corridor vertical columns (thirds)
        l_edge = w // 3
        r_edge = 2 * (w // 3)

        left_zone = depth[:, :l_edge]
        center_zone = depth[:, l_edge:r_edge]
        right_zone = depth[:, r_edge:]

        # Sample robust percentile distance (20th percentile captures closest obstacle without single-pixel noise)
        left_dist = float(np.percentile(left_zone, 20))
        center_dist = float(np.percentile(center_zone, 20))
        right_dist = float(np.percentile(right_zone, 20))

        # Walking path hazard distance (focus on center and bottom-half foreground)
        lower_center = center_zone[int(h * 0.3):, :]
        min_dist = float(np.percentile(lower_center, 10)) if lower_center.size > 0 else center_dist

        # Compute physical danger level (0.0 to 1.0) for acoustic sonification
        if min_dist >= self.warning_dist_m:
            danger = 0.0
        elif min_dist <= self.collision_dist_m:
            danger = 1.0
        else:
            danger = (self.warning_dist_m - min_dist) / (self.warning_dist_m - self.collision_dist_m)
        danger = float(np.clip(danger, 0.0, 1.0))

        # Categorize into descriptive buckets
        def categorize(dist: float) -> str:
            if dist < 1.0:
                return "near"
            elif dist < 2.5:
                return "mid"
            return "far"

        l_bucket = categorize(left_dist)
        c_bucket = categorize(center_dist)
        r_bucket = categorize(right_dist)

        def desc(b: str, dist: float) -> str:
            if b == "near":
                return f"close obstacle ({dist:.1f}m)"
            elif b == "mid":
                return f"ahead ({dist:.1f}m)"
            return f"clear ({dist:.1f}m)"

        prompt_text = (
            f"Metric Depth sensor — "
            f"Left: {desc(l_bucket, left_dist)}. "
            f"Center: {desc(c_bucket, center_dist)}. "
            f"Right: {desc(r_bucket, right_dist)}."
        )

        return {
            "depth_map": depth,
            "is_metric": True,
            "min_distance_m": round(min_dist, 2),
            "center_distance_m": round(center_dist, 2),
            "left_distance_m": round(left_dist, 2),
            "right_distance_m": round(right_dist, 2),
            "center_bucket": c_bucket,
            "left_bucket": l_bucket,
            "right_bucket": r_bucket,
            "danger_level": round(danger, 2),
            "prompt_text": prompt_text,
        }

    def _desc(self, bucket: str) -> str:
        return {"near": "close obstacle", "mid": "object ahead", "far": "clear"}.get(bucket, "clear")

    def _fallback_result(self) -> dict:
        """Safe graceful fallback if model is unavailable."""
        return {
            "depth_map": None,
            "is_metric": False,
            "min_distance_m": 5.0,
            "center_distance_m": 5.0,
            "left_distance_m": 5.0,
            "right_distance_m": 5.0,
            "center_bucket": "far",
            "left_bucket": "far",
            "right_bucket": "far",
            "danger_level": 0.0,
            "prompt_text": "Depth sensor unavailable.",
        }

    @staticmethod
    def colorize_depth(depth_map: np.ndarray, min_d: float = 0.5, max_d: float = 6.0) -> np.ndarray:
        """
        Colorize metric depth map (meters) to BGR for dashboard display.
        Close obstacles are red/warm, distant areas are blue/cool.
        """
        if depth_map is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)
        clipped = np.clip(depth_map, min_d, max_d)
        norm = ((clipped - min_d) / (max_d - min_d) * 255.0).astype(np.uint8)
        # Invert so closer objects are hot (red) and farther are cold (blue)
        return cv2.applyColorMap(255 - norm, cv2.COLORMAP_INFERNO)
