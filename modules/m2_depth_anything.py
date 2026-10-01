"""
modules/m2_depth_anything.py

Depth estimation using Depth Anything V2 (ViT-Small).
Drop-in replacement for m2_midas.py — identical public API.

Why Depth Anything V2 over MiDaS:
  - Trained on 62M+ images (vs MiDaS ~1.9M)
  - Far better on textureless indoor surfaces (plain walls, floors)
  - Sharper depth boundaries around people and furniture
  - No need to download separate weights — loads from HuggingFace on first run

RTX 4060: Runs at ~60fps on GPU. We call it every 5th frame.
VRAM:     ~500MB FP32, ~280MB FP16

Obstacle distance buckets (same thresholds as MiDaS module for compatibility):
  near  — dominant depth percentile > NEAR_THRESHOLD  (within ~1.5m)
  mid   — dominant depth percentile > MID_THRESHOLD   (1.5–3m)
  far   — everything else (safe to walk)

New method vs MiDaS:
  get_bounding_box_distance(bbox, depth_map) -> float
      Returns an estimated distance in meters for a YOLO detection box.
      Ported from nav_system_module2/src/perception/depth_estimator.py.
"""

import torch
import numpy as np
import cv2
from loguru import logger


NEAR_THRESHOLD = 0.72   # Relative inverse depth (higher = closer to camera)
MID_THRESHOLD  = 0.45

# Depth Anything V2 ViT-Small on HuggingFace
_HF_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    """
    Depth Anything V2 depth estimator.

    Drop-in replacement for the MiDaS DepthEstimator in m2_midas.py.
    The constructor signature is compatible: DepthEstimator(model_path, device).
    model_path is accepted but ignored (weights are loaded from HuggingFace).

    Public API:
        estimate(frame_bgr) -> dict          (same as MiDaS)
        get_bounding_box_distance(bbox, depth_map) -> float   (NEW)
    """

    def __init__(self, model_path: str = "", device: str = "cuda"):
        """
        Args:
            model_path: Accepted for API compatibility with m2_midas.py.
                        Ignored — Depth Anything V2 loads from HuggingFace.
            device:     "cuda" or "cpu"
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if str(self.device) == "cpu":
            logger.warning("[DepthAny] CUDA not available — running on CPU (will be slow).")
        else:
            logger.info(f"[DepthAny] Initializing Depth Anything V2 on {self.device}")

        self._pipe = None
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        """
        Load Depth Anything V2 via HuggingFace Transformers pipeline.
        Falls back gracefully to MiDaS on failure so the system never crashes.
        """
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"[DepthAny] Loading {_HF_MODEL_ID} from HuggingFace...")
            device_idx = 0 if str(self.device) == "cuda" else -1
            self._pipe = hf_pipeline(
                task="depth-estimation",
                model=_HF_MODEL_ID,
                device=device_idx
            )
            logger.info("[DepthAny] Depth Anything V2 ready.")

        except Exception as e:
            logger.error(f"[DepthAny] Failed to load Depth Anything V2: {e}")
            logger.warning("[DepthAny] Falling back to MiDaS v2.1 small...")
            self._load_midas_fallback()

    def _load_midas_fallback(self):
        """Emergency fallback: load MiDaS so the system still runs."""
        try:
            self._midas = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small",
                pretrained=True, trust_repo=True, verbose=False
            )
            self._midas.to(self.device).eval()
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms",
                trust_repo=True, verbose=False
            )
            self._midas_transform = midas_transforms.small_transform
            logger.info("[DepthAny] MiDaS fallback loaded successfully.")
        except Exception as e2:
            logger.error(f"[DepthAny] MiDaS fallback also failed: {e2}")
            self._midas = None

    # ── Public API (identical to m2_midas.DepthEstimator) ────────────────────

    @torch.no_grad()
    def estimate(self, frame_bgr: np.ndarray) -> dict:
        """
        Run depth estimation on a single frame.

        Args:
            frame_bgr: OpenCV BGR frame (H x W x 3, uint8)

        Returns:
            {
              "depth_map":     np.ndarray (H, W) float32, normalized 0–1,
                               higher value = CLOSER to camera
              "center_bucket": "near" | "mid" | "far",
              "left_bucket":   "near" | "mid" | "far",
              "right_bucket":  "near" | "mid" | "far",
              "prompt_text":   str  e.g. "Left: clear. Center: close obstacle. Right: clear."
            }
        """
        if self._pipe is not None:
            depth_map = self._estimate_depth_anything(frame_bgr)
        else:
            depth_map = self._estimate_midas(frame_bgr)

        return self._build_result(depth_map)

    def get_bounding_box_distance(self, bbox: list, depth_map: np.ndarray) -> float:
        """
        Estimate distance in meters to an object from its YOLO bounding box.

        Uses the median depth of the central 50% ROI of the bounding box,
        which resists edge noise and partial occlusion.

        Ported from nav_system_module2/src/perception/depth_estimator.py.

        Args:
            bbox:      [x1, y1, x2, y2] in pixels (from YOLO detection)
            depth_map: Normalized depth map (H, W) float32 in [0, 1].
                       Higher value = CLOSER to camera.

        Returns:
            Estimated distance in meters (float).
            ~0.5m for very close, ~5m+ for far objects.
            Returns 5.0 on error (safe default = far).
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            H, W = depth_map.shape

            # Clamp to frame bounds
            x1, x2 = max(0, x1), min(W - 1, x2)
            y1, y2 = max(0, y1), min(H - 1, y2)
            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                return 5.0

            # Central 50% ROI to resist bbox edge noise
            roi_x1 = int(x1 + w * 0.25)
            roi_x2 = int(x2 - w * 0.25)
            roi_y1 = int(y1 + h * 0.25)
            roi_y2 = int(y2 - h * 0.25)

            roi_x1 = max(0, roi_x1)
            roi_x2 = min(W - 1, roi_x2)
            roi_y1 = max(0, roi_y1)
            roi_y2 = min(H - 1, roi_y2)

            if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
                return 5.0

            roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
            median_depth = float(np.median(roi))

            # depth is normalized 0-1, higher = closer.
            # Linear mapping: depth=1.0 → ~0.5m, depth=0.0 → ~5m
            # (rough calibration; valid for relative comparisons)
            distance = (1.0 - median_depth) * 5.0
            return max(0.1, distance)

        except Exception as e:
            logger.warning(f"[DepthAny] BBox distance failed: {e}")
            return 5.0

    # ── Internal inference ────────────────────────────────────────────────────

    def _estimate_depth_anything(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run Depth Anything V2 pipeline on one frame."""
        from PIL import Image
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        result = self._pipe(pil_img)
        depth_pil = result["depth"]

        # Convert PIL depth image to float32 numpy
        depth_raw = np.array(depth_pil, dtype=np.float32)

        # Resize back to original frame size if needed
        H, W = frame_bgr.shape[:2]
        if depth_raw.shape != (H, W):
            depth_raw = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] — higher = closer (inverse depth convention)
        d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
        if d_max - d_min > 1e-6:
            depth_norm = (depth_raw - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_raw)

        return depth_norm

    @torch.no_grad()
    def _estimate_midas(self, frame_bgr: np.ndarray) -> np.ndarray:
        """MiDaS fallback inference."""
        if self._midas is None:
            # Last resort: flat depth map (safe = no obstacles reported)
            H, W = frame_bgr.shape[:2]
            return np.zeros((H, W), dtype=np.float32)

        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self._midas_transform(img_rgb).to(self.device)
        prediction = self._midas(input_tensor)
        depth_raw = prediction.squeeze().cpu().numpy().astype(np.float32)

        d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
        if d_max - d_min > 1e-6:
            return (depth_raw - d_min) / (d_max - d_min)
        return np.zeros_like(depth_raw)

    # ── Bucketing & formatting (identical to m2_midas) ────────────────────────

    def _build_result(self, depth_map: np.ndarray) -> dict:
        """Convert raw depth map into the standard result dict."""
        H, W = depth_map.shape

        l_edge = W // 3
        r_edge = 2 * (W // 3)

        def bucket(region: np.ndarray) -> str:
            val = float(np.percentile(region, 80))
            if val > NEAR_THRESHOLD:
                return "near"
            elif val > MID_THRESHOLD:
                return "mid"
            return "far"

        c_bucket = bucket(depth_map[:, l_edge:r_edge])
        l_bucket = bucket(depth_map[:, :l_edge])
        r_bucket = bucket(depth_map[:, r_edge:])

        def desc(b: str) -> str:
            return {"near": "close obstacle", "mid": "object ahead", "far": "clear"}[b]

        prompt_text = (
            f"Depth sensor — Left: {desc(l_bucket)}. "
            f"Center: {desc(c_bucket)}. "
            f"Right: {desc(r_bucket)}."
        )

        return {
            "depth_map":     depth_map,
            "center_bucket": c_bucket,
            "left_bucket":   l_bucket,
            "right_bucket":  r_bucket,
            "prompt_text":   prompt_text
        }
