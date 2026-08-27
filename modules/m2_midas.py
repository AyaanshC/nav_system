"""
modules/m2_midas.py

Runs MiDaS v2.1 (small variant) on a frame.
Outputs: depth_map (H x W float32, relative inverse depth — larger = closer)
         obstacle_report: dict with distance buckets per frame region

RTX 4060: MiDaS small runs at ~120fps on GPU. We call it every 5th frame.
VRAM:     ~500MB FP32

Obstacle distance buckets:
  near  — dominant depth percentile > NEAR_THRESHOLD  (within ~1.5m)
  mid   — dominant depth percentile > MID_THRESHOLD   (1.5–3m)
  far   — everything else (safe to walk)
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from loguru import logger


NEAR_THRESHOLD = 0.72   # Relative inverse depth (tunable via settings.yaml)
MID_THRESHOLD  = 0.45


class DepthEstimator:
    """
    Singleton-style class. Load once, call estimate() per frame.

    Uses MiDaS v2.1 small via torch.hub. Loads local .pt weights.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Args:
            model_path: Path to midas_v21_small_256.pt
            device:     "cuda" or "cpu"
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if str(self.device) == "cpu":
            logger.warning("[MiDaS] CUDA not available, running on CPU (will be slow).")
        else:
            logger.info(f"[MiDaS] Loading on {self.device}")
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """
        Load MiDaS v2.1 small via torch.hub (with fallback to direct timm load).
        Uses weights_only=False for compatibility with legacy .pt weight files.
        """
        try:
            # Primary: torch hub (cached after first download)
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                "MiDaS_small",
                pretrained=False,
                trust_repo=True,
                verbose=False
            )
        except Exception as e:
            logger.warning(f"[MiDaS] torch.hub load failed ({e}), trying direct timm build...")
            # Fallback: build DPT-small manually via timm
            import timm
            self.model = timm.create_model("tf_efficientnet_lite3", pretrained=False)

        state = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # Build the MiDaS transforms via torch hub
        try:
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms",
                trust_repo=True, verbose=False
            )
            self.transform = midas_transforms.small_transform
        except Exception as e:
            logger.warning(f"[MiDaS] Could not load hub transforms ({e}). Using manual transform.")
            self.transform = self._make_manual_transform()

        logger.info("[MiDaS] Model loaded and ready.")

    def _make_manual_transform(self):
        """Fallback transform if MiDaS hub transforms are unavailable."""
        import torchvision.transforms as T
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize((256, 256), antialias=True),
        ])

    @torch.no_grad()
    def estimate(self, frame_bgr: np.ndarray) -> dict:
        """
        Run depth estimation on a single frame.

        Args:
            frame_bgr: OpenCV BGR frame (H x W x 3, uint8)

        Returns:
            {
              "depth_map":     np.ndarray (H, W) float32, normalized 0-1,
              "center_bucket": "near" | "mid" | "far",
              "left_bucket":   "near" | "mid" | "far",
              "right_bucket":  "near" | "mid" | "far",
              "prompt_text":   str  e.g. "Left: clear. Center: close obstacle. Right: clear."
            }
        """
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).to(self.device)

        prediction = self.model(input_tensor)
        # Use the raw 256x256 output directly — no need to upsample to full
        # frame resolution since we only compute 3 regional bucket values.
        if prediction.dim() == 3:
            depth_raw = prediction.squeeze(0)
        else:
            depth_raw = prediction.squeeze()

        depth = depth_raw.cpu().numpy().astype(np.float32)

        # Normalize 0–1
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        H, W = depth.shape
        # Divide frame into left / center / right columns (thirds)
        l_edge = W // 3
        r_edge = 2 * (W // 3)

        def bucket(region: np.ndarray) -> str:
            """Map 80th percentile depth value to distance label."""
            val = float(np.percentile(region, 80))
            if val > NEAR_THRESHOLD:
                return "near"
            elif val > MID_THRESHOLD:
                return "mid"
            return "far"

        c_bucket = bucket(depth[:, l_edge:r_edge])
        l_bucket = bucket(depth[:, :l_edge])
        r_bucket = bucket(depth[:, r_edge:])

        def desc(b: str) -> str:
            return {"near": "close obstacle", "mid": "object ahead", "far": "clear"}[b]

        prompt_text = (
            f"Depth sensor — Left: {desc(l_bucket)}. "
            f"Center: {desc(c_bucket)}. "
            f"Right: {desc(r_bucket)}."
        )

        return {
            "depth_map":     depth,
            "center_bucket": c_bucket,
            "left_bucket":   l_bucket,
            "right_bucket":  r_bucket,
            "prompt_text":   prompt_text
        }
