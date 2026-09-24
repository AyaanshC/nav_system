import torch
import numpy as np
import cv2
from loguru import logger
import urllib.request
import os

class DepthEstimator:
    """
    Monocular depth estimation using Depth Anything V2 (ViT-Small).
    Replaces the previous MiDaS module.
    """
    def __init__(self, model_path: str = "depth_anything_v2_vits.pth", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"[Depth] Initializing Depth Anything V2 on {self.device}")

        self.model_path = model_path

        # Load Depth Anything V2 from torch hub or fallback
        try:
            try:
                import timm
                self.model = timm.create_model('hf_hub:spacevtx/depth-anything-v2-vits', pretrained=False)
            except Exception as e:
                logger.warning(f"[Depth] timm load failed: {e}. Trying torch hub...")
                self.model = torch.hub.load('huggingface/pytorch-image-models', 'hf_hub:spacevtx/depth-anything-v2-vits', pretrained=False, trust_repo=True)

            if os.path.exists(model_path):
                logger.info(f"[Depth] Loading weights from {model_path}")
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                 logger.warning(f"[Depth] Local weights {model_path} not found. Attempting to load pretrained weights.")
                 try:
                     import timm
                     self.model = timm.create_model('hf_hub:spacevtx/depth-anything-v2-vits', pretrained=True)
                 except Exception:
                     self.model = torch.hub.load('huggingface/pytorch-image-models', 'hf_hub:spacevtx/depth-anything-v2-vits', pretrained=True, trust_repo=True)

            self.model = self.model.to(self.device).eval()
        except Exception as e:
            logger.error(f"[Depth] Failed to load depth model: {e}")
            logger.warning("[Depth] Falling back to a dummy model for testing purposes")
            self.model = None

        # We'll use a simple transform since Depth Anything V2 usually expects specific input preprocessing,
        # but in this implementation we'll manually preprocess or use a basic one if the model is dummy.

    @torch.no_grad()
    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Estimates depth for a given BGR frame.

        Args:
            frame_bgr: BGR numpy array

        Returns:
            depth_map: Normalized depth map (H, W) as float32 in [0, 1]
        """
        if self.model is None:
            # Dummy depth map for testing if model failed to load
            return np.ones((frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.float32) * 0.5

        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        # Typical Depth Anything V2 preprocessing (simplified)
        img_rgb = img_rgb / 255.0
        img_rgb = (img_rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        # Resize to standard input size, e.g., 518x518
        img_resized = cv2.resize(img_rgb, (518, 518))

        input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # Inference
        try:
            depth_raw = self.model(input_tensor)

            if depth_raw.dim() == 4:
                depth_raw = depth_raw.squeeze()
            elif depth_raw.dim() == 3:
                depth_raw = depth_raw.squeeze(0)

            depth_pred = depth_raw.cpu().numpy()

            # Resize back to original
            depth_map = cv2.resize(depth_pred, (W, H))

            # Normalize 0-1
            d_min, d_max = float(depth_map.min()), float(depth_map.max())
            if d_max - d_min > 1e-6:
                depth_map = (depth_map - d_min) / (d_max - d_min)
            else:
                depth_map = np.zeros_like(depth_map)

            return depth_map

        except Exception as e:
            logger.error(f"[Depth] Inference failed: {e}")
            return np.ones((H, W), dtype=np.float32) * 0.5

    def get_bounding_box_distance(self, bbox: list[float], depth_map: np.ndarray) -> float:
        """
        Calculates the median depth of the central 50% ROI of a bounding box.

        Args:
            bbox: [x1, y1, x2, y2]
            depth_map: Normalized depth map (H, W) in [0, 1]. (Higher = Closer usually in these models)

        Returns:
            distance: float representing distance. For depth anything, higher value is closer,
                      so we'll invert it to represent physical distance (in arbitrary relative units,
                      or approximate meters if calibrated). Let's return the median depth value directly
                      for downstream thresholding.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]

        H, W = depth_map.shape
        x1, x2 = max(0, x1), min(W - 1, x2)
        y1, y2 = max(0, y1), min(H - 1, y2)

        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            return 0.0

        # Central 50% ROI to resist edge noise
        roi_x1 = int(x1 + w * 0.25)
        roi_x2 = int(x2 - w * 0.25)
        roi_y1 = int(y1 + h * 0.25)
        roi_y2 = int(y2 - h * 0.25)

        roi_x1, roi_x2 = max(0, roi_x1), min(W - 1, roi_x2)
        roi_y1, roi_y2 = max(0, roi_y1), min(H - 1, roi_y2)

        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return 0.0

        roi_depth = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]

        # Use median to resist noise
        median_depth = np.median(roi_depth)

        # Invert to approximate distance in meters (very rough calibration for the sake of the system logic)
        # Depth maps are typically inverse depth (disparity). Let's assume normalized 0-1.
        # Higher value = closer.
        # If we need meters, we might do 1.0 / (median_depth + 1e-6), but let's just use
        # a calibrated linear scaling for simplicity: 0.8 is very close (e.g., 0.5m), 0.2 is far (e.g., 5m)
        # We need something where < 1.5m can be triggered.
        # Let's say distance = (1.0 - median_depth) * 5.0
        distance = (1.0 - float(median_depth)) * 5.0

        return distance
