"""
modules/m0_image_enhancer.py

Applies fast image processing techniques (CLAHE, Gamma Correction, Unsharp Mask)
to improve frame clarity, lighting, and contrast before sending to VLM.
"""

import cv2
import numpy as np
from loguru import logger

class ImageEnhancer:
    def __init__(self, config: dict = None):
        """
        Initializes the Image Enhancer with configuration settings.
        
        Args:
            config: Dict containing settings like enable_enhancement, clahe_clip_limit, etc.
        """
        self.config = config or {}
        self.enabled = self.config.get("enable_image_enhancement", True)
        self.clahe_clip = self.config.get("clahe_clip_limit", 2.0)
        self.clahe_grid = self.config.get("clahe_grid_size", 8)
        
        # We now use dynamic gamma, but keep a base fallback in config
        self.base_gamma = self.config.get("gamma_correction", 1.2)
        self.unsharp_amount = self.config.get("unsharp_amount", 1.5)
        
        # New settings for advanced pipeline
        self.enable_awb = self.config.get("enable_auto_white_balance", True)
        self.enable_denoise = self.config.get("enable_denoising", True)
        
        # Pre-compute gamma tables for common ranges to avoid runtime math
        self._gamma_tables = {}
        for g in [1.0, 1.2, 1.5]:
            invG = 1.0 / g
            self._gamma_tables[g] = np.array([((i / 255.0) ** invG) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        # Initialize CLAHE object
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip, 
            tileGridSize=(self.clahe_grid, self.clahe_grid)
        )
        
        if self.enabled:
            logger.info(f"[ImageEnhancer] Enabled: CLAHE, Dynamic Gamma, Denoise={self.enable_denoise}, AWB={self.enable_awb}")
        else:
            logger.info("[ImageEnhancer] Disabled.")

    def _auto_white_balance(self, img_bgr: np.ndarray) -> np.ndarray:
        """Simple Gray World assumption for auto white balancing."""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        avg_a = np.average(lab[:, :, 1])
        avg_b = np.average(lab[:, :, 2])
        # Neutralize color casts by moving averages towards 128 (neutral gray in LAB)
        lab[:, :, 1] = lab[:, :, 1] - ((avg_a - 128) * (lab[:, :, 1] / 255.0) * 1.1)
        lab[:, :, 2] = lab[:, :, 2] - ((avg_b - 128) * (lab[:, :, 2] / 255.0) * 1.1)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Applies the configured advanced enhancement pipeline to a BGR image.
        """
        if not self.enabled or img_bgr is None:
            return img_bgr
            
        current = img_bgr.copy()

        # 1. Auto White Balance
        if self.enable_awb:
            current = self._auto_white_balance(current)
            
        # 2. CLAHE (Local Contrast) and Luminance calculation
        lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        mean_luminance = np.mean(l)
        
        cl = self._clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        current = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 3. Dynamic Gamma Correction
        # If very dark, boost heavily. If very bright, skip.
        if mean_luminance < 70:
            target_gamma = 1.5
        elif mean_luminance > 140:
            target_gamma = 1.0
        else:
            target_gamma = 1.2
            
        if target_gamma != 1.0:
            current = cv2.LUT(current, self._gamma_tables[target_gamma])
            
        # 4. Fast Denoising (Bilateral Filter preserves edges while blurring noise)
        if self.enable_denoise:
            # d=5, sigmaColor=25, sigmaSpace=25 (light, fast denoising)
            current = cv2.bilateralFilter(current, 5, 25, 25)
        
        # 5. Unsharp Mask (Sharpening to fix motion blur)
        blurred = cv2.GaussianBlur(current, (5, 5), 1.0)
        sharpened = float(self.unsharp_amount + 1) * current - float(self.unsharp_amount) * blurred
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
