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
        self.gamma = self.config.get("gamma_correction", 1.2)
        self.unsharp_amount = self.config.get("unsharp_amount", 1.5)
        
        # Pre-compute gamma lookup table for performance
        invGamma = 1.0 / self.gamma
        self._gamma_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        # Initialize CLAHE object
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip, 
            tileGridSize=(self.clahe_grid, self.clahe_grid)
        )
        
        if self.enabled:
            logger.info(f"[ImageEnhancer] Enabled: CLAHE(clip={self.clahe_clip}), Gamma={self.gamma}, Unsharp={self.unsharp_amount}")
        else:
            logger.info("[ImageEnhancer] Disabled.")

    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Applies the configured enhancement pipeline to a BGR image.
        
        Args:
            img_bgr: Raw input frame (BGR format)
            
        Returns:
            Enhanced frame (BGR format)
        """
        if not self.enabled or img_bgr is None:
            return img_bgr
            
        # 1. Apply CLAHE (on L channel of LAB color space) to fix lighting
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self._clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 2. Apply Gamma Correction to lift shadows generally
        img_gamma = cv2.LUT(img_clahe, self._gamma_table)
        
        # 3. Apply Unsharp Mask to reduce motion blur and sharpen edges
        blurred = cv2.GaussianBlur(img_gamma, (5, 5), 1.0)
        sharpened = float(self.unsharp_amount + 1) * img_gamma - float(self.unsharp_amount) * blurred
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
