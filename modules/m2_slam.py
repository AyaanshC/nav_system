"""
modules/m2_slam.py

Orchestrates localization. Two modes:
  full     — ORB-SLAM3 (Linux/WSL2 only, requires build)
  fallback — DINOv2 + FAISS (default on Windows, state-of-the-art VPR)

The DINOv2 fallback completely replaces the old ORB BFMatcher approach.
DINOv2 is robust to lighting changes, motion blur, and textureless surfaces
where ORB would fail completely.
"""

import numpy as np
import json
import os
from pathlib import Path
from loguru import logger


class SLAMLocalizer:
    """
    High-level localization wrapper.

    In fallback mode (default), delegates entirely to DINOv2Localizer.
    In full mode, uses ORB-SLAM3 (requires Linux build).

    Public API is identical regardless of mode:
        node_id = slam.process_frame(frame_bgr, timestamp)
    """

    # Localize every N frames (position doesn't change frame-to-frame)
    SKIP_N = 4

    def __init__(self, map_dir: str, use_fallback: bool = True, vocab_path: str = ""):
        """
        Args:
            map_dir:      Path to maps/<building>/ directory
            use_fallback: True = DINOv2+FAISS, False = full ORB-SLAM3
            vocab_path:   Path to ORBvoc.txt (only needed for full mode)
        """
        self.map_dir = Path(map_dir)
        self.use_fallback = use_fallback
        self._slam = None
        self._localizer = None
        self._frame_counter = 0
        self._current_node: str | None = None

        if not use_fallback:
            self._init_orbslam3(vocab_path)
        else:
            self._init_dinov2()

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_dinov2(self):
        """Initialize DINOv2 + FAISS localizer (default fallback)."""
        try:
            from modules.m2_dino_localizer import DINOv2Localizer
            self._localizer = DINOv2Localizer(
                map_dir=str(self.map_dir),
                device="cuda"
            )
            if self._localizer.is_ready:
                logger.info("[SLAM] DINOv2+FAISS localizer ready.")
            else:
                logger.warning(
                    "[SLAM] DINOv2 index not found. "
                    "Run scan_walk.py to build it first."
                )
        except Exception as e:
            logger.error(f"[SLAM] Failed to init DINOv2 localizer: {e}")
            raise

    def _init_orbslam3(self, vocab_path: str):
        try:
            import orbslam3
            slam_map = str(self.map_dir / "slam_map.bin")
            config = str(self.map_dir / "slam_config.yaml")
            self._slam = orbslam3.System(vocab_path, config, orbslam3.Sensor.MONOCULAR)
            self._slam.set_use_viewer(False)
            if os.path.exists(slam_map):
                self._slam.load_atlas_from_file(slam_map)
                logger.info("[SLAM] ORB-SLAM3 loaded existing map.")
            else:
                logger.info("[SLAM] ORB-SLAM3 starting fresh map.")
        except ImportError:
            logger.warning("[SLAM] orbslam3 not installed. Switching to DINOv2 fallback.")
            self.use_fallback = True
            self._init_dinov2()

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray, timestamp: float = 0.0) -> str | None:
        """
        Localize within the current frame.
        Frame-skips to reduce compute — only runs DINOv2 every SKIP_N frames.

        Args:
            frame_bgr: OpenCV BGR frame
            timestamp: seconds (float) — use time.time()

        Returns:
            node_id string or None if localization is lost
        """
        self._frame_counter += 1

        # Fast path: return cached node on skipped frames
        if self._frame_counter % self.SKIP_N != 0:
            return self._current_node

        if self.use_fallback:
            node = self._localizer.localize(frame_bgr) if self._localizer else None
        else:
            node = self._orbslam3_localize(frame_bgr, timestamp)

        if node is not None:
            self._current_node = node

        return self._current_node

    def save_map(self):
        """Save ORB-SLAM3 map to disk (no-op in DINOv2 fallback mode)."""
        if not self.use_fallback and self._slam:
            self._slam.save_atlas_to_file(str(self.map_dir / "slam_map.bin"))
            logger.info("[SLAM] Map saved.")

    def reload_keyframes(self):
        """Reload DINOv2 FAISS index from disk (call after scan_walk completes)."""
        if self.use_fallback and self._localizer:
            self._localizer._load_index()

    # ── ORB-SLAM3 private methods ─────────────────────────────────────────────

    def _orbslam3_localize(self, frame_bgr: np.ndarray, timestamp: float) -> str | None:
        if self._slam is None:
            return None
        import orbslam3
        pose = self._slam.process_image_mono(frame_bgr, timestamp)
        if pose is None:
            return None
        kf_id = self._slam.get_tracking_state()
        return self._kf_id_to_node(str(kf_id))

    def _kf_id_to_node(self, kf_id: str) -> str | None:
        mapping_path = self.map_dir / "kf_to_node.json"
        if not mapping_path.exists():
            return kf_id
        with open(mapping_path) as f:
            mapping = json.load(f)
        return mapping.get(kf_id)
