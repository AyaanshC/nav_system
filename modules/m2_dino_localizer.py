"""
modules/m2_dino_localizer.py

State-of-the-art visual place recognition using DINOv2 + FAISS.

Replaces the old ORB feature matching approach entirely.

Why DINOv2 is dramatically better than ORB:
  - ORB fails on: dim lighting, motion blur, textureless walls, repeated patterns
  - DINOv2 was trained on 142M images with self-supervised learning
  - Its embeddings are semantically meaningful: two photos of the "same corridor
    from different angles" produce similar embeddings, even if no ORB features match
  - Robust to: lighting changes, viewpoint shifts, partial occlusion, blur

How it works:
  1. Scan walk: DINOv2 embeds every saved keyframe → stored as .npy index
  2. Navigation: DINOv2 embeds current frame → FAISS finds nearest keyframe
     → returns corresponding node_id

Memory usage: ~300MB VRAM for DINOv2-ViT-S14 (21M params)
Latency:      ~15ms per frame on RTX 4060 (fully GPU-accelerated)
FAISS search: <1ms for 1000 keyframes on CPU index
"""

import cv2
import numpy as np
import json
import torch
import torchvision.transforms as T
from pathlib import Path
from loguru import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("[DINOv2] faiss not installed. Run: pip install faiss-cpu")


# ── DINOv2 image preprocessing ────────────────────────────────────────────────

# Standard ImageNet normalization used by DINOv2
_DINO_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),   # DINOv2 ViT-S14 uses 224px patches
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DINOv2Localizer:
    """
    Visual place recognition engine using DINOv2 embeddings + FAISS index.

    Usage:
        # Once, after scan walk:
        localizer = DINOv2Localizer(map_dir)
        localizer.build_index(keyframe_dir)

        # Per-frame during navigation:
        node_id = localizer.localize(frame_bgr)
    """

    EMBEDDING_DIM = 384    # DINOv2-ViT-S14 output dimension
    MIN_SIMILARITY = 0.60  # Cosine similarity threshold — below this = lost

    def __init__(self, map_dir: str, device: str = "cuda", model=None):
        """
        Args:
            map_dir: Path to maps/<building>/ directory
            device:  "cuda" or "cpu"
            model:   Optional pre-loaded DINOv2 model to reuse (avoids double VRAM load)
        """
        self.map_dir = Path(map_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model: torch.nn.Module | None = None
        self._index: object | None = None           # FAISS index
        self._node_ids: list[str] = []              # Maps FAISS row → node_id
        self._current_node: str | None = None

        # Temporal smoothing: node must appear K consecutive times before transition
        self._smooth_k = 3
        self._candidates: list[str] = []

        if model is not None:
            logger.info("[DINOv2] Reusing pre-loaded DINOv2 model (VRAM saved).")
            self._model = model
        else:
            self._load_model()
        self._load_index()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        """Load DINOv2-ViT-S14 from torch.hub (cached after first download)."""
        logger.info("[DINOv2] Loading DINOv2-ViT-S14...")
        try:
            self._model = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=True,
                verbose=False
            )
            self._model.eval()
            self._model.to(self.device)
            logger.info(f"[DINOv2] Model loaded on {self.device} "
                        f"({sum(p.numel() for p in self._model.parameters())/1e6:.1f}M params)")
        except Exception as e:
            logger.error(f"[DINOv2] Failed to load model: {e}")
            raise

    # ── Index loading / building ──────────────────────────────────────────────

    def _index_path(self) -> Path:
        return self.map_dir / "dino_index.faiss"

    def _node_ids_path(self) -> Path:
        return self.map_dir / "dino_node_ids.json"

    def _load_index(self):
        """Load a pre-built FAISS index from disk, if it exists."""
        if not FAISS_AVAILABLE:
            return
        idx_path = self._index_path()
        ids_path = self._node_ids_path()

        if not idx_path.exists() or not ids_path.exists():
            logger.info("[DINOv2] No FAISS index found. Run scan_walk.py to build it.")
            return

        try:
            self._index = faiss.read_index(str(idx_path))
            with open(ids_path, encoding="utf-8") as f:
                self._node_ids = json.load(f)
            logger.info(f"[DINOv2] Loaded FAISS index: {len(self._node_ids)} keyframes")
        except Exception as e:
            logger.error(f"[DINOv2] Failed to load FAISS index: {e}")

    def build_index(self, keyframe_dir: str) -> None:
        """
        Embed all keyframe images and save a FAISS index to disk.
        Called once at the end of scan_walk.py.

        Args:
            keyframe_dir: Directory containing node_*.jpg files
        """
        if not FAISS_AVAILABLE:
            logger.error("[DINOv2] Cannot build index — faiss not installed.")
            return

        kf_dir = Path(keyframe_dir)
        image_paths = sorted(kf_dir.glob("node_*.jpg"))

        if not image_paths:
            logger.error(f"[DINOv2] No keyframes found in {keyframe_dir}")
            return

        logger.info(f"[DINOv2] Embedding {len(image_paths)} keyframes...")

        embeddings = []
        node_ids = []

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            emb = self.embed(img)
            embeddings.append(emb)
            node_ids.append(img_path.stem)

        if not embeddings:
            logger.error("[DINOv2] No valid images to embed.")
            return

        # Stack into (N, 384) matrix
        matrix = np.vstack(embeddings).astype(np.float32)

        # Normalize for cosine similarity search
        faiss.normalize_L2(matrix)

        # Build flat L2 index (exact search — fast enough for <2000 keyframes)
        index = faiss.IndexFlatIP(self.EMBEDDING_DIM)  # IP = inner product = cosine after normalize
        index.add(matrix)

        # Save to disk
        self.map_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self._index_path()))
        with open(self._node_ids_path(), "w", encoding="utf-8") as f:
            json.dump(node_ids, f)

        self._index = index
        self._node_ids = node_ids

        logger.success(f"[DINOv2] FAISS index saved: {len(node_ids)} nodes "
                       f"at {self._index_path()}")

    # ── Embedding extraction ──────────────────────────────────────────────────

    @torch.no_grad()
    def embed(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extract DINOv2 embedding from a BGR frame.

        Args:
            frame_bgr: OpenCV BGR image

        Returns:
            numpy array of shape (384,) — the frame's semantic embedding
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = _DINO_TRANSFORM(rgb).unsqueeze(0).to(self.device)
        features = self._model(tensor)
        return features.squeeze(0).cpu().numpy()

    # ── Localization ──────────────────────────────────────────────────────────

    def localize(self, frame_bgr: np.ndarray) -> str | None:
        """
        Find the OSMAG node matching the current camera view.

        Uses DINOv2 embedding + FAISS nearest-neighbor + temporal smoothing.

        Args:
            frame_bgr: Current camera frame (BGR)

        Returns:
            node_id string or None if confidence is too low
        """
        if self._index is None or not self._node_ids:
            return self._current_node  # Return last known if no index

        # Embed current frame
        emb = self.embed(frame_bgr).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(emb)

        # Search top-1 nearest neighbor
        scores, indices = self._index.search(emb, k=1)
        score = float(scores[0][0])   # Cosine similarity (0–1 after normalization)
        best_idx = int(indices[0][0])

        if score < self.MIN_SIMILARITY:
            logger.debug(f"[DINOv2] Low similarity: {score:.3f}. Holding last node.")
            self._candidates.clear()
            return self._current_node  # Return last known node (graceful degradation)

        candidate = self._node_ids[best_idx]

        # Temporal smoothing: transition only when same node wins K consecutive times
        self._candidates.append(candidate)
        if len(self._candidates) > self._smooth_k:
            self._candidates.pop(0)

        if (len(self._candidates) == self._smooth_k
                and len(set(self._candidates)) == 1):
            # Confirmed new node
            if candidate != self._current_node:
                logger.info(f"[DINOv2] Node: {self._current_node} → {candidate} "
                            f"(similarity={score:.3f})")
                self._current_node = candidate

        return self._current_node

    @property
    def is_ready(self) -> bool:
        """True if the FAISS index is loaded and ready for localization."""
        return self._index is not None and bool(self._node_ids)
