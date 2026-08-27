"""
modules/m3_map_builder.py

Improved one-time scan walk → auto-generated labeled OSMAG graph.

Key improvements over original:
  1. Scene-change detection: keyframes only saved when DINOv2 embedding
     changes significantly (not on a fixed timer). This produces denser
     keyframes in complex junctions and sparser ones in plain corridors.

  2. Multi-frame clip labeling: each node is labeled using 3 consecutive
     frames (before, current, after), giving the VLM spatial context from
     multiple viewpoints of the same location.

  3. Edge transition descriptions: for each sequential edge, the VLM
     generates a natural language description of the action needed to move
     between the two nodes ("turn left past the glass door"). Stored as
     edge["description"] in osmag.json for richer path instructions.

  4. DINOv2 index built automatically at end of scan walk.

Model config:
  - Keyframe labeling: qwen2.5vl:7b (offline accuracy)
  - Real-time navigation: qwen2.5vl:3b (speed + VRAM safety)
"""

import cv2
import json
import time
import base64
import os
import re
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
import yaml
from openai import OpenAI
from loguru import logger
from modules.m0_image_enhancer import ImageEnhancer


client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)


# ── DINOv2 for scene-change detection ─────────────────────────────────────────
# Reuse the shared DINOv2 model from m2_dino_localizer to avoid loading
# two copies into VRAM simultaneously (~300MB saved on 8GB GPUs).

_DINO_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DINO_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_SHARED_DINO: object | None = None


def _get_dino_model():
    """Lazy-load DINOv2 once. Shared with m2_dino_localizer to avoid double VRAM usage."""
    global _SHARED_DINO
    if _SHARED_DINO is None:
        logger.info("[MapBuilder] Loading DINOv2 for scene-change detection...")
        _SHARED_DINO = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            pretrained=True, verbose=False
        )
        _SHARED_DINO.eval()
        _SHARED_DINO.to(_DINO_DEVICE)
        logger.info("[MapBuilder] DINOv2 ready.")
    return _SHARED_DINO


@torch.no_grad()
def _embed_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Extract DINOv2 embedding from a BGR frame (384-dim vector)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = _DINO_TRANSFORM(rgb).unsqueeze(0).to(_DINO_DEVICE)
    return _get_dino_model()(tensor).squeeze(0).cpu().numpy()


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2]. 0 = identical, 2 = opposite."""
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - np.dot(a_n, b_n))


# ── Keyframe extraction ────────────────────────────────────────────────────────

def extract_keyframes(
    stream_url: str,
    output_dir: str,
    interval_sec: float = 0.8,
    max_frames: int = 200,
    scene_change_threshold: float = 0.12,
    min_interval_sec: float = 0.5,
) -> list[str]:
    """
    Capture keyframes from phone stream using scene-change detection.
    A keyframe is saved only when the scene meaningfully changes
    (DINOv2 cosine distance > scene_change_threshold from last saved frame),
    with a minimum time gap of min_interval_sec between saves.

    Args:
        stream_url:               IP Webcam URL or camera index
        output_dir:               Directory to save keyframe JPEGs
        interval_sec:             Max seconds between forced keyframe saves
        max_frames:               Maximum number of keyframes to capture
        scene_change_threshold:   DINOv2 cosine distance to trigger new keyframe
        min_interval_sec:         Minimum seconds between any two saves

    Returns:
        List of saved image file paths
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    source = int(stream_url) if str(stream_url).isdigit() else stream_url
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {stream_url}")

    # Preload DINOv2 before walk starts
    _get_dino_model()

    is_video = isinstance(source, str) and source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))

    saved: list[str] = []
    last_saved_time = 0.0
    last_saved_emb: np.ndarray | None = None
    frame_idx = 0
    
    # Initialize image enhancer
    try:
        with open("config/settings.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load config/settings.yaml, using default enhancement: {e}")
        cfg = {}
    enhancer = ImageEnhancer(config=cfg)

    logger.info("[MapBuilder] Starting smart keyframe capture (scene-change detection active).")
    logger.info(f"[MapBuilder] Threshold: cosine distance > {scene_change_threshold} OR > {interval_sec}s elapsed.")
    logger.info("[MapBuilder] Walk through ALL rooms, corridors, and junctions. Press Ctrl+C to stop.")

    try:
        while len(saved) < max_frames:
            ret, frame = cap.read()
            if not ret:
                if is_video:
                    logger.info("[MapBuilder] End of video file reached.")
                    break
                logger.warning("[MapBuilder] Frame read failed. Reconnecting in 2s...")
                cap.release()
                time.sleep(2)
                if isinstance(source, int):
                    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(source)
                continue

            if is_video:
                now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                now = time.time()
            time_elapsed = now - last_saved_time

            # Too soon — skip entirely
            if time_elapsed < min_interval_sec:
                continue

            # Compute DINOv2 embedding for scene-change detection
            emb = _embed_frame(frame)

            # Decide whether to save
            if last_saved_emb is None:
                should_save = True  # Always save the first frame
            else:
                dist = _cosine_distance(emb, last_saved_emb)
                scene_changed = dist > scene_change_threshold
                time_forced = time_elapsed >= interval_sec
                should_save = scene_changed or time_forced
                if should_save:
                    reason = "scene change" if scene_changed else "time interval"
                    logger.debug(f"[MapBuilder] Saving: {reason} (dist={dist:.3f})")

            if should_save:
                frame_idx += 1
                fname = out_path / f"node_{frame_idx:03d}.jpg"
                
                # Apply enhancement before saving so VLM and map see the clean version
                enhanced_frame = enhancer.enhance(frame)
                cv2.imwrite(str(fname), enhanced_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                
                saved.append(str(fname))
                last_saved_emb = emb
                last_saved_time = now
                logger.info(f"[MapBuilder] Keyframe {frame_idx:3d}: {fname.name} "
                            f"({len(saved)} total)")

    except KeyboardInterrupt:
        logger.info("[MapBuilder] Scan walk complete (Ctrl+C pressed).")
    finally:
        cap.release()

    logger.info(f"[MapBuilder] Captured {len(saved)} keyframes.")
    return saved


# ── Multi-frame clip labeling ──────────────────────────────────────────────────

LABEL_PROMPT = """You are labeling a navigation node in an indoor map for a visually impaired user.
You are seeing 3 consecutive frames of the SAME location from slightly different angles.
Use all 3 images to understand the full spatial context of this area.

Respond ONLY with a valid JSON object (no markdown, no explanation):

{
  "name": "<3-6 word location description e.g. 'main corridor near elevator', 'women restroom entrance', 'stairwell landing floor 2'>",
  "areaType": "<'room' or 'corridor' or 'stairwell' or 'elevator' or 'entrance' or 'open_area'>",
  "level": <floor number as integer, guess from visual cues, default 1 if unsure>,
  "landmarks": ["<visible landmark 1>", "<visible landmark 2>", "<visible landmark 3>"]
}

Be specific. Use language a blind person can understand when heard aloud.
Include at least 2 landmarks (doors, signs, furniture, walls, windows, etc.)."""


def _encode_image(image_path: str, width: int = 480) -> str:
    """Resize and base64-encode a JPEG image for API submission."""
    img = cv2.imread(image_path)
    if img is None:
        # Return blank image if file unreadable
        img = np.zeros((270, 480, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    new_h = int(h * width / w)
    img = cv2.resize(img, (width, new_h))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def label_keyframe(
    image_path: str,
    prev_path: str | None = None,
    next_path: str | None = None,
    retries: int = 3
) -> dict:
    """
    Send a keyframe + its neighbors to the VLM for semantic labeling.
    Uses 3 frames (prev, current, next) for richer spatial context.

    Args:
        image_path: Path to the current keyframe JPEG
        prev_path:  Path to the previous keyframe (optional)
        next_path:  Path to the next keyframe (optional)
        retries:    Number of API retry attempts

    Returns:
        dict with keys: node_id, name, areaType, level, landmarks
    """
    node_id = Path(image_path).stem   # e.g. "node_005"

    # Build multi-image content: use whatever frames are available
    content = [{"type": "text", "text": LABEL_PROMPT}]
    for path in [prev_path, image_path, next_path]:
        if path and Path(path).exists():
            b64 = _encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low"
                }
            })

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="qwen2.5vl:7b",
                max_tokens=300,
                messages=[{"role": "user", "content": content}]
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            # Extract JSON object if there's surrounding text
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group()
            data = json.loads(raw)
            data["node_id"] = node_id
            logger.info(f"[MapBuilder] Labeled {node_id}: {data.get('name', '?')}")
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"[MapBuilder] JSON parse error on attempt {attempt+1}: {e}")
            time.sleep(1)
        except Exception as e:
            logger.warning(f"[MapBuilder] Label attempt {attempt+1} failed: {e}")
            time.sleep(1)

    logger.warning(f"[MapBuilder] Using default label for {node_id}")
    return {
        "node_id": node_id,
        "name": f"unknown area {node_id}",
        "areaType": "corridor",
        "level": 1,
        "landmarks": []
    }


# ── Edge transition description ────────────────────────────────────────────────

EDGE_PROMPT = """You are generating a navigation instruction for a visually impaired person.
You will see TWO images: the CURRENT location (image 1) and the NEXT location (image 2).

In ONE short sentence (max 12 words), describe what physical action the person takes
to move from the current location to the next location.

Examples:
- "Turn left through the open doorway."
- "Walk straight ahead past the elevator."
- "Turn right down the corridor toward the window."
- "Continue forward through the double doors."

Respond with ONLY the instruction sentence, nothing else."""


def generate_edge_description(
    from_image_path: str,
    to_image_path: str,
    retries: int = 2
) -> str:
    """
    Ask the VLM to describe the movement between two adjacent keyframes.
    Returns a short navigation instruction stored in the edge.

    Args:
        from_image_path: Path to the 'from' node keyframe
        to_image_path:   Path to the 'to' node keyframe
        retries:         Number of retry attempts

    Returns:
        Short instruction string e.g. "Turn left through the open doorway."
    """
    content = [{"type": "text", "text": EDGE_PROMPT}]
    for path in [from_image_path, to_image_path]:
        if path and Path(path).exists():
            b64 = _encode_image(path, width=320)   # Smaller for edge queries
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
            })

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="qwen2.5vl:7b",
                max_tokens=40,
                temperature=0.1,
                messages=[{"role": "user", "content": content}]
            )
            desc = response.choices[0].message.content.strip().strip('"').strip("'")
            # Ensure it ends with punctuation
            if desc and not desc[-1] in ".!?":
                desc += "."
            return desc
        except Exception as e:
            logger.warning(f"[MapBuilder] Edge description attempt {attempt+1} failed: {e}")
            time.sleep(1)

    return "Continue forward."


# ── OSMAG graph construction ───────────────────────────────────────────────────

def build_osmag_graph(
    keyframe_dir: str,
    output_path: str,
    label_delay: float = 0.3,
    generate_edge_descriptions: bool = True,
) -> dict:
    """
    Full pipeline:
      1. Find all keyframes in directory
      2. Label each one via VLM (multi-frame clip: prev + current + next)
      3. Build sequential graph (node_N connects to node_N+1)
      4. Generate natural language edge transition descriptions
      5. Add vertical edges for stairwell/elevator nodes on adjacent floors
      6. Save osmag.json

    Args:
        keyframe_dir:               Directory containing node_*.jpg files
        output_path:                Where to save the osmag.json
        label_delay:                Seconds between VLM calls
        generate_edge_descriptions: Whether to generate turn-by-turn edge directions

    Returns:
        The complete OSMAG graph dict
    """
    kf_dir = Path(keyframe_dir)
    image_paths = sorted(kf_dir.glob("node_*.jpg"))

    if not image_paths:
        raise ValueError(f"No keyframes found in {keyframe_dir}.")

    logger.info(f"[MapBuilder] Labeling {len(image_paths)} keyframes (multi-frame clip mode)...")

    # ── Label all keyframes with 3-frame context ──────────────────────────────
    nodes: dict[str, dict] = {}
    paths_list = [str(p) for p in image_paths]

    for i, img_path in enumerate(image_paths):
        logger.info(f"[MapBuilder] Labeling frame {i+1}/{len(image_paths)}: {img_path.name}")

        prev_path = paths_list[i - 1] if i > 0 else None
        next_path = paths_list[i + 1] if i < len(paths_list) - 1 else None

        label = label_keyframe(
            image_path=str(img_path),
            prev_path=prev_path,
            next_path=next_path
        )
        node_id = label["node_id"]
        node_name = label.get("name", f"unknown {node_id}")
        nodes[node_id] = {
            "node_id":  node_id,
            "name":     node_name,
            "areaType": label.get("areaType", "corridor"),
            "level":    label.get("level", 1),
            "landmarks": label.get("landmarks", []),
            "image":    img_path.name
        }
        time.sleep(label_delay)

    # ── Build sequential edges ────────────────────────────────────────────────
    sorted_ids = sorted(nodes.keys())
    edges: list[dict] = []

    for i in range(len(sorted_ids) - 1):
        from_id = sorted_ids[i]
        to_id   = sorted_ids[i + 1]
        edge = {
            "from": from_id,
            "to":   to_id,
            "type": "sequential",
            "description": ""   # Filled in next step
        }
        edges.append(edge)

    # ── Generate edge transition descriptions ─────────────────────────────────
    if generate_edge_descriptions:
        logger.info(f"[MapBuilder] Generating {len(edges)} edge transition descriptions...")
        for j, edge in enumerate(edges):
            from_img = str(kf_dir / nodes[edge["from"]]["image"])
            to_img   = str(kf_dir / nodes[edge["to"]]["image"])
            desc = generate_edge_description(from_img, to_img)
            edge["description"] = desc
            logger.info(f"[MapBuilder] Edge {edge['from']} → {edge['to']}: {desc}")
            time.sleep(0.2)

    # ── Add vertical edges for stairwells/elevators ───────────────────────────
    vertical_keywords = ["stair", "elevator", "lift"]
    for nid, node in nodes.items():
        name_lower = node["name"].lower()
        if any(kw in name_lower for kw in vertical_keywords):
            peers = [
                oid for oid, other in nodes.items()
                if oid != nid
                and any(kw in other["name"].lower() for kw in vertical_keywords)
                and abs(other["level"] - node["level"]) == 1
            ]
            existing_pairs = {(e["from"], e["to"]) for e in edges}
            for oid in peers:
                if (nid, oid) not in existing_pairs and (oid, nid) not in existing_pairs:
                    vedge = {
                        "from": nid,
                        "to":   oid,
                        "type": "vertical",
                        "description": "Take the stairs or elevator to the next floor."
                    }
                    edges.append(vedge)
                    existing_pairs.add((nid, oid))

    osmag = {
        "version":  "2.0",
        "building": Path(output_path).parent.name,
        "nodes":    nodes,
        "edges":    edges
    }


    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(osmag, f, indent=2, ensure_ascii=False)

    logger.success(
        f"[MapBuilder] OSMAG v2.0 saved: {output_path} "
        f"({len(nodes)} nodes, {len(edges)} edges)"
    )
    return osmag
