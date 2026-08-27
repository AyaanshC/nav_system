"""
scan_walk.py

Run ONCE per building to build the navigation map.

Usage:
    python scan_walk.py --building mybuilding --stream http://192.168.1.5:8080/video
    python scan_walk.py --building mybuilding --stream 1   (Iriun/local camera)

Steps:
    1. Walk through the building slowly (~0.5 m/s)
    2. Keyframes are saved automatically using scene-change detection
       (only saves when the scene changes significantly, not on a fixed timer)
    3. Press Ctrl+C when you have covered the entire building
    4. Script auto-labels all keyframes via Moondream (multi-frame clip mode)
    5. Edge transition descriptions are generated between adjacent nodes
    6. DINOv2 FAISS index is built for fast localization during navigation
    7. osmag.json v2.0 is saved to maps/<building>/osmag.json
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from modules.m3_map_builder import extract_keyframes, build_osmag_graph, _get_dino_model
from modules.m2_dino_localizer import DINOv2Localizer


def main():
    parser = argparse.ArgumentParser(
        description="Scan walk — build the indoor navigation map for a building."
    )
    parser.add_argument(
        "--building", required=True,
        help="Building name (used as subdirectory name under maps/)"
    )
    parser.add_argument(
        "--stream", required=True,
        help="Phone IP Webcam URL, camera index (e.g. 1), or path to a recorded video (e.g. video.mp4)"
    )
    parser.add_argument(
        "--interval", type=float, default=0.8,
        help="Max seconds between keyframe saves (default: 0.8 — uses scene-change detection)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.12,
        help="Scene-change sensitivity (default: 0.12, lower = more keyframes)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=300,
        help="Maximum number of keyframes to capture (default: 300)"
    )
    parser.add_argument(
        "--skip-edge-descriptions", action="store_true",
        help="Skip VLM edge description generation (faster, but less accurate navigation)"
    )
    args = parser.parse_args()

    map_dir    = Path("maps") / args.building
    kf_dir     = map_dir / "keyframes"
    osmag_path = map_dir / "osmag.json"

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    if osmag_path.exists():
        logger.warning(f"Map already exists at {osmag_path}.")
        answer = input("Overwrite? (y/N): ").strip().lower()
        if answer != "y":
            logger.info("Cancelled.")
            sys.exit(0)

    logger.info(f"=== SCAN WALK v2.0 for building: {args.building} ===")
    logger.info(f"Stream: {args.stream}")
    logger.info(f"Scene-change threshold: {args.threshold} (DINOv2 cosine distance)")
    logger.info(f"Max interval: {args.interval}s")
    logger.info("─" * 55)
    logger.info("Walk through the building slowly.")
    logger.info("Cover ALL rooms, corridors, junctions, and stairwells.")
    logger.info("The system automatically detects scene changes — no need")
    logger.info("to stop at each location.")
    logger.info("Press Ctrl+C when done.")
    logger.info("─" * 55)

    # ── Step 1: Capture keyframes (scene-change detection) ────────────────────
    saved = extract_keyframes(
        stream_url=args.stream,
        output_dir=str(kf_dir),
        interval_sec=args.interval,
        max_frames=args.max_frames,
        scene_change_threshold=args.threshold,
    )

    if not saved:
        logger.error("No keyframes captured. Check your stream URL and camera connection.")
        sys.exit(1)

    logger.info(f"✓ Captured {len(saved)} keyframes.")

    # ── Step 2: Build OSMAG graph with multi-frame labeling ───────────────────
    logger.info("=== Auto-labeling keyframes (multi-frame clip mode) ===")
    logger.info("Each keyframe is labeled using 3 consecutive frames for richer context...")

    try:
        build_osmag_graph(
            keyframe_dir=str(kf_dir),
            output_path=str(osmag_path),
            label_delay=0.3,
            generate_edge_descriptions=not args.skip_edge_descriptions,
        )
    except Exception as e:
        logger.error(f"Map building failed: {e}")
        sys.exit(1)

    # ── Step 3: Build DINOv2 FAISS index ──────────────────────────────────────
    logger.info("=== Building DINOv2 FAISS localization index ===")
    logger.info("This enables sub-meter localization during navigation...")
    try:
        # Reuse the already-loaded DINOv2 model to avoid double VRAM usage
        shared_model = _get_dino_model()
        localizer = DINOv2Localizer(map_dir=str(map_dir), device="cuda", model=shared_model)
        localizer.build_index(keyframe_dir=str(kf_dir))
    except Exception as e:
        logger.error(f"DINOv2 index build failed: {e}")
        logger.warning("Navigation will still work but localization may be less accurate.")

    # ── Done ──────────────────────────────────────────────────────────────────
    logger.success("=" * 55)
    logger.success(f"Map complete! Building: {args.building}")
    logger.success(f"OSMAG: {osmag_path}")
    logger.success(f"DINOv2 index: {map_dir / 'dino_index.faiss'}")
    logger.success("=" * 55)
    logger.info(f"Now run: python main.py --building {args.building} --stream <your_stream>")


if __name__ == "__main__":
    main()
