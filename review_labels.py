"""
review_labels.py

Interactive label review tool.
Opens each keyframe image in a window alongside its VLM-generated label so
you can visually assess accuracy. Press:
  [y] or [Enter] = Good label
  [n]            = Bad label (saves to review_results.json)
  [q]            = Quit early

Usage:
    python review_labels.py --building mybuilding_v2
"""

import json
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np


def put_text_wrapped(img, text, start_y, max_width, font_scale=0.7, color=(255, 255, 255), thickness=2):
    """Draw word-wrapped text on an image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = current + (" " if current else "") + word
        w, _ = cv2.getTextSize(test, font, font_scale, thickness)[0]
        if w > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)

    y = start_y
    for line in lines:
        cv2.putText(img, line, (15, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += 30
    return y


def main():
    parser = argparse.ArgumentParser(description="Review VLM-generated keyframe labels.")
    parser.add_argument("--building", required=True, help="Building name (e.g. mybuilding_v2)")
    args = parser.parse_args()

    map_dir     = Path("maps") / args.building
    osmag_path  = map_dir / "osmag.json"
    kf_dir      = map_dir / "keyframes"

    if not osmag_path.exists():
        print(f"[ERROR] No map found at {osmag_path}. Run scan_walk.py first.")
        sys.exit(1)

    with open(osmag_path, encoding="utf-8") as f:
        osmag = json.load(f)

    nodes = osmag["nodes"]
    node_list = sorted(nodes.items())   # sorted by node_id

    print(f"\n{'='*55}")
    print(f"  Label Review Tool — {args.building}")
    print(f"  {len(node_list)} nodes to review")
    print(f"  Controls: [y/Enter]=Good  [n]=Bad  [q]=Quit")
    print(f"{'='*55}\n")

    results = {"good": [], "bad": [], "skipped": []}
    img_w = 900

    for idx, (node_id, node) in enumerate(node_list):
        img_path = kf_dir / node.get("image", f"{node_id}.jpg")
        if not img_path.exists():
            print(f"  [{idx+1}/{len(node_list)}] {node_id}: image not found, skipping.")
            results["skipped"].append(node_id)
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            results["skipped"].append(node_id)
            continue

        # Resize frame to fixed width
        h, w = frame.shape[:2]
        new_h = int(h * img_w / w)
        frame = cv2.resize(frame, (img_w, new_h))

        # Build info panel below the image
        panel_h = 160
        panel = np.zeros((panel_h, img_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        # Header
        header = f"[{idx+1}/{len(node_list)}]  {node_id}"
        cv2.putText(panel, header, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 200, 255), 2, cv2.LINE_AA)

        # Name label (main result to evaluate)
        name_text = f"Label : {node['name']}"
        put_text_wrapped(panel, name_text, 65, img_w - 20, font_scale=0.75, color=(50, 220, 120))

        # Area type + floor
        meta = f"Type  : {node.get('areaType','?')}   Floor: {node.get('level','?')}"
        cv2.putText(panel, meta, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)

        # Landmarks
        lm = "Landmarks: " + ", ".join(node.get("landmarks", [])) if node.get("landmarks") else "Landmarks: (none)"
        put_text_wrapped(panel, lm, 130, img_w - 20, font_scale=0.52, color=(170, 170, 170), thickness=1)

        # Key hint bar
        hint = np.zeros((40, img_w, 3), dtype=np.uint8)
        hint[:] = (50, 50, 50)
        cv2.putText(hint, "  [Y / Enter] = Correct        [N] = Wrong        [Q] = Quit",
                    (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 100), 1, cv2.LINE_AA)

        combined = np.vstack([frame, panel, hint])
        cv2.imshow("Label Review", combined)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('y'), ord('Y'), 13):   # y or Enter
                results["good"].append(node_id)
                print(f"  [{idx+1}] ✓ GOOD  — {node_id}: '{node['name']}'")
                break
            elif key in (ord('n'), ord('N')):
                results["bad"].append(node_id)
                print(f"  [{idx+1}] ✗ BAD   — {node_id}: '{node['name']}'")
                break
            elif key in (ord('q'), ord('Q'), 27):  # q or Esc
                print("\n  Review stopped early.")
                cv2.destroyAllWindows()
                _save_results(results, args.building)
                return

    cv2.destroyAllWindows()
    _save_results(results, args.building)


def _save_results(results, building):
    total = len(results["good"]) + len(results["bad"]) + len(results["skipped"])
    if total == 0:
        return
    reviewed = len(results["good"]) + len(results["bad"])
    accuracy = len(results["good"]) / reviewed * 100 if reviewed > 0 else 0.0

    print(f"\n{'='*55}")
    print(f"  Review Complete — {building}")
    print(f"  Reviewed : {reviewed}  |  Good: {len(results['good'])}  |  Bad: {len(results['bad'])}")
    print(f"  Accuracy : {accuracy:.1f}%")
    if results["bad"]:
        print(f"\n  Bad labels:")
        for nid in results["bad"]:
            print(f"    - {nid}")
    print(f"{'='*55}")

    out_path = Path("maps") / building / "label_review.json"
    results["accuracy_pct"] = round(accuracy, 1)
    results["reviewed"] = reviewed
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
