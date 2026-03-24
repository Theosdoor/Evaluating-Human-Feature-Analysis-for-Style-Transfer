"""
scripts/add_annotation.py

Add one or more images to a manual annotation set by running the full
YOLO-detect → bbox-crop → 256×256 resize → save pipeline.

Usage
-----
    # Single image, auto-pick highest-confidence detection:
    python3 scripts/add_annotation.py downloaded_data/Train/game/MafiaVideogame.png \\
        --label head_shoulder_front

    # Multiple images at once:
    python3 scripts/add_annotation.py img1.png img2.png --label others

    # Target a specific annotation run (default: most recent in output/manual_annotated/):
    python3 scripts/add_annotation.py img.png --label full_body_front \\
        --anno-run 20260314-195748-6

    # Show all detections and pick interactively rather than auto-selecting:
    python3 scripts/add_annotation.py img.png --label head_shoulder_front --interactive

Output
------
    Patch saved to  output/manual_annotated/<run>/<label>/<filename>.jpg
    annotations.json updated with {"label": ..., "source": "manual"}
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
CLASSES = ["full_body_front", "full_body_back",
           "head_shoulder_front", "head_shoulder_back", "others"]
PATCH_SIZE   = 256
JPEG_QUALITY = 95

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR     = os.path.join(PROJECT_ROOT, "output")
ANNO_ROOT    = os.path.join(SAVE_DIR, "manual_annotated")
DETECT_MODEL = os.path.join(PROJECT_ROOT, "models", "yolov8m.pt")


# ---------------------------------------------------------------------------

def latest_anno_run(anno_root: str) -> str:
    runs = sorted(
        d for d in os.listdir(anno_root)
        if os.path.isdir(os.path.join(anno_root, d))
    )
    if not runs:
        raise FileNotFoundError(f"No annotation runs found in {anno_root}")
    return runs[-1]


def load_annotations(anno_dir: str) -> dict:
    path = os.path.join(anno_dir, "annotations.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_annotations(anno_dir: str, annotations: dict) -> None:
    path = os.path.join(anno_dir, "annotations.json")
    with open(path, "w") as f:
        json.dump(annotations, f, indent=2)


def detect_persons(img: np.ndarray, model_path: str) -> list[dict]:
    """Run YOLO person detection; return list of {conf, bbox, patch}."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    results = model(img, classes=[0], verbose=False)
    detections = []
    h, w = img.shape[:2]
    for box in (results[0].boxes or []):
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = img[y1:y2, x1:x2]
        detections.append({"conf": conf, "bbox": (x1, y1, x2, y2), "patch": patch})
    detections.sort(key=lambda d: -d["conf"])
    return detections


def make_filename(src_path: str, det_idx: int, label: str) -> str:
    base = os.path.splitext(os.path.basename(src_path))[0]
    return f"manual_{base}_d{det_idx:02d}_{label}.jpg"


def process_image(
    img_path: str,
    label: str,
    anno_dir: str,
    annotations: dict,
    interactive: bool,
) -> int:
    """Detect, crop, and save patches from a single image. Returns count added."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[add_annotation] ERROR: could not read {img_path}", file=sys.stderr)
        return 0

    print(f"[add_annotation] Detecting persons in {os.path.basename(img_path)} …")
    detections = detect_persons(img, DETECT_MODEL)

    if not detections:
        print(f"[add_annotation] No persons detected — skipping.")
        return 0

    print(f"[add_annotation] Found {len(detections)} person(s):")
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["bbox"]
        print(f"  [{i}] conf={d['conf']:.3f}  bbox=({x1},{y1})->({x2},{y2})")

    if interactive:
        sel = input(
            f"  Select detection index (0-{len(detections)-1}), "
            "comma-separated for multiple, or 'a' for all: "
        ).strip()
        if sel.lower() == "a":
            chosen = list(range(len(detections)))
        else:
            chosen = [int(x) for x in sel.split(",") if x.strip().isdigit()]
    else:
        chosen = [0]   # highest-confidence detection
        print(f"  Auto-selecting detection [0] (highest conf).")

    cls_dir = os.path.join(anno_dir, label)
    os.makedirs(cls_dir, exist_ok=True)

    added = 0
    for idx in chosen:
        if idx < 0 or idx >= len(detections):
            print(f"  [add_annotation] Index {idx} out of range, skipping.")
            continue
        patch_256 = cv2.resize(detections[idx]["patch"], (PATCH_SIZE, PATCH_SIZE))
        fname = make_filename(img_path, idx, label)
        out_path = os.path.join(cls_dir, fname)
        cv2.imwrite(out_path, patch_256, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        annotations[fname] = {"label": label, "source": "manual"}
        print(f"  → saved {fname}")
        added += 1

    return added


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Add images to a manual annotation set.")
    parser.add_argument("images", nargs="+", help="Image file(s) to annotate.")
    parser.add_argument(
        "--label", required=True, choices=CLASSES,
        help="Ground-truth class label for the detected person.",
    )
    parser.add_argument(
        "--anno-run", default=None,
        help="Annotation run name (default: most recent in output/manual_annotated/).",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Prompt to choose which detection(s) to keep (default: highest-conf only).",
    )
    args = parser.parse_args()

    run_name = args.anno_run or latest_anno_run(ANNO_ROOT)
    anno_dir = os.path.join(ANNO_ROOT, run_name)
    print(f"[add_annotation] Target annotation run: {run_name}")

    annotations = load_annotations(anno_dir)
    total_added = 0

    for img_path in args.images:
        total_added += process_image(img_path, args.label, anno_dir, annotations, args.interactive)

    save_annotations(anno_dir, annotations)
    print(f"[add_annotation] Done. Added {total_added} patch(es). "
          f"Total annotations: {len(annotations)}")


if __name__ == "__main__":
    main()
