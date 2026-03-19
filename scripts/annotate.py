"""
scripts/annotate.py

Interactive annotation tool for manually labelling human patches.

Loads patches from an existing classification directory (output/classifications/<run>),
shows the deep diagnostic alongside the rule-based prediction, and lets you accept
or correct each label. State is saved to a JSON file after every annotation so
you can quit and resume at any time.

Usage
-----
    python3 scripts/annotate.py --cls-dir output/classifications/20260314-195748

    # Resume a partially-annotated session
    python3 scripts/annotate.py --cls-dir output/classifications/20260314-195748 \\
        --out-dir output/manual_annotated/20260314-195748

Keys
----
    y          Accept the rule-based prediction and move to the next image.
    n          Reject — shows reclassification prompt in the terminal.
    b          Go back one image (re-opens the previous image for correction).
    q          Quit and save progress.

    When reclassifying (after pressing n):
    1  full_body_front
    2  full_body_back
    3  head_shoulder_front
    4  head_shoulder_back
    5  others
    6  bad_extraction
    b  Cancel reclassification and go back to current image.

Output
------
    output/manual_annotated/<run>/annotations.json
        { filename: { "label": str, "source": "auto" | "manual" } }

    output/manual_annotated/<run>/<class>/
        Copies of annotated patches in per-class subdirs, mirroring the
        classification output format so downstream pipeline can consume them
        without changes.  Written on exit (or can be triggered mid-session
        with Ctrl+C — handled gracefully).
"""

import argparse
import json
import os
import shutil
import signal
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or scripts/
# ---------------------------------------------------------------------------

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.classification import (
    CLASSES,
    DEFAULT_CONFIG,
    classify_keypoints,
    render_diagnostic,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_LABELS = CLASSES + ['bad_extraction']  # 6 total

LABEL_KEYS = {
    '1': 'full_body_front',
    '2': 'full_body_back',
    '3': 'head_shoulder_front',
    '4': 'head_shoulder_back',
    '5': 'others',
    '6': 'bad_extraction',
}

WINDOW_NAME = "Annotator  —  y=accept  n=reclassify  b=back  q=quit"

# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_annotations(json_path: str) -> dict:
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return {}


def save_annotations(json_path: str, annotations: dict) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=2)


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------

def collect_patches(cls_dir: str) -> list[tuple[str, str]]:
    """
    Walk a classification output directory and return a list of
    (absolute_path, rule_based_label) tuples, sorted by filename.

    Includes all five CLASSES subdirs.  Does not include debug_viz.
    """
    patches = []
    for cls in CLASSES:
        subdir = os.path.join(cls_dir, cls)
        if not os.path.isdir(subdir):
            continue
        for fname in sorted(os.listdir(subdir)):
            if fname.lower().endswith(('.jpg', '.png')):
                patches.append((os.path.join(subdir, fname), cls))
    patches.sort(key=lambda x: os.path.basename(x[0]))
    return patches


# ---------------------------------------------------------------------------
# Diagnostic rendering (re-runs YOLO-pose on the patch)
# ---------------------------------------------------------------------------

def build_display_image(
    img_bgr: np.ndarray,
    pose_model,
    predicted_label: str,
    cfg=DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Run YOLO-pose on img_bgr and render the deep diagnostic.
    Returns a BGR display image.
    """
    result = pose_model(img_bgr, verbose=False)[0]
    return render_diagnostic(result, img_bgr, cfg, predicted_class=predicted_label)


# ---------------------------------------------------------------------------
# Output: copy annotated files into per-class subdirs
# ---------------------------------------------------------------------------

def write_output_dirs(
    patches: list[tuple[str, str]],
    annotations: dict,
    out_dir: str,
) -> None:
    """
    Copy each annotated patch into out_dir/<label>/<filename>.
    Only copies patches that appear in annotations.
    Skips files that already exist at the destination.
    """
    for label in ALL_LABELS:
        os.makedirs(os.path.join(out_dir, label), exist_ok=True)

    copied = 0
    for src_path, _ in patches:
        fname = os.path.basename(src_path)
        entry = annotations.get(fname)
        if entry is None:
            continue
        label = entry["label"]
        dst   = os.path.join(out_dir, label, fname)
        if not os.path.exists(dst):
            shutil.copy2(src_path, dst)
            copied += 1

    print(f"Wrote {copied} new files to {out_dir}")


# ---------------------------------------------------------------------------
# Terminal prompt for reclassification
# ---------------------------------------------------------------------------

def prompt_reclassify(current_label: str) -> str | None:
    """
    Show reclassification options in the terminal.
    Returns the chosen label string, or None if the user cancels.
    """
    print("\n  Reclassify as:")
    for key, label in LABEL_KEYS.items():
        marker = " ← current" if label == current_label else ""
        print(f"    {key}  {label}{marker}")
    print("    b  cancel")
    while True:
        choice = input("  Choice: ").strip().lower()
        if choice == 'b':
            return None
        if choice in LABEL_KEYS:
            return LABEL_KEYS[choice]
        print("  Invalid key. Try again.")


# ---------------------------------------------------------------------------
# Main annotation loop
# ---------------------------------------------------------------------------

def run_annotation(
    cls_dir: str,
    out_dir: str,
    pose_model_path: str,
    cfg=DEFAULT_CONFIG,
) -> None:
    patches = collect_patches(cls_dir)
    if not patches:
        print(f"No patches found in {cls_dir}")
        return

    json_path   = os.path.join(out_dir, "annotations.json")
    annotations = load_annotations(json_path)

    # Build ordered list of filenames not yet annotated
    all_fnames  = [os.path.basename(p) for p, _ in patches]
    fname_to_patch = {os.path.basename(p): (p, lbl) for p, lbl in patches}

    todo = [f for f in all_fnames if f not in annotations]
    done = [f for f in all_fnames if f in annotations]

    print(f"\nLoaded {len(patches)} patches from {cls_dir}")
    print(f"  Already annotated: {len(done)}")
    print(f"  Remaining:         {len(todo)}")
    print(f"  Output dir:        {out_dir}")
    print(f"\nKeys: y=accept  n=reclassify  b=back  q=quit\n")

    if not todo:
        print("All patches already annotated.")
        write_output_dirs(patches, annotations, out_dir)
        return

    # Load YOLO-pose model (CPU is fine for annotation)
    from ultralytics import YOLO
    pose_model = YOLO(pose_model_path)
    # Deliberately keep on CPU for local use

    # Handle Ctrl+C gracefully
    interrupted = [False]
    def _sigint_handler(sig, frame):
        interrupted[0] = True
    signal.signal(signal.SIGINT, _sigint_handler)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1400, 500)

    # Work queue — a stack so 'b' can re-insert
    queue = list(todo)  # index 0 = next to annotate
    idx   = 0           # position in queue

    while idx < len(queue) and not interrupted[0]:
        fname             = queue[idx]
        src_path, rule_lbl = fname_to_patch[fname]

        # Current label: use existing annotation if re-visiting via 'b'
        current_lbl = annotations.get(fname, {}).get("label", rule_lbl)

        img_bgr = cv2.imread(src_path)
        if img_bgr is None:
            print(f"  Warning: could not read {src_path}, skipping.")
            idx += 1
            continue

        # Build diagnostic display
        total_done = len([f for f in all_fnames if f in annotations])
        progress   = f"[{total_done}/{len(patches)}]  {fname}"
        print(f"\n{progress}")
        print(f"  Rule-based: {rule_lbl}   Current: {current_lbl}")

        diag = build_display_image(img_bgr, pose_model, current_lbl, cfg)
        cv2.imshow(WINDOW_NAME, diag)

        # Key loop for this image
        while not interrupted[0]:
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q'):
                interrupted[0] = True
                break

            elif key == ord('y'):
                source = "auto" if current_lbl == rule_lbl else "manual"
                annotations[fname] = {"label": current_lbl, "source": source}
                save_annotations(json_path, annotations)
                print(f"  ✓ accepted: {current_lbl}")
                idx += 1
                break

            elif key == ord('n'):
                new_lbl = prompt_reclassify(current_lbl)
                if new_lbl is not None:
                    annotations[fname] = {"label": new_lbl, "source": "manual"}
                    save_annotations(json_path, annotations)
                    print(f"  ✓ reclassified: {current_lbl} → {new_lbl}")
                    idx += 1
                    # Refresh display with new label
                    diag = build_display_image(img_bgr, pose_model, new_lbl, cfg)
                    cv2.imshow(WINDOW_NAME, diag)
                else:
                    print("  Reclassification cancelled.")
                break

            elif key == ord('b'):
                if idx > 0:
                    idx -= 1
                    prev_fname = queue[idx]
                    # Remove annotation for previous image so it re-appears
                    if prev_fname in annotations:
                        del annotations[prev_fname]
                        save_annotations(json_path, annotations)
                    print(f"  ← going back to {prev_fname}")
                else:
                    print("  Already at the first image.")
                break

    cv2.destroyAllWindows()

    # Final save + copy to output dirs
    save_annotations(json_path, annotations)
    total_done = len([f for f in all_fnames if f in annotations])
    print(f"\nSession ended. Annotated {total_done}/{len(patches)} patches.")
    print(f"Annotations saved to {json_path}")

    write_output_dirs(patches, annotations, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_out_dir(cls_dir: str) -> str:
    run_name = os.path.basename(os.path.normpath(cls_dir))
    return os.path.join(PROJECT_ROOT, "output", "manual_annotated", run_name)


def _default_pose_model(project_root: str) -> str:
    return os.path.join(project_root, "models", "yolo26m-pose.pt")


def main():
    parser = argparse.ArgumentParser(
        description="Interactively annotate human patches from a classification run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cls-dir", required=True,
        help="Path to a classification output directory (e.g. output/classifications/20260314-195748).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write annotations.json and per-class output dirs. "
             "Defaults to output/manual_annotated/<run_name>.",
    )
    parser.add_argument(
        "--pose-model", default=None,
        help="Path to YOLO-pose weights. Defaults to models/yolo26m-pose.pt.",
    )
    args = parser.parse_args()

    cls_dir    = os.path.abspath(args.cls_dir)
    out_dir    = os.path.abspath(args.out_dir) if args.out_dir else _default_out_dir(cls_dir)
    pose_model = args.pose_model or _default_pose_model(PROJECT_ROOT)

    if not os.path.isdir(cls_dir):
        print(f"Error: cls-dir not found: {cls_dir}")
        sys.exit(1)
    if not os.path.exists(pose_model):
        print(f"Error: pose model not found: {pose_model}")
        sys.exit(1)

    run_annotation(cls_dir, out_dir, pose_model)


if __name__ == "__main__":
    main()