"""
quote - A patch is classified as front if YOLO detects nose and both eyes above a confidence threshold, 
back if all three are absent, and others otherwise — correctly routing side profiles, 
which produce partial face detections, into the ambiguous class.

1.2
classification.py

Pose-based human patch classification into five categories:
  full_body_front, full_body_back, head_shoulder_front, head_shoulder_back, others

Classification logic
--------------------
Orientation (applied first):
  front  — nose AND both eyes visible above confidence threshold.
           Requires bilateral facial symmetry; a side profile will typically
           missing one eye and fail this test.
  back   — none of nose, left_eye, right_eye visible even at low confidence.
           A side profile produces at least a partial nose detection, so it
           falls through to 'others' rather than being misclassified as back.
  others — anything else (side-on, occluded, ambiguous).

Extent (applied only once orientation is decided):
  full_body     — lower-body keypoints (hips/knees/ankles) visible, corroborated
                  by bounding-box aspect ratio h/w >= 1.5.
  head_shoulder — shoulders or upper-body keypoints visible, lower-body absent.
  others        — insufficient keypoints to determine extent.

COCO keypoint indices (17 points):
  0: nose          1: left_eye      2: right_eye
  3: left_ear      4: right_ear     5: left_shoulder
  6: right_shoulder  7: left_elbow  8: right_elbow
  9: left_wrist   10: right_wrist  11: left_hip
 12: right_hip    13: left_knee    14: right_knee
 15: left_ankle   16: right_ankle
"""

import glob
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keypoint confidence thresholds.
# FACE_CONF: minimum confidence to consider a face keypoint present.
# BACK_CONF:  maximum confidence below which a face keypoint is considered
#             absent — slightly relaxed to absorb YOLO noise on low-quality frames.
FACE_CONF = 0.30
BACK_CONF = 0.10  # all face KPs must be below this to call 'back'

# Minimum confidence for lower/upper body keypoints.
BODY_CONF = 0.20

CLASSES = [
    'full_body_front',
    'full_body_back',
    'head_shoulder_front',
    'head_shoulder_back',
    'others',
]

# COCO index groups
FACE_KPS   = [0, 1, 2]        # nose, left_eye, right_eye
LOWER_KPS  = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
ANKLE_KPS  = [15, 16]
UPPER_KPS  = [5, 6]            # shoulders (minimum upper-body signal)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def n_visible(keypoints: np.ndarray, indices: list[int], threshold: float) -> int:
    return sum(1 for i in indices if keypoints[i, 2] >= threshold)


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def classify_orientation(keypoints: np.ndarray) -> str | None:
    """
    Returns 'front', 'back', or None (side-on / ambiguous → 'others').

    front: nose AND both eyes above FACE_CONF.
    back:  all of nose, left_eye, right_eye below BACK_CONF.
    None:  partial face visibility — side profile or occluded.
    """
    nose_conf    = keypoints[0, 2]
    left_eye_conf  = keypoints[1, 2]
    right_eye_conf = keypoints[2, 2]

    all_face_present = (
        nose_conf      >= FACE_CONF and
        left_eye_conf  >= FACE_CONF and
        right_eye_conf >= FACE_CONF
    )
    all_face_absent = (
        nose_conf      < BACK_CONF and
        left_eye_conf  < BACK_CONF and
        right_eye_conf < BACK_CONF
    )

    if all_face_present:
        return 'front'
    if all_face_absent:
        return 'back'
    return None  # partial — side-on or ambiguous


# ---------------------------------------------------------------------------
# Body extent
# ---------------------------------------------------------------------------

def classify_extent(keypoints: np.ndarray, bbox: np.ndarray | None) -> str | None:
    """
    Returns 'full_body', 'head_shoulder', or None (insufficient keypoints).

    full_body requires lower-body evidence AND a plausible aspect ratio.
    head_shoulder requires at least one shoulder visible.
    """
    n_lower  = n_visible(keypoints, LOWER_KPS, BODY_CONF)
    n_ankles = n_visible(keypoints, ANKLE_KPS, BODY_CONF)

    has_lower_body = n_ankles >= 1 or n_lower >= 3

    if has_lower_body:
        # Aspect-ratio sanity check: a standing full-body patch should be
        # clearly taller than wide.  Fails for crouching/seated subjects but
        # avoids mislabelling wide head-shoulder crops.
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            h = y2 - y1
            w = (x2 - x1) + 1e-6
            if h / w < 1.5:
                return 'head_shoulder'
        return 'full_body'

    if n_visible(keypoints, UPPER_KPS, BODY_CONF) >= 1:
        return 'head_shoulder'

    return None  # not enough to decide


# ---------------------------------------------------------------------------
# Top-level per-patch classifier
# ---------------------------------------------------------------------------

def classify_keypoints(keypoints: np.ndarray, bbox: np.ndarray | None = None) -> str:
    """
    Classify a single set of 17 COCO keypoints (numpy array [17, 3]).
    Returns one of the five CLASSES strings.
    """
    orientation = classify_orientation(keypoints)
    if orientation is None:
        return 'others'

    extent = classify_extent(keypoints, bbox)
    if extent is None:
        return 'others'

    return f"{extent}_{orientation}"


# ---------------------------------------------------------------------------
# Batched directory classification
# ---------------------------------------------------------------------------

def classify_directory(
    pose_model,
    input_dir: str,
    output_dir: str,
    batch_size: int = 32,
    copy_files: bool = True,
    save_debug_viz: bool = False,
) -> tuple[dict, dict]:
    """
    Classify all .jpg/.png files in input_dir using batched YOLO inference.

    Args:
        pose_model:     YOLOv8 pose model instance.
        input_dir:      Directory of cropped human patches.
        output_dir:     Root output directory; per-class subdirs created automatically.
        batch_size:     Images per inference call.
        copy_files:     If True, copy patches into per-class subdirs.
        save_debug_viz: If True, save YOLO-annotated images to output_dir/debug_viz/.

    Returns:
        results: dict mapping filename -> class string.
        summary: dict mapping class string -> count.
    """
    image_paths = sorted(
        glob.glob(os.path.join(input_dir, '*.jpg')) +
        glob.glob(os.path.join(input_dir, '*.png'))
    )
    if not image_paths:
        print(f"No images found in {input_dir}")
        return {}, {}

    if copy_files:
        for cls in CLASSES:
            os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    if save_debug_viz:
        os.makedirs(os.path.join(output_dir, 'debug_viz'), exist_ok=True)

    results = {}
    summary = {cls: 0 for cls in CLASSES}
    n_batches = (len(image_paths) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(image_paths), batch_size), total=n_batches, desc="Classifying", unit="batch"):
        batch_paths   = image_paths[i: i + batch_size]
        batch_results = pose_model(batch_paths, verbose=False)

        for img_path, result in zip(batch_paths, batch_results):
            fname = os.path.basename(img_path)

            if result.keypoints is None or result.keypoints.data.shape[0] == 0:
                cls = 'others'
            else:
                boxes    = result.boxes.xyxy.cpu().numpy()
                areas    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_idx = int(np.argmax(areas))
                cls = classify_keypoints(
                    result.keypoints.data[best_idx].cpu().numpy(),
                    bbox=boxes[best_idx],
                )

            results[fname]  = cls
            summary[cls]   += 1

            if copy_files:
                shutil.copy(img_path, os.path.join(output_dir, cls, fname))

            if save_debug_viz:
                cv2.imwrite(
                    os.path.join(output_dir, 'debug_viz', fname),
                    result.plot(),
                )

    print(f"\nDone. {dict((k, v) for k, v in summary.items() if v > 0)}")
    return results, summary


# ---------------------------------------------------------------------------
# Reload from existing output directory
# ---------------------------------------------------------------------------

def reload_classification_results(cls_save_path: str) -> tuple[dict, dict]:
    """
    Reconstruct results and summary from an existing classification directory.
    """
    results = {}
    summary = {cls: 0 for cls in CLASSES}

    for cls in CLASSES:
        cls_dir = os.path.join(cls_save_path, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.png')):
                results[fname]  = cls
                summary[cls]   += 1

    total = sum(summary.values())
    print(f"Loaded {total} patches from {cls_save_path}")
    print("  " + "  ".join(f"{k}: {v}" for k, v in summary.items() if v > 0))
    return results, summary