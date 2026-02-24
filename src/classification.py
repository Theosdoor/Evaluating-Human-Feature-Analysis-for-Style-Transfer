"""
classification.py

Pose-based human patch classification into five categories:
  full_body_front, full_body_back, head_shoulder_front, head_shoulder_back, others

Key improvements over original:
  - Batched YOLO inference rather than one image at a time
  - Front/back uses a weighted evidence system rather than a single geometry check
  - Confidence-aware fallback: ambiguous cases go to 'others' rather than
    defaulting to front, which inflates that class
  - classify_directory() returns per-image results and a summary dict

COCO keypoint indices (17 points):
  0: nose          1: left_eye      2: right_eye
  3: left_ear      4: right_ear     5: left_shoulder
  6: right_shoulder  7: left_elbow  8: right_elbow
  9: left_wrist   10: right_wrist  11: left_hip
 12: right_hip    13: left_knee    14: right_knee
 15: left_ankle   16: right_ankle
"""

import os
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONF_HIGH = 0.35      # Confident keypoint detection
CONF_LOW  = 0.15      # Marginal — used only as tiebreaker evidence

CLASSES = [
    'full_body_front',
    'full_body_back',
    'head_shoulder_front',
    'head_shoulder_back',
    'others',
]


# ---------------------------------------------------------------------------
# Keypoint helpers
# ---------------------------------------------------------------------------

def kp(keypoints, idx, threshold=CONF_HIGH):
    """
    Return (x, y, conf) for keypoint `idx`.
    If confidence is below threshold, returns (0, 0, 0) to signal invisible.
    """
    x, y, c = keypoints[idx]
    if c < threshold:
        return (0.0, 0.0, 0.0)
    return (float(x), float(y), float(c))


def is_visible(keypoints, idx, threshold=CONF_HIGH):
    return float(keypoints[idx][2]) >= threshold


def count_visible(keypoints, indices, threshold=CONF_HIGH):
    return sum(1 for i in indices if is_visible(keypoints, i, threshold))


# ---------------------------------------------------------------------------
# Body extent classification
# ---------------------------------------------------------------------------

def classify_body_extent(keypoints):
    """
    Returns 'full_body', 'head_shoulder', or None (insufficient keypoints).

    Full body requires evidence of lower limbs.
    Head-shoulder requires at least one shoulder visible.
    """
    # Lower body evidence
    lower_indices = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
    ankle_indices = [15, 16]

    n_lower = count_visible(keypoints, lower_indices)
    n_ankles = count_visible(keypoints, ankle_indices)
    n_lower_low = count_visible(keypoints, lower_indices, CONF_LOW)

    # Strong case: ankles visible
    if n_ankles >= 1:
        return 'full_body'

    # Moderate case: hips + at least one knee
    knee_hips = count_visible(keypoints, [11, 12, 13, 14])
    if knee_hips >= 3:
        return 'full_body'

    # Weak case: two or more lower body points at low confidence
    if n_lower >= 2 or n_lower_low >= 3:
        return 'full_body'

    # Upper body check
    upper_indices = [0, 1, 2, 3, 4, 5, 6]  # face + shoulders
    n_upper = count_visible(keypoints, upper_indices)
    n_shoulders = count_visible(keypoints, [5, 6])

    if n_shoulders >= 1 or n_upper >= 2:
        return 'head_shoulder'

    return None  # not enough to classify


# ---------------------------------------------------------------------------
# Front / back classification
# ---------------------------------------------------------------------------

def classify_orientation(keypoints):
    """
    Returns 'front', 'back', or None (ambiguous).

    Uses a weighted evidence accumulation approach:
      - Face features (nose, eyes, ears) are strong front evidence
      - Structural geometry (ear/shoulder positioning) provides secondary signal
      - Avoids defaulting to front when evidence is genuinely ambiguous
    """
    front_score = 0.0
    back_score  = 0.0

    # --- Primary evidence: face features ---
    face_indices = [0, 1, 2, 3, 4]  # nose, eyes, ears
    n_face = count_visible(keypoints, face_indices)
    n_face_low = count_visible(keypoints, face_indices, CONF_LOW)

    if n_face >= 2:
        front_score += 3.0
    elif n_face == 1:
        front_score += 1.2   # reduced: single keypoint is unreliable
    elif n_face_low >= 1:
        front_score += 0.5

    # Back evidence scaled by how few high-confidence face features are present
    if n_face_low == 0:
        back_score += 2.5   # zero face signal at any confidence
    elif n_face == 0:
        back_score += 1.5   # only ghost low-confidence keypoints
    elif n_face == 1:
        back_score += 0.8   # single stray keypoint — likely background noise

    # --- Secondary evidence: nose between shoulders ---
    nose = kp(keypoints, 0)
    l_shoulder = kp(keypoints, 5)
    r_shoulder = kp(keypoints, 6)

    if all(p[2] > 0 for p in [nose, l_shoulder, r_shoulder]):
        s_min = min(l_shoulder[0], r_shoulder[0])
        s_max = max(l_shoulder[0], r_shoulder[0])
        if s_min < nose[0] < s_max:
            front_score += 1.0
        else:
            # Nose outside shoulder width — unusual for front view
            back_score += 0.5

    # --- Tertiary evidence: ear visibility vs shoulder alignment ---
    l_ear = kp(keypoints, 3)
    r_ear = kp(keypoints, 4)
    ears_visible = (l_ear[2] > 0) + (r_ear[2] > 0)

    if ears_visible == 2 and l_shoulder[2] > 0 and r_shoulder[2] > 0:
        # Both ears + both shoulders visible: consistent with front
        front_score += 0.5
    elif ears_visible == 0 and l_shoulder[2] > 0 and r_shoulder[2] > 0:
        # Neither ear visible with both shoulders: back evidence
        back_score += 1.0

    # --- Quaternary: shoulder/hip width ratio ---
    l_hip = kp(keypoints, 11)
    r_hip = kp(keypoints, 12)

    if (l_shoulder[2] > 0 and r_shoulder[2] > 0 and
            l_hip[2] > 0 and r_hip[2] > 0):
        sh_w = abs(l_shoulder[0] - r_shoulder[0])
        hip_w = abs(l_hip[0] - r_hip[0])
        ratio = sh_w / (hip_w + 1e-6)
        # Front: shoulders visibly wider than hips (ratio > 1.1)
        # Back: more similar widths — less reliable, small weight only
        if ratio > 1.15:
            front_score += 0.3
        elif 0.85 < ratio < 1.15:
            back_score += 0.2

    # --- Decision ---
    total = front_score + back_score
    if total < 0.5:
        return None  # insufficient evidence

    margin = abs(front_score - back_score) / total
    if margin < 0.2:
        return None  # too close to call

    return 'front' if front_score > back_score else 'back'


# ---------------------------------------------------------------------------
# Top-level classifier
# ---------------------------------------------------------------------------

def classify_keypoints(keypoints):
    """
    Classify a single set of 17 COCO keypoints (numpy array [17, 3]).
    Returns one of the five class strings.
    """
    extent = classify_body_extent(keypoints)
    if extent is None:
        return 'others'

    orientation = classify_orientation(keypoints)
    if orientation is None:
        return 'others'  # ambiguous — don't guess

    return f"{extent}_{orientation}"


def classify_patch(pose_model, image_path):
    """
    Classify a single image file.
    Convenience wrapper around batched inference for single-image use.
    """
    results = pose_model(image_path, verbose=False)

    if not results or results[0].keypoints is None:
        return 'others'

    kp_data = results[0].keypoints.data
    if kp_data.shape[0] == 0:
        return 'others'

    # Use the largest bounding box — for cropped patches this is the primary
    # subject, not a smaller background person with a visible frontal face.
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [N, 4]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = int(np.argmax(areas))

    keypoints = kp_data[best_idx].cpu().numpy()  # [17, 3]
    return classify_keypoints(keypoints)


# ---------------------------------------------------------------------------
# Batched directory classification
# ---------------------------------------------------------------------------

def classify_directory(
    pose_model,
    input_dir,
    output_dir,
    batch_size=32,
    copy_files=True,
    save_debug_viz=False,
):
    """
    Classify all .jpg/.png files in input_dir using batched inference.

    Batching is the main speedup vs the original single-image loop —
    on a 2080 Ti, batch_size=32 should keep the GPU well-utilised.

    Args:
        pose_model:      YOLOv8 pose model instance
        input_dir:       directory of cropped human patches
        output_dir:      root output directory; per-class subdirs created automatically
        batch_size:      images per inference call
        copy_files:      if True, copy (not move) patches into class subdirs
        save_debug_viz:  if True, save YOLO-annotated images to output_dir/debug_viz/

    Returns:
        results: dict mapping filename -> class string
        summary: dict mapping class string -> count
    """
    image_paths = sorted(
        glob.glob(os.path.join(input_dir, '*.jpg')) +
        glob.glob(os.path.join(input_dir, '*.png'))
    )

    if not image_paths:
        print(f"No images found in {input_dir}")
        return {}, {}

    # Create output class directories
    if copy_files:
        for cls in CLASSES:
            os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    debug_dir = os.path.join(output_dir, 'debug_viz')
    if save_debug_viz:
        os.makedirs(debug_dir, exist_ok=True)

    results  = {}
    summary  = {cls: 0 for cls in CLASSES}

    # Process in batches
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(image_paths), batch_size), total=total_batches, desc="Pose classification", unit="batch"):
        batch_paths = image_paths[batch_start: batch_start + batch_size]

        # Run pose estimation on the whole batch in one call
        batch_results = pose_model(batch_paths, verbose=False)

        for img_path, result in zip(batch_paths, batch_results):
            fname = os.path.basename(img_path)

            if result.keypoints is None or result.keypoints.data.shape[0] == 0:
                cls = 'others'
            else:
                # Pick the largest bounding box — for cropped patches this is the
                # primary subject, not a smaller background person whose face may
                # have higher detection confidence.
                boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4]
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_idx  = int(np.argmax(areas))
                keypoints = result.keypoints.data[best_idx].cpu().numpy()
                cls       = classify_keypoints(keypoints)

            results[fname] = cls
            summary[cls]  += 1

            if copy_files:
                dst = os.path.join(output_dir, cls, fname)
                shutil.copy(img_path, dst)

            if save_debug_viz:
                annotated = result.plot()  # BGR numpy array with boxes + keypoints
                cv2.imwrite(os.path.join(debug_dir, fname), annotated)

    print(f"\nDone. Distribution: { {k: v for k, v in summary.items() if v > 0} }")
    return results, summary