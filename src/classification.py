"""
1.2
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

import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)
_FACE_MODEL_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "blaze_face_short_range.tflite",
)


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
# Adaptive confidence threshold
# ---------------------------------------------------------------------------

def adaptive_conf_high(keypoints):
    """
    Compute a per-patch confidence threshold from the median keypoint confidence.

    Film footage tends to have systemically lower pose confidence than game
    footage; a hard global threshold discards too much useful signal.  We use
    median - 0.1, clipped to [0.15, 0.30] so we never accept near-random
    detections or set the bar higher than the default CONF_HIGH.
    """
    median_conf = float(np.median(keypoints[:, 2]))
    return float(np.clip(median_conf - 0.1, 0.15, 0.30))


# ---------------------------------------------------------------------------
# MediaPipe face detection helper
# ---------------------------------------------------------------------------

def build_face_detector(min_confidence=0.5, model_path=None):
    """
    Return a MediaPipe FaceDetector (Tasks API, mediapipe >= 0.10).

    The blaze_face_short_range.tflite model is auto-downloaded to models/
    if not already present.  Returns None on any failure so the pipeline
    degrades gracefully.

    Args:
        min_confidence: minimum detection confidence threshold.
        model_path:     explicit path to the .tflite model; defaults to
                        models/blaze_face_short_range.tflite.
    """
    model_path = model_path or _FACE_MODEL_DEFAULT

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"[classification] Downloading face detector model -> {model_path}")
        try:
            urllib.request.urlretrieve(_FACE_MODEL_URL, model_path)
        except Exception as e:
            print(f"[classification] Download failed: {e}. Face detection disabled.")
            return None

    try:
        BaseOptions         = mp.tasks.BaseOptions
        FaceDetector        = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode   = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        return FaceDetector.create_from_options(options)
    except Exception as e:
        print(f"[classification] Could not build face detector: {e}")
        return None


def _run_face_detection(face_detector, image_path, bbox=None):
    """
    Return True if the MediaPipe face detector fires on the given image.

    Args:
        face_detector: MediaPipe FaceDetector (Tasks API) instance, or None.
        image_path:    Path to the patch image.
        bbox:          Optional (x1, y1, x2, y2) in pixel coords of the primary
                       YOLO detection.  When provided, MediaPipe only sees that
                       crop — background figures outside the bbox are ignored,
                       preventing their faces from biasing the front/back decision.

    Silently returns False if image loading fails or face_detector is None.
    """
    if face_detector is None:
        return False
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return False

    if bbox is not None:
        h, w = img_bgr.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))
        if x2 > x1 and y2 > y1:
            img_bgr = img_bgr[y1:y2, x1:x2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result   = face_detector.detect(mp_image)
        return bool(result.detections)
    except Exception:
        return False


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

def classify_body_extent(keypoints, conf_high=None, bbox=None):
    """
    Returns 'full_body', 'head_shoulder', or None (insufficient keypoints).

    Full body requires evidence of lower limbs.
    Head-shoulder requires at least one shoulder visible.

    Args:
        conf_high: adaptive confidence threshold; falls back to CONF_HIGH if None.
        bbox:      (x1, y1, x2, y2) from the pose model; used to veto full_body
                   when aspect ratio (h/w) < 1.5 — a patch that is not
                   substantially taller than wide cannot contain a standing
                   full body.
    """
    ch = conf_high if conf_high is not None else CONF_HIGH

    # Lower body evidence
    lower_indices = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
    ankle_indices = [15, 16]

    n_lower     = count_visible(keypoints, lower_indices, ch)
    n_ankles    = count_visible(keypoints, ankle_indices, ch)
    n_lower_low = count_visible(keypoints, lower_indices, CONF_LOW)

    # Strong case: ankles visible
    if n_ankles >= 1:
        extent = 'full_body'

    # Moderate case: hips + at least one knee
    elif count_visible(keypoints, [11, 12, 13, 14], ch) >= 3:
        extent = 'full_body'

    # Weak case: two or more lower body points at low confidence
    elif n_lower >= 2 or n_lower_low >= 3:
        extent = 'full_body'

    else:
        # Upper body check
        upper_indices = [0, 1, 2, 3, 4, 5, 6]  # face + shoulders
        n_upper     = count_visible(keypoints, upper_indices, ch)
        n_shoulders = count_visible(keypoints, [5, 6], ch)

        if n_shoulders >= 1 or n_upper >= 2:
            extent = 'head_shoulder'
        else:
            return None  # not enough to classify

    # Aspect-ratio veto: a bounding box that is not clearly taller than wide
    # (h/w < 1.5) is implausible for a standing full-body detection.
    if extent == 'full_body' and bbox is not None:
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = (x2 - x1) + 1e-6
        if h / w < 1.5:
            extent = 'head_shoulder'

    return extent


# ---------------------------------------------------------------------------
# Front / back classification
# ---------------------------------------------------------------------------

def classify_orientation(keypoints, conf_high=None, face_detected=False):
    """
    Returns 'front', 'back', or None (ambiguous).

    Uses a weighted evidence accumulation approach:
      - MediaPipe face detection (independent of YOLO) is the strongest signal.
      - YOLO face keypoints (nose, eyes, ears) add corroborating front evidence.
      - Structural geometry (ear/shoulder positioning) provides secondary signal.
      - Avoids defaulting to front when evidence is genuinely ambiguous.

    Args:
        conf_high:     adaptive confidence threshold (falls back to CONF_HIGH).
        face_detected: True if a dedicated face detector fired on this patch.
    """
    ch = conf_high if conf_high is not None else CONF_HIGH

    front_score = 0.0
    back_score  = 0.0

    # --- Compute face keypoint visibility upfront (needed for both primary and secondary evidence) ---
    face_indices    = [0, 1, 2]  # nose, l_eye, r_eye only
    ear_indices     = [3, 4]     # treated separately below
    n_face          = count_visible(keypoints, face_indices, ch)
    n_face_low      = count_visible(keypoints, face_indices, CONF_LOW)
    n_ears          = count_visible(keypoints, ear_indices, ch)

    # Total keypoints detected at any confidence — used to gate back evidence.
    n_kp_any = count_visible(keypoints, list(range(17)), CONF_LOW)

    # --- Primary evidence: dedicated face detector ---
    # Full bonus only when YOLO also sees face keypoints on the primary subject.
    # If the detector fired but YOLO sees nothing, a background face within the
    # bbox is the likely cause.
    if face_detected and n_face >= 1:
        front_score += 3.0
    elif face_detected:
        front_score += 1.0

    # --- Secondary evidence: YOLO face keypoints ---
    # Only nose and eyes are reliable front-only indicators.
    # Ears are visible from both front and back, so excluded here.

    if n_face >= 2:
        front_score += 3.0
    elif n_face == 1:
        front_score += 1.2   # reduced: single keypoint is unreliable
    elif n_face_low >= 1:
        front_score += 0.5

    # Back evidence: only penalise absent face when there are enough keypoints
    # overall — if the model barely detected anyone, missing face KPs say
    # nothing about orientation.
    if n_face_low == 0 and n_kp_any >= 6:
        back_score += 2.5
    elif n_face == 0 and n_face_low >= 1:
        back_score += 1.5   # ghost low-confidence keypoints only
    elif n_face == 1:
        back_score += 0.8   # single stray keypoint — likely background noise

    # --- Tertiary evidence: nose between shoulders ---
    nose       = kp(keypoints, 0, ch)
    l_shoulder = kp(keypoints, 5, ch)
    r_shoulder = kp(keypoints, 6, ch)

    if all(p[2] > 0 for p in [nose, l_shoulder, r_shoulder]):
        s_min = min(l_shoulder[0], r_shoulder[0])
        s_max = max(l_shoulder[0], r_shoulder[0])
        if s_min < nose[0] < s_max:
            front_score += 1.0
        else:
            # Nose outside shoulder width — unusual for front view
            back_score += 0.5

    # --- Quaternary evidence: ear visibility ---
    # Ears visible without any nose/eyes: back-of-head presentation.
    # Ears visible alongside nose/eyes: corroborates front view.
    l_ear = kp(keypoints, 3, ch)
    r_ear = kp(keypoints, 4, ch)
    ears_visible = (l_ear[2] > 0) + (r_ear[2] > 0)

    if ears_visible >= 1 and n_face == 0:
        # Ears with no frontal face features → back
        back_score += 1.2 * ears_visible
    elif ears_visible >= 1 and n_face >= 2:
        # Ears alongside nose/eyes → corroborates front
        front_score += 0.3 * ears_visible
    # ears_visible >= 1 and n_face == 1: no score either way — ambiguous

    # --- Quinary: shoulder/hip width ratio ---
    l_hip = kp(keypoints, 11, ch)
    r_hip = kp(keypoints, 12, ch)

    if (l_shoulder[2] > 0 and r_shoulder[2] > 0 and
            l_hip[2] > 0 and r_hip[2] > 0):
        sh_w  = abs(l_shoulder[0] - r_shoulder[0])
        hip_w = abs(l_hip[0]      - r_hip[0])
        ratio = sh_w / (hip_w + 1e-6)
        if ratio > 1.15:
            front_score += 0.3
        elif 0.85 < ratio < 1.15:
            back_score += 0.2

    # --- Decision ---
    total = front_score + back_score
    if total < 0.5:
        return None  # insufficient evidence

    # Absolute score floor: strong positive front evidence overrides a narrow
    # margin so patches with e.g. face_detected=True don't end up in 'others'.
    if front_score > 2.5 and front_score > back_score:
        return 'front'

    margin = abs(front_score - back_score) / total
    if margin < 0.12:          # was 0.20 — fewer genuinely-decided patches wasted
        return None

    return 'front' if front_score > back_score else 'back'


def classify_orientation_debug(keypoints, conf_high=None, face_detected=False):
    """
    Identical logic to classify_orientation but returns a full score trace dict
    alongside the decision.  Use for pipeline diagnostics only.

    Returns
    -------
    orientation : str | None
    trace : dict with keys: conf_high, face_detected, n_face, n_face_low,
            n_kp_any, front_score, back_score, margin, override_fired,
            decision, and a list of per-step dicts under 'steps'.
    """
    ch = conf_high if conf_high is not None else CONF_HIGH
    trace = {"conf_high": ch, "face_detected": face_detected, "steps": []}

    front_score = 0.0
    back_score  = 0.0

    def record(label, delta_front=0.0, delta_back=0.0, note=""):
        nonlocal front_score, back_score
        front_score += delta_front
        back_score  += delta_back
        trace["steps"].append({
            "label": label,
            "delta_front": delta_front,
            "delta_back": delta_back,
            "front_score": round(front_score, 3),
            "back_score": round(back_score, 3),
            "note": note,
        })

    if face_detected:
        record("face_detector", delta_front=3.0, note="MediaPipe fired")
    else:
        record("face_detector", note="MediaPipe not used / no face")

    # --- Secondary evidence: YOLO face keypoints ---
    # Only nose and eyes are reliable front-only indicators.
    # Ears are visible from both front and back, so excluded here.
    face_indices = [0, 1, 2]  # nose, l_eye, r_eye only
    ear_indices  = [3, 4]
    n_face     = count_visible(keypoints, face_indices, ch)
    n_face_low = count_visible(keypoints, face_indices, CONF_LOW)
    n_ears     = count_visible(keypoints, ear_indices, ch)
    n_kp_any   = count_visible(keypoints, list(range(17)), CONF_LOW)
    trace.update({"n_face": n_face, "n_face_low": n_face_low, "n_kp_any": n_kp_any, "n_ears": n_ears})

    if n_face >= 2:
        record("yolo_face_kps", delta_front=3.0, note=f"n_face={n_face} >= 2")
    elif n_face == 1:
        record("yolo_face_kps", delta_front=0.5, note="n_face=1 (unreliable)")
    elif n_face_low >= 1:
        record("yolo_face_kps", delta_front=0.5, note=f"n_face_low={n_face_low} (low conf)")
    else:
        record("yolo_face_kps", note="no face kps")

    if n_face_low == 0 and n_kp_any >= 6:
        record("back_no_face", delta_back=1.0, note=f"no face at any conf, n_kp_any={n_kp_any}")
    elif n_face == 0 and n_face_low >= 1:
        record("back_ghost_face", delta_back=1.5, note="ghost low-conf face kps")
    elif n_face == 1:
        record("back_stray_kp", delta_back=0.8, note="single stray face kp")
    else:
        record("back_face", note="no back evidence from face kps")

    nose       = kp(keypoints, 0, ch)
    l_shoulder = kp(keypoints, 5, ch)
    r_shoulder = kp(keypoints, 6, ch)

    if all(p[2] > 0 for p in [nose, l_shoulder, r_shoulder]):
        s_min = min(l_shoulder[0], r_shoulder[0])
        s_max = max(l_shoulder[0], r_shoulder[0])
        if s_min < nose[0] < s_max:
            record("nose_in_shoulders", delta_front=1.0, note="nose between shoulders")
        else:
            record("nose_outside_shoulders", delta_back=0.5, note="nose outside shoulder width")
    else:
        record("nose_shoulders", note="nose or shoulders not visible")

    l_ear = kp(keypoints, 3, ch)
    r_ear = kp(keypoints, 4, ch)
    ears_visible = (l_ear[2] > 0) + (r_ear[2] > 0)

    if ears_visible >= 1 and n_face == 0:
        record("ears", delta_back=1.2 * ears_visible,
               note=f"ears={ears_visible} without nose/eyes -> back")
    elif ears_visible >= 1 and n_face >= 1:
        record("ears", delta_front=0.3 * ears_visible,
               note=f"ears={ears_visible} with nose/eyes -> front")
    else:
        record("ears", note=f"ears_visible={ears_visible}, no strong signal")

    l_hip = kp(keypoints, 11, ch)
    r_hip = kp(keypoints, 12, ch)

    if all(p[2] > 0 for p in [l_shoulder, r_shoulder, l_hip, r_hip]):
        sh_w  = abs(l_shoulder[0] - r_shoulder[0])
        hip_w = abs(l_hip[0]      - r_hip[0])
        ratio = sh_w / (hip_w + 1e-6)
        if ratio > 1.15:
            record("sh_hip_ratio", delta_front=0.3, note=f"ratio={ratio:.2f} > 1.15")
        elif 0.85 < ratio < 1.15:
            record("sh_hip_ratio", delta_back=0.2, note=f"ratio={ratio:.2f} near 1.0")
        else:
            record("sh_hip_ratio", note=f"ratio={ratio:.2f}")
    else:
        record("sh_hip_ratio", note="hips/shoulders not both visible")

    total    = front_score + back_score
    margin   = abs(front_score - back_score) / total if total > 0.5 else 0.0
    override = front_score > 2.5 and front_score > back_score

    trace.update({
        "front_score": round(front_score, 3),
        "back_score":  round(back_score, 3),
        "total":       round(total, 3),
        "margin":      round(margin, 3),
        "override_fired": override,
    })

    if total < 0.5:
        orientation = None
    elif override:
        orientation = 'front'
    elif margin < 0.12:
        orientation = None
    else:
        orientation = 'front' if front_score > back_score else 'back'

    trace["decision"] = orientation
    return orientation, trace


# ---------------------------------------------------------------------------
# Top-level classifier
# ---------------------------------------------------------------------------

def classify_keypoints(keypoints, face_detected=False, bbox=None):
    """
    Classify a single set of 17 COCO keypoints (numpy array [17, 3]).
    Returns one of the five class strings.

    Args:
        face_detected: True if a dedicated face detector fired on this patch.
        bbox:          (x1, y1, x2, y2) bounding box from the pose model;
                       used for the aspect-ratio extent veto.
    """
    ch = adaptive_conf_high(keypoints)

    extent = classify_body_extent(keypoints, conf_high=ch, bbox=bbox)
    if extent is None:
        return 'others'

    orientation = classify_orientation(keypoints, conf_high=ch, face_detected=face_detected)
    if orientation is None:
        return 'others'  # ambiguous — don't guess

    return f"{extent}_{orientation}"


def classify_patch(pose_model, image_path, face_detector=None):
    """
    Classify a single image file.
    Convenience wrapper around batched inference for single-image use.

    Args:
        pose_model:     YOLOv8 pose model instance.
        image_path:     Path to the image file.
        face_detector:  Optional MediaPipe FaceDetection instance (from
                        build_face_detector()).  When provided, used as a
                        parallel front-orientation signal that is independent
                        of YOLO pose keypoint confidence.
    """
    results = pose_model(image_path, verbose=False)

    if not results or results[0].keypoints is None:
        return 'others'

    kp_data = results[0].keypoints.data
    if kp_data.shape[0] == 0:
        return 'others'

    # Use the largest bounding box — for cropped patches this is the primary
    # subject, not a smaller background person with a visible frontal face.
    boxes    = results[0].boxes.xyxy.cpu().numpy()  # [N, 4]
    areas    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = int(np.argmax(areas))

    keypoints     = kp_data[best_idx].cpu().numpy()  # [17, 3]
    bbox          = boxes[best_idx]                   # (x1, y1, x2, y2)
    face_detected = _run_face_detection(face_detector, image_path, bbox=bbox)

    return classify_keypoints(keypoints, face_detected=face_detected, bbox=bbox)


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
    face_detector=None,
):
    """
    Classify all .jpg/.png files in input_dir using batched inference.

    Batching is the main speedup vs the original single-image loop —
    on a 2080 Ti, batch_size=32 should keep the GPU well-utilised.
    MediaPipe face detection (if face_detector is provided) is run per-image
    outside the YOLO batch; it is fast enough (~3 ms/image) not to be a
    bottleneck.

    Args:
        pose_model:      YOLOv8 pose model instance
        input_dir:       directory of cropped human patches
        output_dir:      root output directory; per-class subdirs created automatically
        batch_size:      images per inference call
        copy_files:      if True, copy (not move) patches into class subdirs
        save_debug_viz:  if True, save YOLO-annotated images to output_dir/debug_viz/
        face_detector:   optional MediaPipe FaceDetection instance (from
                         build_face_detector()); improves front/back accuracy

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
                boxes    = result.boxes.xyxy.cpu().numpy()  # [N, 4]
                areas    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_idx = int(np.argmax(areas))
                keypoints         = result.keypoints.data[best_idx].cpu().numpy()
                bbox              = boxes[best_idx]
                face_det_result   = _run_face_detection(face_detector, img_path, bbox=bbox)
                cls               = classify_keypoints(
                    keypoints,
                    face_detected=face_det_result,
                    bbox=bbox,
                )

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


def reload_classification_results(cls_save_path):
    """
    Reconstruct ``results`` and ``summary`` from an existing classification
    directory (the per-class subdirectories written by ``classify_directory``).

    Parameters
    ----------
    cls_save_path : str
        Root classification output directory containing per-class subdirs.

    Returns
    -------
    results : dict[str, str]
        Mapping of filename -> class string.
    summary : dict[str, int]
        Mapping of class string -> count.
    """
    results = {}
    summary = {cls: 0 for cls in CLASSES}

    for cls in CLASSES:
        cls_dir = os.path.join(cls_save_path, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.png')):
                results[fname] = cls
                summary[cls] += 1

    total = sum(summary.values())
    print(f"Reloaded classification results from {cls_save_path}")
    print(f"  Total: {total}  |  " + "  ".join(f"{k}: {v}" for k, v in summary.items() if v > 0))
    return results, summary