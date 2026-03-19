"""
Question 1.2
classification.py

Pose-based human patch classification into five categories:
  full_body_front, full_body_back, head_shoulder_front, head_shoulder_back, others

Classification logic
--------------------
Orientation (applied first):
  front  — nose AND both eyes visible above FACE_CONF.
  back   — nose below NOSE_BACK_CONF AND at most MAX_EYES_FOR_BACK eyes visible
           above BACK_CONF.  Covers two sub-cases:
             (a) fully turned away — nose and both eyes absent;
             (b) body facing back but head rotated sideways — nose absent,
                 one eye partially visible.
           A true side profile typically produces a nose detection, so it
           still falls through to 'others'.
  others — anything else (side-on, occluded, ambiguous).

Extent (applied only once orientation is decided):
  full_body     — lower-body keypoints (hips/knees/ankles) visible.
                  Optionally gated by bounding-box aspect ratio h/w >= aspect_ratio_min.
                  For back orientation, optionally requires both shoulders visible.
  head_shoulder — at least one shoulder visible, lower-body absent.
  others        — insufficient keypoints.

Ablation flags (set via ClassifierConfig)
-----------------------------------------
  use_aspect_ratio_check          — gate full_body on h/w >= aspect_ratio_min.
  require_both_shoulders_for_back — require both shoulders for full_body_back.

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
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES = [
    'full_body_front',
    'full_body_back',
    'head_shoulder_front',
    'head_shoulder_back',
    'others',
]

COCO_KP_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

# COCO index groups
LOWER_KPS = [11, 12, 13, 14, 15, 16]   # hips, knees, ankles
ANKLE_KPS = [15, 16]
UPPER_KPS = [5, 6]                      # shoulders


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ClassifierConfig:
    """
    All tuneable thresholds and ablation flags in one place.

    Thresholds
    ----------
    face_conf      : min confidence to consider a face keypoint 'present' (front check).
    back_conf      : max confidence below which an eye is considered 'absent' (back check).
    nose_back_conf : max confidence below which the nose is considered 'absent' for the
                     back check.  Defaults to back_conf if None.  Raise independently to
                     catch turned-head-back cases where YOLO fires a weak nose detection
                     (e.g. nose=0.20 on the Godfather frame that was misclassified as
                     'others' in v1.2 — raising this to ~0.25 would catch it).
    body_conf      : min confidence for shoulder / lower-body keypoints.
    max_eyes_for_back : max eyes allowed visible (>= back_conf) while still classifying
                     orientation as 'back'.  1 = allow one visible eye (turned-head case);
                     0 = strict fully-away-only.

    aspect_ratio_min : h/w threshold used when use_aspect_ratio_check is True.

    Ablation flags
    --------------
    use_aspect_ratio_check          : if False, skip the h/w gate on full_body.
    require_both_shoulders_for_back : if False, skip the two-shoulder requirement
                                      for full_body_back.
    """
    # Thresholds
    face_conf:         float = 0.30
    back_conf:         float = 0.10
    nose_back_conf:    float = 0.25
    body_conf:         float = 0.20
    max_eyes_for_back: int   = 1
    aspect_ratio_min:  float = 1.5

    # Ablation flags
    use_aspect_ratio_check:          bool = True
    require_both_shoulders_for_back: bool = True

    def __post_init__(self):
        if self.nose_back_conf is None:
            self.nose_back_conf = self.back_conf


# Default config
DEFAULT_CONFIG = ClassifierConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def n_visible(keypoints: np.ndarray, indices: list[int], threshold: float) -> int:
    return sum(1 for i in indices if keypoints[i, 2] >= threshold)


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def classify_orientation(keypoints: np.ndarray, cfg: ClassifierConfig) -> str | None:
    """
    Returns 'front', 'back', or None (side-on / ambiguous → 'others').

    front: nose AND both eyes >= face_conf.
    back:  nose < nose_back_conf AND eyes visible <= max_eyes_for_back.
    None:  everything else.
    """
    nose_conf      = keypoints[0, 2]
    left_eye_conf  = keypoints[1, 2]
    right_eye_conf = keypoints[2, 2]

    all_face_present = (
        nose_conf      >= cfg.face_conf and
        left_eye_conf  >= cfg.face_conf and
        right_eye_conf >= cfg.face_conf
    )

    nose_absent    = nose_conf < cfg.nose_back_conf
    n_eyes_visible = sum(1 for c in (left_eye_conf, right_eye_conf) if c >= cfg.back_conf)
    is_back        = nose_absent and n_eyes_visible <= cfg.max_eyes_for_back

    if all_face_present:
        return 'front'
    if is_back:
        return 'back'
    return None


# ---------------------------------------------------------------------------
# Body extent
# ---------------------------------------------------------------------------

def classify_extent(
    keypoints: np.ndarray,
    bbox: np.ndarray | None,
    cfg: ClassifierConfig,
) -> str | None:
    """
    Returns 'full_body', 'head_shoulder', or None (insufficient keypoints).
    """
    n_lower  = n_visible(keypoints, LOWER_KPS, cfg.body_conf)
    n_ankles = n_visible(keypoints, ANKLE_KPS, cfg.body_conf)

    has_lower_body = n_ankles >= 1 or n_lower >= 3

    if has_lower_body:
        if cfg.use_aspect_ratio_check and bbox is not None:
            x1, y1, x2, y2 = bbox
            h = y2 - y1
            w = (x2 - x1) + 1e-6
            if h / w < cfg.aspect_ratio_min:
                return 'head_shoulder'
        return 'full_body'

    if n_visible(keypoints, UPPER_KPS, cfg.body_conf) >= 1:
        return 'head_shoulder'

    return None


# ---------------------------------------------------------------------------
# Top-level per-patch classifier
# ---------------------------------------------------------------------------

def classify_keypoints(
    keypoints: np.ndarray,
    bbox: np.ndarray | None = None,
    cfg: ClassifierConfig = DEFAULT_CONFIG,
) -> str:
    """
    Classify a single set of 17 COCO keypoints (numpy array [17, 3]).
    Returns one of the five CLASSES strings.
    """
    orientation = classify_orientation(keypoints, cfg)
    if orientation is None:
        return 'others'

    extent = classify_extent(keypoints, bbox, cfg)
    if extent is None:
        return 'others'

    if (
        orientation == 'back'
        and extent == 'full_body'
        and cfg.require_both_shoulders_for_back
        and n_visible(keypoints, UPPER_KPS, cfg.body_conf) < 2
    ):
        extent = 'head_shoulder'

    return f"{extent}_{orientation}"


# ---------------------------------------------------------------------------
# Deep diagnostic rendering  (used by classify_directory and annotate.py)
# ---------------------------------------------------------------------------

# CV2 drawing constants
_BG        = (18,  18,  22)    # dark background
_WHITE     = (220, 220, 228)
_MUTED     = (100, 100, 110)
_GREEN     = (80,  220, 80)    # PASS / above face_conf
_GOLD      = (40,  200, 220)   # above back_conf
_RED_BGR   = (70,  70,  240)   # FAIL / below back_conf
_ORANGE    = (40,  160, 230)   # 'others' verdict
_BLUE      = (220, 140, 60)    # secondary detections
_ACCENT    = (70,  255, 230)   # accent (yellow-green)

# COCO skeleton edges for drawing limbs
_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),          # face
    (5,6),(5,7),(7,9),(6,8),(8,10),   # arms
    (5,11),(6,12),(11,12),            # torso
    (11,13),(13,15),(12,14),(14,16),  # legs
]

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_MONO = cv2.FONT_HERSHEY_PLAIN


def _put(img, text, xy, scale=0.38, color=_WHITE, thickness=1):
    cv2.putText(img, text, xy, _FONT, scale, color, thickness, cv2.LINE_AA)


def _panel(h, w):
    """Blank dark panel."""
    return np.full((h, w, 3), _BG, dtype=np.uint8)


def _draw_detections(img_bgr, boxes, kps_all, primary, cfg):
    """
    Panel 1: original patch with all bboxes + skeleton of primary detection.
    Returns a copy — does not modify the input.
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # All bounding boxes
    for i, box in enumerate(boxes):
        bx1, by1, bx2, by2 = [int(v) for v in box]
        color = (50, 50, 220) if i == primary else _BLUE  # red-ish for primary
        thick = 2 if i == primary else 1
        cv2.rectangle(out, (bx1, by1), (bx2, by2), color, thick)
        area = (bx2 - bx1) * (by2 - by1)
        _put(out, f"#{i} {int(area)}", (bx1, max(by1 - 4, 10)), 0.32,
             color, 1)

    # Skeleton + keypoints for primary detection only
    kps = kps_all[primary]
    for a, b in _SKELETON:
        xa, ya, ca = kps[a]
        xb, yb, cb = kps[b]
        if ca >= cfg.back_conf and cb >= cfg.back_conf:
            cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), _MUTED, 1,
                     cv2.LINE_AA)
    for ki, (kx, ky, kc) in enumerate(kps):
        if kc >= cfg.back_conf:
            color = _GREEN if kc >= cfg.face_conf else _GOLD
            cv2.circle(out, (int(kx), int(ky)), 3, color, -1, cv2.LINE_AA)
            _put(out, COCO_KP_NAMES[ki], (int(kx) + 3, int(ky) + 3), 0.28,
                 color, 1)

    # Legend
    _put(out, f"{len(boxes)} det  primary=#0", (4, h - 20), 0.30, _MUTED)
    return out


def _draw_crop(crop_bgr, orientation, extent, final_cls):
    """Panel 2: primary bbox crop with verdict text."""
    if crop_bgr is None or crop_bgr.size == 0:
        out = _panel(200, 160)
        _put(out, "no crop", (10, 100), 0.40, _MUTED)
        return out
    out     = crop_bgr.copy()
    h, w    = out.shape[:2]
    color   = _GREEN if final_cls not in ('others', 'bad_extraction') else _ORANGE
    _put(out, f"ori: {orientation or 'none'}", (4, 14), 0.34, color)
    _put(out, f"ext: {extent   or 'none'}", (4, 26), 0.34, color)
    return out


def _draw_bar_chart(kp_confs, cfg, panel_h, panel_w):
    """
    Panel 3: horizontal confidence bars for all 17 keypoints, drawn with CV2.
    """
    out     = _panel(panel_h, panel_w)
    n       = 17
    row_h   = (panel_h - 30) // n
    bar_max = panel_w - 90   # pixels for a conf=1.0 bar
    x0      = 72             # left edge of bars

    # Column header
    _put(out, "kp confidence", (x0, 12), 0.34, _MUTED)

    for i, (name, conf) in enumerate(zip(COCO_KP_NAMES, kp_confs)):
        y_center = 24 + i * row_h + row_h // 2
        y_top    = y_center - row_h // 2 + 2
        y_bot    = y_center + row_h // 2 - 2

        # Keypoint name
        _put(out, name, (2, y_center + 4), 0.30, _MUTED)

        # Bar colour by threshold
        if conf < cfg.back_conf:
            bar_col = _RED_BGR
        elif conf < cfg.face_conf:
            bar_col = _GOLD
        else:
            bar_col = _GREEN

        bar_w = max(1, int(conf * bar_max))
        cv2.rectangle(out, (x0, y_top), (x0 + bar_w, y_bot), bar_col, -1)

        # Conf value text
        _put(out, f"{conf:.2f}", (x0 + bar_w + 3, y_center + 4), 0.28, bar_col)

    # Threshold lines
    for thresh, color, label in [
        (cfg.back_conf,  _GOLD,        f"bk={cfg.back_conf:.2f}"),
        (cfg.body_conf,  _ACCENT,      f"bd={cfg.body_conf:.2f}"),
        (cfg.face_conf,  _GREEN,       f"fc={cfg.face_conf:.2f}"),
    ]:
        lx = x0 + int(thresh * bar_max)
        cv2.line(out, (lx, 18), (lx, panel_h - 4), color, 1, cv2.LINE_AA)
        _put(out, label, (lx + 2, panel_h - 4), 0.27, color)

    return out


def _draw_trace(trace_rows, orientation, extent, final_cls, panel_h, panel_w):
    """
    Panel 4: rule-by-rule decision trace table drawn with CV2.
    """
    out   = _panel(panel_h, panel_w)
    color = _GREEN if final_cls not in ('others', 'bad_extraction') else _ORANGE

    # Title
    _put(out, f"Decision trace", (6, 14), 0.38, _WHITE)
    _put(out, f"ori={orientation or 'none'}  ext={extent or 'none'}", (6, 26), 0.32, _MUTED)
    _put(out, f"=> {final_cls}", (6, 40), 0.40, color, 1)

    # Table header
    cols   = [6, 120, 260, 390]
    header = ["rule", "measured", "threshold", "result"]
    y_hdr  = 58
    for cx, h_text in zip(cols, header):
        _put(out, h_text, (cx, y_hdr), 0.32, _MUTED)
    cv2.line(out, (4, y_hdr + 4), (panel_w - 4, y_hdr + 4), _MUTED, 1)

    row_h = (panel_h - y_hdr - 16) // max(len(trace_rows), 1)
    for ri, (rule, measured, threshold, result) in enumerate(trace_rows):
        y = y_hdr + 16 + ri * row_h
        res_color = _GREEN if result == "PASS" else _RED_BGR
        # Highlight row bg for FAIL
        if result != "PASS":
            cv2.rectangle(out, (4, y - row_h + 4), (panel_w - 4, y + 4),
                          (30, 20, 50), -1)
        _put(out, rule,      (cols[0], y), 0.32, _WHITE)
        _put(out, measured,  (cols[1], y), 0.30, _MUTED)
        _put(out, threshold, (cols[2], y), 0.30, _MUTED)
        _put(out, result,    (cols[3], y), 0.34, res_color, 1)

    return out


def render_diagnostic(
    result,
    img_bgr: np.ndarray,
    cfg: ClassifierConfig = DEFAULT_CONFIG,
    predicted_class: str | None = None,
) -> np.ndarray:
    """
    Render the deep classification diagnostic as a BGR image using CV2 only.
    ~10-20x faster than the matplotlib version with no external dependencies.

    4-panel layout:
      1. Original patch — all bboxes + skeleton of primary detection
      2. Primary bbox crop
      3. Keypoint confidence bars
      4. Rule-by-rule decision trace

    Args:
        result:           Single YOLO pose result object (already inferred).
        img_bgr:          Original BGR image the result was run on.
        cfg:              ClassifierConfig.
        predicted_class:  Pre-computed class label; re-derived if None.

    Returns:
        BGR numpy array (H, W, 3).
    """
    h_img, w_img = img_bgr.shape[:2]
    OUT_H = max(h_img, 340)   # panel height — at least 340px for readability

    # --- No detections ---
    if result.keypoints is None or result.keypoints.data.shape[0] == 0:
        out = _panel(OUT_H, w_img + 600)
        # Show the original patch on the left
        ph = min(h_img, OUT_H)
        out[:ph, :w_img] = img_bgr[:ph]
        _put(out, "No pose detections found", (w_img + 10, OUT_H // 2 - 10),
             0.50, _ORANGE, 1)
        _put(out, "=> classified as 'others'", (w_img + 10, OUT_H // 2 + 14),
             0.42, _MUTED)
        return out

    boxes   = result.boxes.xyxy.cpu().numpy()
    kp_data = result.keypoints.data.cpu().numpy()
    areas   = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    primary = int(np.argmax(areas))
    kps     = kp_data[primary]
    bbox    = boxes[primary]

    # Primary crop (BGR)
    x1c = max(0, int(bbox[0])); y1c = max(0, int(bbox[1]))
    x2c = min(w_img, int(bbox[2])); y2c = min(h_img, int(bbox[3]))
    crop_bgr = img_bgr[y1c:y2c, x1c:x2c]

    # Classification
    orientation = classify_orientation(kps, cfg)
    extent      = classify_extent(kps, bbox, cfg)
    final_cls   = predicted_class if predicted_class is not None \
                  else classify_keypoints(kps, bbox, cfg)

    # Decision trace values
    nose, l_eye, r_eye = kps[0, 2], kps[1, 2], kps[2, 2]
    n_lower  = n_visible(kps, LOWER_KPS, cfg.body_conf)
    n_ankles = n_visible(kps, ANKLE_KPS, cfg.body_conf)
    n_shldrs = n_visible(kps, UPPER_KPS, cfg.body_conf)
    h_box    = bbox[3] - bbox[1]
    w_box    = (bbox[2] - bbox[0]) + 1e-6
    aspect   = h_box / w_box

    trace_rows = [
        ("Front",       f"nose={nose:.2f} eL={l_eye:.2f} eR={r_eye:.2f}",
         f"all>={cfg.face_conf:.2f}",
         "PASS" if (nose>=cfg.face_conf and l_eye>=cfg.face_conf and r_eye>=cfg.face_conf) else "FAIL"),
        ("Back",        f"nose={nose:.2f} eyes_vis={sum(1 for c in (l_eye,r_eye) if c>=cfg.back_conf)}",
         f"nose<{cfg.nose_back_conf:.2f} eyes<={cfg.max_eyes_for_back}",
         "PASS" if (nose<cfg.nose_back_conf and sum(1 for c in (l_eye,r_eye) if c>=cfg.back_conf)<=cfg.max_eyes_for_back) else "FAIL"),
        ("Lower-body",  f"lower={n_lower} ankles={n_ankles}",
         "ankles>=1 OR lower>=3",
         "PASS" if (n_ankles>=1 or n_lower>=3) else "FAIL"),
        ("Aspect",      f"h/w={aspect:.2f}",
         f">={cfg.aspect_ratio_min:.1f}",
         "PASS" if aspect>=cfg.aspect_ratio_min else "FAIL"),
        ("Shoulder",    f"visible={n_shldrs}",
         f">=1@{cfg.body_conf:.2f}",
         "PASS" if n_shldrs>=1 else "FAIL"),
    ]

    # --- Build panels ---
    # P1: detection overlay — same size as input patch
    p1 = _draw_detections(img_bgr, boxes, kp_data, primary, cfg)
    p1 = cv2.resize(p1, (int(w_img * OUT_H / h_img), OUT_H))

    # P2: primary crop — fixed width proportional to crop aspect ratio, capped
    if crop_bgr.size > 0:
        cw = int(crop_bgr.shape[1] * OUT_H / max(crop_bgr.shape[0], 1))
        cw = max(80, min(cw, 200))
        p2 = _draw_crop(cv2.resize(crop_bgr, (cw, OUT_H)), orientation, extent, final_cls)
    else:
        p2 = _draw_crop(None, orientation, extent, final_cls)
        p2 = cv2.resize(p2, (160, OUT_H))

    # P3: confidence bars — fixed width
    p3 = _draw_bar_chart(kps[:, 2], cfg, OUT_H, 200)

    # P4: trace table — fixed width
    p4 = _draw_trace(trace_rows, orientation, extent, final_cls, OUT_H, 420)

    # 2px dark separator between panels
    sep = _panel(OUT_H, 2)
    out = np.concatenate([p1, sep, p2, sep, p3, sep, p4], axis=1)
    return out


# ---------------------------------------------------------------------------
# Batched directory classification
# ---------------------------------------------------------------------------

def classify_directory(
    pose_model,
    input_dir: str,
    output_dir: str,
    cfg: ClassifierConfig = DEFAULT_CONFIG,
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
        cfg:            ClassifierConfig controlling thresholds and ablation flags.
        batch_size:     Images per inference call.
        copy_files:     If True, copy patches into per-class subdirs.
        save_debug_viz: If True, save deep diagnostic images to output_dir/.diag_cache/.

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
        os.makedirs(os.path.join(output_dir, '.diag_cache'), exist_ok=True)

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
                    cfg=cfg,
                )

            results[fname]  = cls
            summary[cls]   += 1

            if copy_files:
                shutil.copy(img_path, os.path.join(output_dir, cls, fname))

            if save_debug_viz:
                img_bgr = cv2.imread(img_path)
                if img_bgr is not None:
                    diag = render_diagnostic(result, img_bgr, cfg, predicted_class=cls)
                    cv2.imwrite(os.path.join(output_dir, '.diag_cache', fname), diag)

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