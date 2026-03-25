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

# Colour palette  (BGR)
_BG        = (246, 244, 244)   # light warm white  — panel background
_SURFACE   = (255, 255, 255)   # pure white        — crop panel
_SEP       = (210, 208, 208)   # mid-grey          — panel dividers
_TEXT      = (40,  38,  38)    # near-black        — primary text
_MUTED     = (140, 136, 136)   # mid-grey          — secondary text
_GREEN     = (60,  170,  60)   # limegreen-ish
_GOLD      = (30,  165, 200)   # gold/amber
_RED       = (70,   70, 220)   # tomato-ish
_PASS_BG   = (210, 240, 210)   # light green tint  — PASS cell bg
_FAIL_BG   = (210, 215, 248)   # light red tint    — FAIL cell bg
_ACCENT    = (160,  80,  30)   # deep blue-teal    — verdict colour
_OTHERS    = (40,  120, 220)   # orange            — 'others' verdict

# COCO skeleton edges for drawing limbs (also used by gcn.py for graph construction)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
_F   = cv2.FONT_HERSHEY_DUPLEX
_FT  = cv2.FONT_HERSHEY_PLAIN   # small mono for table cells


def _put(img, text, xy, scale=0.45, color=_TEXT, thickness=1, font=_F):
    cv2.putText(img, str(text), xy, font, scale, color, thickness, cv2.LINE_AA)


def _text_w(text, scale=0.45, thickness=1, font=_F):
    (w, _), _ = cv2.getTextSize(str(text), font, scale, thickness)
    return w


def _draw_detections(img_bgr, boxes, kps_all, primary, cfg):
    """
    Panel 1: original patch with all bboxes + skeleton of primary detection.
    Returns a copy — does not modify the input.
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]

    for i, box in enumerate(boxes):
        bx1, by1, bx2, by2 = [int(v) for v in box]
        color = (50, 50, 210) if i == primary else (200, 120, 60)
        thick = 2 if i == primary else 1
        cv2.rectangle(out, (bx1, by1), (bx2, by2), color, thick)

    kps = kps_all[primary]
    for a, b in COCO_SKELETON:
        xa, ya, ca = kps[a]
        xb, yb, cb = kps[b]
        if ca >= cfg.back_conf and cb >= cfg.back_conf:
            cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)),
                     (180, 180, 180), 1, cv2.LINE_AA)
    for kx, ky, kc in kps:
        if kc >= cfg.back_conf:
            col = _GREEN if kc >= cfg.face_conf else _GOLD
            cv2.circle(out, (int(kx), int(ky)), 4, col, -1, cv2.LINE_AA)
            cv2.circle(out, (int(kx), int(ky)), 4, (60, 60, 60), 1, cv2.LINE_AA)

    _put(out, f"{len(boxes)} det", (4, h - 8), 0.38, (255, 255, 255), 1)
    return out


def _draw_crop(crop_bgr, orientation, extent, final_cls, panel_h, panel_w):
    """Panel 2: primary crop on white background with verdict text and border."""
    out = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)

    if crop_bgr is not None and crop_bgr.size > 0:
        ch, cw = crop_bgr.shape[:2]
        scale  = min((panel_h - 30) / max(ch, 1), (panel_w - 10) / max(cw, 1))
        nw, nh = int(cw * scale), int(ch * scale)
        resized = cv2.resize(crop_bgr, (nw, nh))
        x0 = (panel_w - nw) // 2
        y0 = 24
        out[y0:y0 + nh, x0:x0 + nw] = resized
        cv2.rectangle(out, (x0, y0), (x0 + nw, y0 + nh), _SEP, 1)

    vcol = _ACCENT if final_cls not in ('others', 'bad_extraction') else _OTHERS
    _put(out, f"ori: {orientation or 'none'}", (6, 14), 0.42, vcol)
    _put(out, f"ext: {extent or 'none'}",      (6, 30), 0.42, vcol)
    return out


def _draw_bar_chart(kp_confs, kp_names, cfg, panel_h, panel_w):
    """
    Panel 3: horizontal confidence bars for all 17 keypoints, drawn with CV2.
    Labels left-aligned in a fixed column; numeric values right of bars.
    """
    out = np.full((panel_h, panel_w, 3), _BG, dtype=np.uint8)

    n       = len(kp_names)
    top_pad = 28
    bot_pad = 30
    usable  = panel_h - top_pad - bot_pad
    row_h   = usable // n

    label_col_w = 80   # fixed width for keypoint names
    val_col_w   = 38   # fixed width for numeric value on the right
    bar_x0      = label_col_w + 4
    bar_max_w   = panel_w - bar_x0 - val_col_w - 8

    _put(out, "Keypoint confidences", (6, 16), 0.42, _MUTED)

    for i, (name, conf) in enumerate(zip(kp_names, kp_confs)):
        y_mid = top_pad + i * row_h + row_h // 2
        y_txt = y_mid + 5

        _put(out, name, (4, y_txt), 0.38, _MUTED, font=_FT)

        if conf < cfg.back_conf:
            bar_col = _RED
        elif conf < cfg.face_conf:
            bar_col = _GOLD
        else:
            bar_col = _GREEN

        bar_w = max(1, int(conf * bar_max_w))
        y_top = y_mid - row_h // 2 + 3
        y_bot = y_mid + row_h // 2 - 3
        cv2.rectangle(out, (bar_x0, y_top), (bar_x0 + bar_w, y_bot), bar_col, -1)

        _put(out, f"{conf:.2f}", (bar_x0 + bar_max_w + 4, y_txt),
             0.36, _TEXT, font=_FT)

    # Threshold lines
    for thresh, col, label in [
        (cfg.back_conf,  _GOLD,  f"bk={cfg.back_conf:.2f}"),
        (cfg.body_conf,  (180, 120, 60), f"bd={cfg.body_conf:.2f}"),
        (cfg.face_conf,  _GREEN, f"fc={cfg.face_conf:.2f}"),
    ]:
        lx = bar_x0 + int(thresh * bar_max_w)
        cv2.line(out, (lx, top_pad), (lx, panel_h - bot_pad), col, 1, cv2.LINE_AA)
        _put(out, label, (lx + 2, panel_h - 6), 0.30, col, font=_FT)

    return out


def _draw_trace(trace_rows, orientation, extent, final_cls, panel_h, panel_w):
    """
    Panel 4: rule-by-rule decision trace table drawn with CV2.
    PASS rows get a green tint; FAIL rows get a red tint.
    """
    out = np.full((panel_h, panel_w, 3), _BG, dtype=np.uint8)

    vcol = _ACCENT if final_cls not in ('others', 'bad_extraction') else _OTHERS

    _put(out, "Decision trace", (6, 16), 0.50, _TEXT, thickness=1)
    _put(out, f"ori={orientation or 'none'}  ext={extent or 'none'}",
         (6, 34), 0.40, _MUTED)
    _put(out, f"=> {final_cls}", (6, 52), 0.52, vcol, thickness=1)

    col_x    = [6, 140, 310, 430]    # Rule | Measured | Threshold | Result
    col_w    = [132, 168, 118, 60]
    hdr_y    = 70
    row_start= 84
    n_rows   = len(trace_rows)
    row_h    = (panel_h - row_start - 8) // max(n_rows, 1)

    # Header bar
    cv2.rectangle(out, (4, hdr_y - 14), (panel_w - 4, hdr_y + 4), _SEP, -1)
    for cx, label in zip(col_x, ["Rule", "Measured", "Threshold", "Result"]):
        _put(out, label, (cx + 2, hdr_y), 0.38, _MUTED, font=_FT)

    for ri, (rule, measured, threshold, result) in enumerate(trace_rows):
        y0   = row_start + ri * row_h
        y1   = y0 + row_h - 2
        ytxt = y0 + row_h - 6

        bg = _PASS_BG if result == "PASS" else _FAIL_BG
        cv2.rectangle(out, (4, y0), (panel_w - 4, y1), bg, -1)

        for cx, cw, text in zip(col_x, col_w, [rule, measured, threshold, result]):
            rcol = _GREEN if result == "PASS" and text == result else (
                   _RED   if result == "FAIL" and text == result else _TEXT)
            # Truncate text to fit column width
            s = text
            while len(s) > 1 and _text_w(s, 0.34, 1, _FT) > cw - 4:
                s = s[:-1]
            _put(out, s, (cx + 2, ytxt), 0.34, rcol, font=_FT)

        cv2.line(out, (4, y1), (panel_w - 4, y1), _SEP, 1)

    return out


def _no_detection_panel(img_bgr: np.ndarray, out_h: int) -> np.ndarray:
    """Return a panel image for the 'no detections' case."""
    h_img, w_img = img_bgr.shape[:2]
    total_w = w_img + 340 + 500 + 6
    out = np.full((out_h, total_w, 3), _BG, dtype=np.uint8)
    ph  = min(h_img, out_h)
    out[:ph, :w_img] = img_bgr[:ph]
    cv2.line(out, (w_img + 2, 0), (w_img + 2, out_h), _SEP, 2)
    _put(out, "No pose detections found",
         (w_img + 12, out_h // 2 - 10), 0.55, _OTHERS, thickness=1)
    _put(out, "=> classified as 'others'",
         (w_img + 12, out_h // 2 + 18), 0.48, _MUTED)
    return out


def _build_trace_rows(kps: np.ndarray, bbox: np.ndarray, cfg) -> list:
    """Build the decision-trace rows shared by both render_diagnostic variants."""
    nose, l_eye, r_eye = kps[0, 2], kps[1, 2], kps[2, 2]
    n_lower  = n_visible(kps, LOWER_KPS, cfg.body_conf)
    n_ankles = n_visible(kps, ANKLE_KPS, cfg.body_conf)
    n_shldrs = n_visible(kps, UPPER_KPS, cfg.body_conf)
    h_box    = bbox[3] - bbox[1]
    w_box    = (bbox[2] - bbox[0]) + 1e-6
    aspect   = h_box / w_box
    return [
        ("Front",
         f"n={nose:.2f} eL={l_eye:.2f} eR={r_eye:.2f}",
         f"all>={cfg.face_conf:.2f}",
         "PASS" if (nose >= cfg.face_conf and l_eye >= cfg.face_conf and r_eye >= cfg.face_conf) else "FAIL"),
        ("Back",
         f"n={nose:.2f} eyes_vis={sum(1 for c in (l_eye, r_eye) if c >= cfg.back_conf)}",
         f"n<{cfg.nose_back_conf:.2f} eyes<={cfg.max_eyes_for_back}",
         "PASS" if (nose < cfg.nose_back_conf and
                    sum(1 for c in (l_eye, r_eye) if c >= cfg.back_conf) <= cfg.max_eyes_for_back)
                else "FAIL"),
        ("Lower-body",
         f"lower={n_lower} ankles={n_ankles}",
         "ankles>=1 OR lower>=3",
         "PASS" if (n_ankles >= 1 or n_lower >= 3) else "FAIL"),
        ("Aspect",
         f"h/w={aspect:.2f}",
         f">={cfg.aspect_ratio_min:.1f}",
         "PASS" if aspect >= cfg.aspect_ratio_min else "FAIL"),
        ("Shoulder",
         f"vis={n_shldrs}",
         f">=1@{cfg.body_conf:.2f}",
         "PASS" if n_shldrs >= 1 else "FAIL"),
    ]


def _render_panels(
    img_bgr: np.ndarray,
    boxes: np.ndarray,
    kp_data: np.ndarray,
    primary: int,
    kps: np.ndarray,
    bbox: np.ndarray,
    cfg,
    predicted_class,
) -> np.ndarray:
    """Build and concatenate the 4 diagnostic panels (detection, crop, bars, trace)."""
    h_img, w_img = img_bgr.shape[:2]
    OUT_H = max(h_img, 380)

    x1c = max(0, int(bbox[0])); y1c = max(0, int(bbox[1]))
    x2c = min(w_img, int(bbox[2])); y2c = min(h_img, int(bbox[3]))
    crop_bgr = img_bgr[y1c:y2c, x1c:x2c]

    orientation = classify_orientation(kps, cfg)
    extent      = classify_extent(kps, bbox, cfg)
    final_cls   = predicted_class if predicted_class is not None \
                  else classify_keypoints(kps, bbox, cfg)

    trace_rows = _build_trace_rows(kps, bbox, cfg)

    scale_p1 = OUT_H / max(h_img, 1)
    p1_w     = int(w_img * scale_p1)
    p1 = _draw_detections(img_bgr, boxes, kp_data, primary, cfg)
    p1 = cv2.resize(p1, (p1_w, OUT_H), interpolation=cv2.INTER_LINEAR)

    crop_panel_w = max(120, min(200, int(OUT_H * 0.5)))
    p2 = _draw_crop(crop_bgr, orientation, extent, final_cls, OUT_H, crop_panel_w)

    bar_panel_w = 240
    p3 = _draw_bar_chart(kps[:, 2], COCO_KP_NAMES, cfg, OUT_H, bar_panel_w)

    trace_panel_w = 500
    p4 = _draw_trace(trace_rows, orientation, extent, final_cls, OUT_H, trace_panel_w)

    sep = np.full((OUT_H, 2, 3), _SEP, dtype=np.uint8)
    out = np.concatenate([p1, sep, p2, sep, p3, sep, p4], axis=1)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), _SEP, 1)
    return out


def render_diagnostic(
    result,
    img_bgr,
    cfg=None,
    predicted_class=None,
):
    """
    Render classification diagnostic as a BGR image using CV2 only (~10-20x
    faster than the matplotlib version).  Matches the matplotlib layout and
    colour scheme as closely as possible.

    4-panel layout (left to right):
      1. Detection overlay — bboxes + skeleton on the original patch
      2. Primary bbox crop — white background, labelled
      3. Keypoint confidence bars — light background, labelled
      4. Decision trace table  — PASS/FAIL cells with colour coding

    Args:
        result:          Single YOLO pose result (already inferred).
        img_bgr:         Original BGR image.
        cfg:             ClassifierConfig; uses DEFAULT_CONFIG if None.
        predicted_class: Pre-computed class string; re-derived if None.

    Returns:
        BGR numpy array (H, W, 3).
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    h_img = img_bgr.shape[0]
    OUT_H = max(h_img, 380)

    if result.keypoints is None or result.keypoints.data.shape[0] == 0:
        return _no_detection_panel(img_bgr, OUT_H)

    boxes   = result.boxes.xyxy.cpu().numpy()
    kp_data = result.keypoints.data.cpu().numpy()
    areas   = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    primary = int(np.argmax(areas))
    return _render_panels(img_bgr, boxes, kp_data, primary, kp_data[primary], boxes[primary], cfg, predicted_class)


def render_diagnostic_from_kps(
    img_bgr,
    kps,
    bbox,
    cfg=None,
    predicted_class=None,
):
    """
    Render the same 4-panel classification diagnostic as render_diagnostic,
    but accepts pre-computed numpy arrays instead of a live YOLO result object.
    Useful for the annotator when keypoints have already been cached in _keypoints.npz.

    Args:
        img_bgr:         Original BGR image (H, W, 3).
        kps:             np.ndarray [17, 3] (x, y, conf) or None if no detection.
        bbox:            np.ndarray [4] (x1, y1, x2, y2) or None if no detection.
        cfg:             ClassifierConfig; uses DEFAULT_CONFIG if None.
        predicted_class: Pre-computed class string; derived from kps if None.

    Returns:
        BGR numpy array (H, W, 3).
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    h_img = img_bgr.shape[0]
    OUT_H = max(h_img, 380)

    if kps is None or bbox is None:
        return _no_detection_panel(img_bgr, OUT_H)

    # Single-detection layout: wrap kps/bbox so _draw_detections gets [1, ...] arrays.
    return _render_panels(img_bgr, bbox[np.newaxis], kps[np.newaxis], 0, kps, bbox, cfg, predicted_class)


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
    save_keypoints: bool = True,
) -> tuple[dict, dict]:
    """
    Classify all .jpg/.png files in input_dir using batched YOLO inference.

    Args:
        pose_model:     YOLO26 pose model instance.
        input_dir:      Directory of cropped human patches.
        output_dir:     Root output directory; per-class subdirs created automatically.
        cfg:            ClassifierConfig controlling thresholds and ablation flags.
        batch_size:     Images per inference call.
        copy_files:     If True, copy patches into per-class subdirs.
        save_debug_viz: If True, save deep diagnostic images to output_dir/.diag_cache/.
        save_keypoints: If True, save raw keypoint tensors to
                        input_dir/_keypoints.npz for downstream use (GCN, annotator).
                        Stored alongside the patches so they travel with the data.

    Returns:
        results: dict mapping filename -> class string.
        summary: dict mapping class string -> count.
    """
    image_paths = sorted(
        glob.glob(os.path.join(input_dir, '*.jpg')) +
        glob.glob(os.path.join(input_dir, '*.png'))
    )
    if not image_paths:
        print(f"[CLS] No images found in {input_dir}")
        return {}, {}

    if copy_files:
        for cls in CLASSES:
            os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    if save_debug_viz:
        os.makedirs(os.path.join(output_dir, '.diag_cache'), exist_ok=True)

    results       = {}
    summary       = {cls: 0 for cls in CLASSES}
    keypoints_buf = {} if save_keypoints else None   # fname -> {kps, bbox} | None

    n_batches = (len(image_paths) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(image_paths), batch_size), total=n_batches, desc="Classifying", unit="batch"):
        batch_paths   = image_paths[i: i + batch_size]
        batch_results = pose_model(batch_paths, verbose=False)

        for img_path, result in zip(batch_paths, batch_results):
            fname = os.path.basename(img_path)

            if result.keypoints is None or result.keypoints.data.shape[0] == 0:
                cls = 'others'
                if keypoints_buf is not None:
                    keypoints_buf[fname] = None
            else:
                boxes    = result.boxes.xyxy.cpu().numpy()
                areas    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_idx = int(np.argmax(areas))
                kps_np   = result.keypoints.data[best_idx].cpu().numpy()
                bbox_np  = boxes[best_idx]

                cls = classify_keypoints(kps_np, bbox=bbox_np, cfg=cfg)

                if keypoints_buf is not None:
                    keypoints_buf[fname] = {"kps": kps_np, "bbox": bbox_np}

            results[fname]  = cls
            summary[cls]   += 1

            if copy_files:
                shutil.copy(img_path, os.path.join(output_dir, cls, fname))

            if save_debug_viz:
                img_bgr = cv2.imread(img_path)
                if img_bgr is not None:
                    diag = render_diagnostic(result, img_bgr, cfg, predicted_class=cls)
                    cv2.imwrite(os.path.join(output_dir, '.diag_cache', fname), diag)

    # Persist keypoints next to the source patches
    if keypoints_buf is not None:
        from src.gcn import save_keypoints as _save_kp
        npz_path = os.path.join(input_dir, "_keypoints.npz")
        _save_kp(keypoints_buf, npz_path)
        print(f"[CLS] Keypoints saved → {npz_path}")

    print(f"[CLS] Done. {dict((k, v) for k, v in summary.items() if v > 0)}")
    return results, summary


# ---------------------------------------------------------------------------
# Summary persistence
# ---------------------------------------------------------------------------

def save_classification_summary(
    save_dir: str,
    summary: dict,
    input_dir: str,
) -> str:
    """
    Write a human-readable classification summary to save_dir/_summary.txt.

    Args:
        save_dir  : classification output directory (e.g. init_classifications/<run>).
        summary   : dict mapping class name → count (as returned by classify_directory).
        input_dir : source patch directory (written into the summary for traceability).

    Returns:
        Path to the written summary file.
    """
    import time as _time
    total = sum(summary.values())
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, "_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Rule-based classification summary\n")
        f.write(f"Save time: {_time.strftime('%Y%m%d-%H%M%S')}\n")
        f.write(f"Input: {input_dir}\n")
        f.write(f"Total patches classified: {total}\n\n")
        f.write(f"{'Class':<25} {'Count':>6}  {'%':>6}\n")
        f.write("-" * 42 + "\n")
        for cls, count in sorted(summary.items(), key=lambda x: -x[1]):
            pct = 100 * count / total if total else 0
            f.write(f"{cls:<25} {count:>6}  {pct:>5.1f}%\n")
    print(f"[CLS] Summary saved → {summary_path}")
    return summary_path


# ---------------------------------------------------------------------------
# Reload from existing output directory
# ---------------------------------------------------------------------------

def reload_classification_results(cls_save_path: str) -> tuple[dict, dict]:
    """
    Reconstruct results and summary from an existing classification directory.
    Works for both init_classifications and gcn_results directories.
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
    print(f"[CLS] Loaded {total} patches from {cls_save_path}")
    print("[CLS] " + "  ".join(f"{k}: {v}" for k, v in summary.items() if v > 0))
    return results, summary