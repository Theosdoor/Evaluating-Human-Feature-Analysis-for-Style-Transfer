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

def render_diagnostic(
    result,
    img_bgr: np.ndarray,
    cfg: ClassifierConfig = DEFAULT_CONFIG,
    predicted_class: str | None = None,
) -> np.ndarray:
    """
    Render the deep classification diagnostic as a BGR image.

    Produces a 4-panel figure:
      1. Original patch with all YOLO detections + keypoints annotated
      2. Primary bbox crop
      3. Per-keypoint confidence bar chart
      4. Rule-by-rule decision trace table

    Args:
        result:           Single YOLO pose result object (already inferred).
        img_bgr:          Original BGR image the result was run on.
        cfg:              ClassifierConfig used for classification thresholds.
        predicted_class:  Optional pre-computed class label to show in title.
                          If None, re-derives it from result.

    Returns:
        BGR numpy array (H, W, 3) suitable for cv2.imwrite or display.
        Returns a plain grey placeholder if no pose detections were found.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from io import BytesIO

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]

    # --- No detections: return a simple placeholder ---
    if result.keypoints is None or result.keypoints.data.shape[0] == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.imshow(img_rgb)
        ax.set_title("No pose detections found\n→ classified as 'others'", color="orange", fontsize=10)
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        arr = np.frombuffer(buf.read(), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    boxes   = result.boxes.xyxy.cpu().numpy()
    kp_data = result.keypoints.data.cpu().numpy()
    areas   = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    primary = int(np.argmax(areas))
    kps     = kp_data[primary]
    bbox    = boxes[primary]

    # Primary crop
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w_img, x2), min(h_img, y2)
    crop_rgb  = img_rgb[y1c:y2c, x1c:x2c]

    # Classification
    orientation = classify_orientation(kps, cfg)
    extent      = classify_extent(kps, bbox, cfg)
    final_cls   = predicted_class if predicted_class is not None else classify_keypoints(kps, bbox, cfg)

    # Intermediate values for trace
    nose      = kps[0, 2]
    l_eye     = kps[1, 2]
    r_eye     = kps[2, 2]
    n_lower   = n_visible(kps, LOWER_KPS, cfg.body_conf)
    n_ankles  = n_visible(kps, ANKLE_KPS, cfg.body_conf)
    n_shldrs  = n_visible(kps, UPPER_KPS, cfg.body_conf)
    h_box     = bbox[3] - bbox[1]
    w_box     = (bbox[2] - bbox[0]) + 1e-6
    aspect    = h_box / w_box

    front_pass  = nose >= cfg.face_conf and l_eye >= cfg.face_conf and r_eye >= cfg.face_conf
    back_pass   = nose < cfg.nose_back_conf and sum(
        1 for c in (l_eye, r_eye) if c >= cfg.back_conf
    ) <= cfg.max_eyes_for_back
    has_lower   = n_ankles >= 1 or n_lower >= 3
    aspect_pass = aspect >= cfg.aspect_ratio_min
    has_shldr   = n_shldrs >= 1

    trace_rows = [
        ["Front check",
         f"nose={nose:.2f} eyeL={l_eye:.2f} eyeR={r_eye:.2f}",
         f">= face_conf({cfg.face_conf:.2f})",
         "PASS" if front_pass else "FAIL"],
        ["Back check",
         f"nose={nose:.2f} eyeL={l_eye:.2f} eyeR={r_eye:.2f}",
         f"nose<{cfg.nose_back_conf:.2f}, eyes<={cfg.max_eyes_for_back}",
         "PASS" if back_pass else "FAIL"],
        ["Lower-body",
         f"lower={n_lower} ankles={n_ankles}",
         "ankles>=1 OR lower>=3",
         "PASS" if has_lower else "FAIL"],
        ["Aspect ratio",
         f"h/w={aspect:.2f}",
         f">= {cfg.aspect_ratio_min:.1f} for full_body",
         "PASS" if aspect_pass else "FAIL"],
        ["Shoulder",
         f"visible={n_shldrs}",
         f">=1 @ body_conf({cfg.body_conf:.2f})",
         "PASS" if has_shldr else "FAIL"],
    ]

    kp_confs = kps[:, 2]

    # --- Figure ---
    fig = plt.figure(figsize=(20, 6))
    gs  = fig.add_gridspec(1, 4, width_ratios=[3, 1.5, 1.5, 3], wspace=0.3)

    # Panel 1: all detections + keypoints
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_rgb)
    for i, (box, area) in enumerate(zip(boxes, areas)):
        bx1, by1, bx2, by2 = box
        color = "red" if i == primary else "dodgerblue"
        lw    = 2.5  if i == primary else 1.2
        rect  = mpatches.Rectangle(
            (bx1, by1), bx2 - bx1, by2 - by1,
            linewidth=lw, edgecolor=color, facecolor="none",
        )
        ax1.add_patch(rect)
        ax1.text(bx1, by1 - 4, f"#{i} area={int(area)}", color=color, fontsize=7)
    for ki, (kx, ky, kc) in enumerate(kps):
        if kc >= cfg.back_conf:
            c = "lime" if kc >= cfg.face_conf else "gold"
            ax1.plot(kx, ky, "o", color=c, markersize=4)
            ax1.text(kx + 2, ky, COCO_KP_NAMES[ki], color=c, fontsize=5)
    ax1.set_title(
        f"All detections ({len(boxes)} found)\n"
        f"Red=primary  green kp>=face_conf={cfg.face_conf:.2f}  gold>=back_conf",
        fontsize=8,
    )
    ax1.axis("off")

    # Panel 2: primary crop
    ax2 = fig.add_subplot(gs[1])
    if crop_rgb.size > 0:
        ax2.imshow(crop_rgb)
    verdict_color = "green" if final_cls != "others" else "orange"
    ax2.set_title(
        f"Primary crop\norientation={orientation}  extent={extent}",
        fontsize=8, color=verdict_color,
    )
    ax2.axis("off")

    # Panel 3: keypoint confidence bars
    ax3 = fig.add_subplot(gs[2])
    bar_colors = [
        "tomato" if c < cfg.back_conf
        else ("gold" if c < cfg.face_conf else "limegreen")
        for c in kp_confs
    ]
    ax3.barh(range(17), kp_confs, color=bar_colors)
    ax3.axvline(cfg.back_conf,  color="gold",       linestyle=":",  linewidth=1,
                label=f"back_conf={cfg.back_conf:.2f}")
    ax3.axvline(cfg.body_conf,  color="dodgerblue", linestyle="-.", linewidth=1,
                label=f"body_conf={cfg.body_conf:.2f}")
    ax3.axvline(cfg.face_conf,  color="limegreen",  linestyle="--", linewidth=1,
                label=f"face_conf={cfg.face_conf:.2f}")
    ax3.set_yticks(range(17))
    ax3.set_yticklabels(COCO_KP_NAMES, fontsize=7)
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("confidence", fontsize=8)
    ax3.set_title("Keypoint confidences\n(primary detection)", fontsize=8)
    ax3.invert_yaxis()
    ax3.legend(fontsize=6, loc="lower right")

    # Panel 4: decision trace table
    ax4 = fig.add_subplot(gs[3])
    ax4.axis("off")
    col_labels = ["Rule", "Measured", "Threshold", "Result"]
    tbl = ax4.table(cellText=trace_rows, colLabels=col_labels,
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.auto_set_column_width(range(len(col_labels)))
    # Colour PASS/FAIL cells
    for (row, col), cell in tbl.get_celld().items():
        if col == 3 and row > 0:
            cell.set_facecolor("#d4edda" if cell.get_text().get_text() == "PASS" else "#f8d7da")
    ax4.set_title(
        f"Decision trace\norientation={orientation}  extent={extent}\n"
        f"Predicted: {final_cls}",
        fontsize=8, color=verdict_color,
    )

    # Render to BGR numpy array
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


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
        save_debug_viz: If True, save deep diagnostic images to output_dir/debug_viz/.

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
                    cv2.imwrite(os.path.join(output_dir, 'debug_viz', fname), diag)

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