# %%
import os
import sys
import subprocess

# Add project root to path for imports
# scripts/ lives one level below the project root, so go up one directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# clone & get deps if in colab
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    # install deps from pyproject.toml
    url = "https://github.com/Theosdoor/ACV_cswk.git"
    subprocess.run(["git", "clone", url], check=True)
    os.chdir("ACV_cswk")
    subprocess.run(["uv", "pip", "install", "--system", "-r", "pyproject.toml"], check=True)

import time
import numpy as np
import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO # https://github.com/ultralytics/ultralytics

from src.feat_extract import *
from src.classification import *

DATA_DIR = os.path.join(PROJECT_ROOT, "downloaded_data") # name of dir where downloaded videos are
TRAIN_PATHS = [
    os.path.join(DATA_DIR, "Train/game/MafiaVideogame.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheGodfather.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheIrishman.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheSopranos.mp4"),
]

TEST_PATH = os.path.join(DATA_DIR,"Test/Test.mp4")
SAVE_DIR = os.path.join(PROJECT_ROOT, "output")
SAVE_NAME = time.strftime('%Y%m%d-%H%M%S')

RELOAD_RUN = "20260224-224712"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %%
# ── Change these three variables to inspect a different patch ───────────────
DIAG_RUN   = "20260314-195748-1"
DIAG_CLASS = "others"
DIAG_IMAGE = "human_0002_TheGodfather_f001504_conf0.92_score0.86"

# %%
# Classification viewer — paste a relative path to a classified patch and see
# the original crop alongside its debug_viz annotation.
#
# Example:
#   VIEW_PATH = "output/classifications/20260225-104100/full_body_front/human_0001_TheGodfather_f001959_conf0.91_score0.87.jpg"

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

def show_patch_debug(rel_path: str, root: str = PROJECT_ROOT) -> None:
    """Display a classified patch and its debug_viz counterpart side by side.

    Args:
        rel_path: Workspace-relative path to the patch, e.g.
            'output/classifications/<timestamp>/<class>/<filename>.jpg'
        root: Absolute path to workspace root (default: PROJECT_ROOT).
    """
    abs_path = os.path.join(root, rel_path) if not os.path.isabs(rel_path) else rel_path

    # Derive debug_viz path: swap the class subfolder for 'debug_viz'
    # Structure: <run_dir>/<class_folder>/<filename>
    filename  = os.path.basename(abs_path)
    run_dir   = os.path.dirname(os.path.dirname(abs_path))  # up two levels
    debug_path = os.path.join(run_dir, "debug_viz", filename)

    print(f"root      : {root}")
    print(f"patch     : {abs_path}  [{'EXISTS' if os.path.exists(abs_path) else 'MISSING'}]")
    print(f"debug_viz : {debug_path}  [{'EXISTS' if os.path.exists(debug_path) else 'MISSING'}]")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, path, title in zip(
        axes,
        [abs_path, debug_path],
        ["Classified patch", "Debug viz"],
    ):
        if os.path.exists(path):
            ax.imshow(mpimg.imread(path))
            ax.set_title(f"{title}\n{os.path.basename(path)}", fontsize=9)
        else:
            ax.text(0.5, 0.5, f"Not found:\n{path}",
                    ha="center", va="center", transform=ax.transAxes, color="red", fontsize=7, wrap=True)
            ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()




view_path = os.path.join("output", "classifications", DIAG_RUN, DIAG_CLASS, DIAG_IMAGE + ".jpg")
show_patch_debug(view_path)

# %%

# %%
# ── Deep classification diagnostic ──────────────────────────────────────────
# Runs the full pipeline on a single patch and annotates every step:
#   col 1 — original patch with ALL YOLO pose detections (primary = red box)
#   col 2 — primary bbox crop selected for classification
#   col 3 — keypoint confidence heatmap for the primary detection
#   col 4 — rule-by-rule decision trace (pose-only pipeline)

import src.classification as _clf_mod

COCO_KP_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

# Uses DIAG_RUN / DIAG_CLASS / DIAG_IMAGE defined in the viewer cell above
DIAG_PATH = os.path.join(PROJECT_ROOT, "output", "classifications", DIAG_RUN, DIAG_CLASS, DIAG_IMAGE + ".jpg")

# --- load models if not in scope ---
if "pose_model" not in globals():
    _pose_model = YOLO(os.path.join(PROJECT_ROOT, "models/yolo26m-pose.pt"))
    _pose_model.to(DEVICE)
else:
    _pose_model = globals()["pose_model"]

# --- run YOLO ---
_result  = _pose_model(DIAG_PATH, verbose=False)[0]
_img_bgr = cv2.imread(DIAG_PATH)
if _img_bgr is None:
    raise FileNotFoundError(f"Could not read image: {DIAG_PATH}")
if _result.keypoints is None or _result.keypoints.data.shape[0] == 0:
    raise RuntimeError(f"No YOLO pose detections found for: {DIAG_PATH}")
_img_rgb = cv2.cvtColor(_img_bgr, cv2.COLOR_BGR2RGB)
_h, _w   = _img_rgb.shape[:2]

# --- select primary detection (largest bbox area) ---
_boxes   = _result.boxes.xyxy.cpu().numpy()         # [N, 4]
_kp_data = _result.keypoints.data.cpu().numpy()     # [N, 17, 3]
_areas   = (_boxes[:, 2] - _boxes[:, 0]) * (_boxes[:, 3] - _boxes[:, 1])
_primary = int(np.argmax(_areas))
_kps     = _kp_data[_primary]                       # [17, 3]
_bbox    = _boxes[_primary]

# --- primary crop (scoped to YOLO bbox) ---
_x1, _y1, _x2, _y2 = [int(v) for v in _bbox]
_x1c, _y1c = max(0, _x1), max(0, _y1)
_x2c, _y2c = min(_w, _x2), min(_h, _y2)
_crop_rgb  = _img_rgb[_y1c:_y2c, _x1c:_x2c]

# --- pose-only decision trace ---
_orientation = _clf_mod.classify_orientation(_kps)
_extent = _clf_mod.classify_extent(_kps, _bbox)
_final_cls = _clf_mod.classify_keypoints(_kps, _bbox)

_nose = _kps[0, 2]
_l_eye = _kps[1, 2]
_r_eye = _kps[2, 2]

_n_lower = _clf_mod.n_visible(_kps, _clf_mod.LOWER_KPS, _clf_mod.BODY_CONF)
_n_ankles = _clf_mod.n_visible(_kps, _clf_mod.ANKLE_KPS, _clf_mod.BODY_CONF)
_n_shoulders = _clf_mod.n_visible(_kps, _clf_mod.UPPER_KPS, _clf_mod.BODY_CONF)
_h_box = _bbox[3] - _bbox[1]
_w_box = (_bbox[2] - _bbox[0]) + 1e-6
_aspect = _h_box / _w_box

_front_pass = (_nose >= _clf_mod.FACE_CONF and _l_eye >= _clf_mod.FACE_CONF and _r_eye >= _clf_mod.FACE_CONF)
_back_pass = (_nose < _clf_mod.BACK_CONF and _l_eye < _clf_mod.BACK_CONF and _r_eye < _clf_mod.BACK_CONF)
_has_lower = (_n_ankles >= 1) or (_n_lower >= 3)
_aspect_pass = _aspect >= 1.5
_has_shoulder = _n_shoulders >= 1

_trace_rows = [
    [
        "Front check",
        f"nose={_nose:.2f} eyeL={_l_eye:.2f} eyeR={_r_eye:.2f}",
        f">= FACE_CONF({_clf_mod.FACE_CONF:.2f})",
        "PASS" if _front_pass else "FAIL",
    ],
    [
        "Back check",
        f"nose={_nose:.2f} eyeL={_l_eye:.2f} eyeR={_r_eye:.2f}",
        f"< BACK_CONF({_clf_mod.BACK_CONF:.2f})",
        "PASS" if _back_pass else "FAIL",
    ],
    [
        "Lower-body evidence",
        f"lower={_n_lower} ankles={_n_ankles}",
        "ankles>=1 OR lower>=3",
        "PASS" if _has_lower else "FAIL",
    ],
    [
        "Aspect ratio",
        f"h/w={_aspect:.2f}",
        ">= 1.5 for full_body",
        "PASS" if _aspect_pass else "FAIL",
    ],
    [
        "Shoulder evidence",
        f"visible_shoulders={_n_shoulders}",
        f">=1 @ BODY_CONF({_clf_mod.BODY_CONF:.2f})",
        "PASS" if _has_shoulder else "FAIL",
    ],
]

# --- keypoint confidence bar colours ---
_kp_confs = _kps[:, 2]

# ── Figure: 4-panel diagnostic ───────────────────────────────────────────────
fig = plt.figure(figsize=(20, 8))
gs  = fig.add_gridspec(1, 4, width_ratios=[3, 1.5, 1.5, 3], wspace=0.3)

# Panel 1: all detections
ax1 = fig.add_subplot(gs[0])
ax1.imshow(_img_rgb)
for i, (box, area) in enumerate(zip(_boxes, _areas)):
    bx1, by1, bx2, by2 = box
    color = "red" if i == _primary else "dodgerblue"
    lw    = 2.5  if i == _primary else 1.2
    rect  = mpatches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1,
                               linewidth=lw, edgecolor=color, facecolor="none")
    ax1.add_patch(rect)
    ax1.text(bx1, by1 - 4, f"#{i} area={int(area)}", color=color, fontsize=7)
for ki, (kx, ky, kc) in enumerate(_kps):
    if kc >= _clf_mod.BACK_CONF:
        c = "lime" if kc >= _clf_mod.FACE_CONF else "gold"
        ax1.plot(kx, ky, "o", color=c, markersize=4)
        ax1.text(kx + 2, ky, COCO_KP_NAMES[ki], color=c, fontsize=5)
ax1.set_title(
    f"All detections ({len(_boxes)} found)\n"
    f"Red=primary  Green kp >= FACE_CONF={_clf_mod.FACE_CONF:.2f}  Gold>=BACK_CONF",
    fontsize=8,
)
ax1.axis("off")

# Panel 2: primary bbox crop
ax2 = fig.add_subplot(gs[1])
ax2.imshow(_crop_rgb)
ax2.set_title(
    f"Primary bbox crop\norientation={_orientation}  extent={_extent}",
    fontsize=8,
    color="green" if _final_cls != "others" else "orange",
)
ax2.axis("off")

# Panel 3: per-keypoint confidence bars
ax3 = fig.add_subplot(gs[2])
bar_colors = [
    "tomato" if c < _clf_mod.BACK_CONF else ("gold" if c < _clf_mod.FACE_CONF else "limegreen")
    for c in _kp_confs
]
ax3.barh(range(17), _kp_confs, color=bar_colors)
ax3.axvline(_clf_mod.BACK_CONF, color="gold", linestyle=":", linewidth=1, label=f"back_conf={_clf_mod.BACK_CONF:.2f}")
ax3.axvline(_clf_mod.BODY_CONF, color="dodgerblue", linestyle="-.", linewidth=1, label=f"body_conf={_clf_mod.BODY_CONF:.2f}")
ax3.axvline(_clf_mod.FACE_CONF, color="limegreen", linestyle="--", linewidth=1, label=f"face_conf={_clf_mod.FACE_CONF:.2f}")
ax3.set_yticks(range(17))
ax3.set_yticklabels(COCO_KP_NAMES, fontsize=7)
ax3.set_xlim(0, 1)
ax3.set_xlabel("confidence", fontsize=8)
ax3.set_title("Keypoint confidences\n(primary detection)", fontsize=8)
ax3.invert_yaxis()
ax3.legend(fontsize=6, loc="lower right")

# Panel 4: score trace table
ax4 = fig.add_subplot(gs[3])
ax4.axis("off")
col_labels = ["Rule", "Measured", "Threshold", "Result"]
tbl = ax4.table(cellText=_trace_rows, colLabels=col_labels, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
tbl.auto_set_column_width(range(len(col_labels)))
verdict_color = "green" if _final_cls != "others" else "orange"
ax4.set_title(
    f"Decision trace (pose-only)\n"
    f"orientation={_orientation}  extent={_extent}\n"
    f"Final class: {_final_cls}",
    fontsize=8, color=verdict_color,
)

plt.suptitle(os.path.basename(DIAG_PATH), fontsize=9, y=1.01)
plt.show()

# %%

# %%
