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
# Classification viewer — paste a relative path to a classified patch and see
# the original crop alongside its debug_viz annotation.
#
# Example:
#   VIEW_PATH = "output/classifications/20260225-104100/full_body_front/human_0001_TheGodfather_f001959_conf0.91_score0.87.jpg"

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


# ── Set this to any classified-patch relative path and re-run the cell ──────
view_path = "output/classifications/20260225-104100/full_body_front/human_0001_TheGodfather_f001959_conf0.91_score0.87.jpg"
show_patch_debug(view_path)

# %%
# ── Deep classification diagnostic ──────────────────────────────────────────
# Runs the full pipeline on a single patch and annotates every step:
#   col 1 — original patch with ALL YOLO pose detections (primary = red box)
#   col 2 — primary bbox crop sent to MediaPipe for face detection
#   col 3 — keypoint confidence heatmap for the primary detection
#   col 4 — score trace table (front / back contributions per rule)

import importlib
import src.classification as _clf_mod

COCO_KP_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

DIAG_PATH = os.path.join(
    PROJECT_ROOT,
    "output/classifications/20260225-104100/full_body_front/"
    "human_0001_TheGodfather_f001959_conf0.91_score0.87.jpg"
)

# --- load models if not in scope ---
if "pose_model" not in dir():
    _pose_model = YOLO(os.path.join(PROJECT_ROOT, "models/yolo26m-pose.pt"))
    _pose_model.to(DEVICE)
else:
    _pose_model = pose_model  # noqa: F821

# Build face detector using Tasks API (auto-downloads model if needed)
importlib.reload(_clf_mod)
_face_detector = _clf_mod.build_face_detector()

# --- run YOLO ---
_result  = _pose_model(DIAG_PATH, verbose=False)[0]
_img_bgr = cv2.imread(DIAG_PATH)
_img_rgb = cv2.cvtColor(_img_bgr, cv2.COLOR_BGR2RGB)
_h, _w   = _img_rgb.shape[:2]

# --- select primary detection (largest bbox area) ---
_boxes   = _result.boxes.xyxy.cpu().numpy()         # [N, 4]
_kp_data = _result.keypoints.data.cpu().numpy()     # [N, 17, 3]
_areas   = (_boxes[:, 2] - _boxes[:, 0]) * (_boxes[:, 3] - _boxes[:, 1])
_primary = int(np.argmax(_areas))
_kps     = _kp_data[_primary]                       # [17, 3]
_bbox    = _boxes[_primary]
_ch      = _clf_mod.adaptive_conf_high(_kps)

# --- primary crop for MediaPipe (scoped to YOLO bbox) ---
_x1, _y1, _x2, _y2 = [int(v) for v in _bbox]
_x1c, _y1c = max(0, _x1), max(0, _y1)
_x2c, _y2c = min(_w, _x2), min(_h, _y2)
_crop_rgb  = _img_rgb[_y1c:_y2c, _x1c:_x2c]
_face_det  = _clf_mod._run_face_detection(_face_detector, DIAG_PATH, _bbox)

# --- score trace ---
_, _trace = _clf_mod.classify_orientation_debug(_kps, conf_high=_ch, face_detected=_face_det)

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
    rect  = plt.Rectangle((bx1, by1), bx2 - bx1, by2 - by1,
                           linewidth=lw, edgecolor=color, facecolor="none")
    ax1.add_patch(rect)
    ax1.text(bx1, by1 - 4, f"#{i} area={int(area)}", color=color, fontsize=7)
for ki, (kx, ky, kc) in enumerate(_kps):
    if kc >= _clf_mod.CONF_LOW:
        c = "lime" if kc >= _ch else "yellow"
        ax1.plot(kx, ky, "o", color=c, markersize=4)
        ax1.text(kx + 2, ky, COCO_KP_NAMES[ki], color=c, fontsize=5)
ax1.set_title(
    f"All detections ({len(_boxes)} found)\n"
    f"Red=primary  Green kp >= conf_high={_ch:.2f}  Yellow=low-conf",
    fontsize=8,
)
ax1.axis("off")

# Panel 2: primary bbox crop
ax2 = fig.add_subplot(gs[1])
ax2.imshow(_crop_rgb)
ax2.set_title(
    f"Primary bbox crop\nMediaPipe face: {'YES' if _face_det else 'NO'}",
    fontsize=8,
    color="green" if _face_det else "red",
)
ax2.axis("off")

# Panel 3: per-keypoint confidence bars
ax3 = fig.add_subplot(gs[2])
bar_colors = [
    "tomato" if c < _clf_mod.CONF_LOW else ("gold" if c < _ch else "limegreen")
    for c in _kp_confs
]
ax3.barh(range(17), _kp_confs, color=bar_colors)
ax3.axvline(_ch, color="limegreen", linestyle="--", linewidth=1, label=f"conf_high={_ch:.2f}")
ax3.axvline(_clf_mod.CONF_LOW, color="gold", linestyle=":", linewidth=1, label=f"conf_low={_clf_mod.CONF_LOW:.2f}")
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
col_labels = ["Rule", "Dfront", "Dback", "front", "back", "Note"]
rows = []
for s in _trace["steps"]:
    rows.append([
        s["label"],
        f"{s['delta_front']:+.1f}" if s["delta_front"] else "-",
        f"{s['delta_back']:+.1f}"  if s["delta_back"]  else "-",
        f"{s['front_score']:.2f}",
        f"{s['back_score']:.2f}",
        s["note"][:32],
    ])
tbl = ax4.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
tbl.auto_set_column_width(range(len(col_labels)))
verdict_color = "green" if _trace["decision"] == "back" else ("red" if _trace["decision"] == "front" else "orange")
ax4.set_title(
    f"Score trace  front={_trace['front_score']}  back={_trace['back_score']}\n"
    f"override_fired={_trace['override_fired']}  margin={_trace['margin']:.2f}\n"
    f"Decision: {_trace['decision']}",
    fontsize=8, color=verdict_color,
)

plt.suptitle(os.path.basename(DIAG_PATH), fontsize=9, y=1.01)
plt.savefig(os.path.join(PROJECT_ROOT, "output", "diag_classification.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to output/diag_classification.png")

# %%
