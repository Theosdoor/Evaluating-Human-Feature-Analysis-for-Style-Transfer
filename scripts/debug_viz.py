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

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %%
# ── Change these three variables to inspect a different patch ───────────────
DIAG_RUN   = "20260314-195748-4"
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
