# %%
# blabla

# %%
import os
import sys
import subprocess

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

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
    # os.path.join(DATA_DIR, "Train/game/MafiaVideogame.mp4"), # TODO v large, so temporarily ignore
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
# 1.1. Human Patch Extraction
extract_save_path = f"{SAVE_DIR}/extracted_humans/{SAVE_NAME}"
n2save = 1000
detection_b_size = 32  # 2080 Ti (11 GB VRAM) comfortably handles 32 frames/batch

# %% 
model = YOLO(os.path.join(PROJECT_ROOT, 'models/yolov8m.pt'))
model.to(DEVICE)

# Per-video budget: distribute n2save evenly; remainder goes to last video
per_video_target = n2save // len(TRAIN_PATHS)
targets = [per_video_target] * len(TRAIN_PATHS)
targets[-1] += n2save - sum(targets)  # absorb rounding remainder

detections = []         # all raw detections (for diagnostics)
selected_detections = []

for path, target in tqdm(zip(TRAIN_PATHS, targets), desc="Processing training videos", unit="video", total=len(TRAIN_PATHS)):
    video_dets = extract_humans_from_video(model, path, yolo_batch_size=detection_b_size)

    for det in video_dets:
        det['score'] = score_detection(det)
    video_dets.sort(key=lambda x: x['score'], reverse=True)

    selected = diverse_sampling(video_dets, target_count=target)
    save_patches(selected, extract_save_path)

    detections += video_dets
    selected_detections += selected
# 50 to submit (do once we've got good results)

# %%
RELOAD_EXTRACT_PATH = SAVE_DIR + "/extracted_humans/20260224-213806"
# RELOAD_EXTRACT_PATH = extract_save_path

if RELOAD_EXTRACT_PATH:
    extract_save_path = RELOAD_EXTRACT_PATH
    patch_files = [
        f for f in os.listdir(extract_save_path)
        if f.endswith(('.jpg'))
    ]
    print(f"Reloaded {len(patch_files)} patches from {extract_save_path}")
else:
    print(f"Using freshly-extracted patches from {extract_save_path}")

# %%
# 1.2. Classification
cls_input_path = extract_save_path
cls_save_path = f"{SAVE_DIR}/classifications/{SAVE_NAME}"
classify_b_size = 32

pose_model = YOLO(os.path.join(PROJECT_ROOT, 'models/yolo26m-pose.pt'))
pose_model.to(DEVICE)

results, summary = classify_directory(
    pose_model,
    input_dir=cls_input_path,
    output_dir=cls_save_path,
    batch_size=classify_b_size,
    copy_files=True, # copy files to classifications dir for easy reference next to pose results
)

print(summary)


# %%
