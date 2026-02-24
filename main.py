# %%
# blabla

# %%
import os
import sys
import subprocess

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

DATA_DIR = "downloaded_data" # name of dir where downloaded videos are
TRAIN_PATHS = [
    os.path.join(DATA_DIR, "Train/game/MafiaVideogame.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheGodfather.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheIrishman.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheSopranos.mp4"),
]

TEST_PATH = os.path.join(DATA_DIR,"Test/Test.mp4")
SAVE_DIR = "output"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %%
# 1.1. Human Patch Extraction
extract_save_path = f"{SAVE_DIR}/extracted_humans/{time.strftime('%Y%m%d-%H%M%S')}"
n2save = 1000

# %% 
model = YOLO('models/yolov8m.pt')
model.to(DEVICE)

detections = []
for path in tqdm(TRAIN_PATHS, desc="Extracting from training videos", unit="video"):
    detections += extract_humans_from_video(model, path, yolo_batch_size=8)

# Score all detections
for det in tqdm(detections, desc="Scoring detections", unit="det"):
    det['score'] = score_detection(det)
detections_sorted = sorted(detections, key=lambda x: x['score'], reverse=True)

selected_detections = diverse_sampling(detections_sorted, target_count=n2save)

save_patches(selected_detections, extract_save_path)
# 50 to submit (do once we've got good results)

# %%
# DEBUG
# Diagnostic: check extraction yield per video
print(f"Total raw detections: {len(detections)}")
print(f"After scoring + sorting: {len(detections_sorted)}")
print(f"After diverse_sampling: {len(selected_detections)}")
print()

from collections import Counter
source_counts = Counter(os.path.basename(d['video_path']) for d in detections)
print("Raw detections per video:")
for src, count in source_counts.most_common():
    print(f"  {src}: {count}")

print()
score_vals = [d['score'] for d in detections]
if score_vals:
    print(f"Score range: {min(score_vals):.3f} – {max(score_vals):.3f}")
    print(f"Score mean:  {sum(score_vals)/len(score_vals):.3f}")

# %%
RELOAD_DIR_NAME = "20260222-130433"
RELOAD_EXTRACT_PATH = f"{SAVE_DIR}/extracted_humans/{RELOAD_DIR_NAME}"
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
cls_save_path = extract_save_path.replace("extracted_humans", "classifications")

pose_model = YOLO('models/yolo26m-pose.pt')
pose_model.to(DEVICE)

results, summary = classify_directory(
    pose_model,
    input_dir=cls_input_path,
    output_dir=cls_save_path,
    batch_size=32,
    copy_files=True,
)

print(summary)


# %%
