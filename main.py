# %%
# blabla

# %%
import os
import sys

# IN_COLAB = "google.colab" in sys.modules
# if IN_COLAB:
#     # install deps from pyproject.toml
#     url = "https://raw.githubusercontent.com/Theosdoor/ACV_cswk/refs/heads/main/pyproject.toml?token=GHSAT0AAAAAADMXCD2D27EMDM3ERVBN5GE62MOA47A"
#     !wget -O pyproject.toml {url}
#     !uv pip install --system -r pyproject.toml

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %%
# 1.1. Human Patch Extraction
# if __name__ == "__main__":
extract_save_path = f"{SAVE_DIR}/extracted_humans/{time.strftime('%Y%m%d-%H%M%S')}"

# .to(DEVICE) ensures the model is on GPU when running under Slurm.
# Batched YOLO calls (yolo_batch_size) are the main GPU throughput win.
model = YOLO('models/yolov8m.pt')
model.to(DEVICE)
n2save = 1000

# Sequential: Slurm allocates a single GPU, so parallel threads would just
# serialise at the CUDA level and thrash memory. Batching inside each video
# call is where the real speedup comes from.
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
# 1.2. Classification
cls_input_path = extract_save_path
cls_save_path = f"{SAVE_DIR}/classifications/{time.strftime('%Y%m%d-%H%M%S')}"

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
