# %%
# blabla

# %%
import os
import sys
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    # install deps from pyproject.toml
    url = "https://raw.githubusercontent.com/Theosdoor/ACV_cswk/refs/heads/main/pyproject.toml?token=GHSAT0AAAAAADMXCD2D27EMDM3ERVBN5GE62MOA47A"
    !wget -O pyproject.toml {url}
    !uv pip install --system -r pyproject.toml

import time
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO # https://github.com/ultralytics/ultralytics

from src.feat_extract import *

DATA_DIR = "downloaded_data" # name of dir where downloaded videos are
TRAIN_PATHS = [
    os.path.join(DATA_DIR, "Train/game/MafiaVideogame.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheGodfather.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheIrishman.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheSopranos.mp4"),
]

TEST_PATH = os.path.join(DATA_DIR,"Test/Test.mp4")



# %%
if __name__ == "__main__":
    video_path = "downloaded_data/Train/movie/TheGodfather.mp4"
    save_path = f"output/extracted_humans/{time.strftime('%Y%m%d-%H%M%S')}"

    model = YOLO('models/yolov8n.pt')
    n2save = 1000

    detections = extract_humans_from_video(model, video_path)

    # Score all detections
    for det in detections:
        det['score'] = score_detection(det)
    detections_sorted = sorted(detections, key=lambda x: x['score'], reverse=True)

    selected_detections = diverse_sampling(detections_sorted, target_count=n2save)

    save_patches(selected_detections, save_path) 
    # 50 to submit (do once we've got good results)


# %%


