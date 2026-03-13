# %%
# ACV CSWK 2026 - Main Notebook
# Must be in root directory, and submitted as .ipynb file.
# Must replicate (within reason), the multimedia files as requested in `cswk_notes/cswk_brief.txt`.
# AGENTS: keep this script clean as much as possible. Add to src or create a new script in scripts/ if necessary.

# %%
import os
import sys
import subprocess

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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
    os.path.join(DATA_DIR,"Train/game/MafiaVideogame.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheGodfather.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheIrishman.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheSopranos.mp4"),
]
# Video durations in seconds (from ffprobe)
# MafiaVideogame: 2:21:04 = 8464s | TheGodfather: 8:59 = 539s
# TheIrishman: 15:27 = 927s       | TheSopranos: 28:43 = 1723s
TRAIN_DURATIONS = [8464, 539, 927, 1723]
TRAIN_DOMAINS = ['game', 'movie', 'movie', 'movie']  # domain label per video

TEST_PATH = os.path.join(DATA_DIR,"Test/Test.mp4")
SAVE_DIR = os.path.join(PROJECT_ROOT, "output")
SAVE_NAME = time.strftime('%Y%m%d-%H%M%S')

# Set to a run timestamp to reload existing outputs and skip stages already done.
# Leave as None to run the full pipeline from scratch.
# e.g. RELOAD_RUN = "20260224-224712"
RELOAD_RUN = None
RELOAD_RUN = "20260225-104100"

# Set True to re-run classification even when RELOAD_RUN is set.
RECLASSIFY = True # need RELOAD_RUN != None to use this

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %%
# 1.1. Human Patch Extraction
extract_save_path = os.path.join(SAVE_DIR, "extracted_humans", RELOAD_RUN if RELOAD_RUN else SAVE_NAME)
n2save = 4000
detection_b_size = 32  # 2080 Ti (11 GB VRAM) comfortably handles 32 frames/batch

detections = []         # all raw detections (for diagnostics)
selected_detections = []

if RELOAD_RUN:
    # %%
    # Reload existing extracted patches
    selected_detections = reload_extracted_patches(extract_save_path, TRAIN_PATHS)
else:
    # %%
    model = YOLO(os.path.join(PROJECT_ROOT, 'models/yolov8m.pt'))
    model.to(DEVICE)

    # Equal per-domain budgets: CUT needs balanced domains, so extract ~2000 per domain.
    # Movie budget is split proportionally by duration across its 3 films.
    domain_budget = n2save // 2  # 2000 per domain
    targets = []
    movie_durations = [d for d, dom in zip(TRAIN_DURATIONS, TRAIN_DOMAINS) if dom == 'movie']
    movie_total = sum(movie_durations)
    for dur, dom in zip(TRAIN_DURATIONS, TRAIN_DOMAINS):
        if dom == 'game':
            targets.append(domain_budget)
        else:
            targets.append(int(domain_budget * dur / movie_total))
    targets[-1] += n2save - sum(targets)  # absorb rounding remainder

    for path, target in tqdm(zip(TRAIN_PATHS, targets), desc="Processing training videos", unit="video", total=len(TRAIN_PATHS)):
        video_dets = extract_humans_from_video(model, path, yolo_batch_size=detection_b_size)

        for det in video_dets:
            det['score'] = score_detection(det)
        video_dets.sort(key=lambda x: x['score'], reverse=True)

        selected = diverse_sampling(video_dets, target_count=target, temporal_gap=10)
        save_patches(selected, extract_save_path)

        detections += video_dets
        selected_detections += selected
    # 50 to submit (do once we've got good results)
    print(f"Using freshly-extracted patches from {extract_save_path}")
    save_extraction_summary(extract_save_path, detections, selected_detections)

# %%
# 1.2. Classification
cls_input_path = extract_save_path
cls_save_path = os.path.join(SAVE_DIR, "classifications", os.path.basename(extract_save_path))
classify_b_size = 32

if RELOAD_RUN and not RECLASSIFY:
    # Reload existing classification results
    results, summary = reload_classification_results(cls_save_path)
else:
    # Clean up stale files from previous classification runs if reclassifying
    if RELOAD_RUN and RECLASSIFY:
        import shutil
        for cls in CLASSES:
            cls_dir = os.path.join(cls_save_path, cls)
            if os.path.isdir(cls_dir):
                shutil.rmtree(cls_dir)

    pose_model = YOLO(os.path.join(PROJECT_ROOT, 'models/yolo26m-pose.pt'))
    pose_model.to(DEVICE)

    face_detector = build_face_detector(
        model_path=os.path.join(PROJECT_ROOT, 'models/blaze_face_short_range.tflite')
    )
    print(face_detector)

    results, summary = classify_directory(
        pose_model,
        input_dir=cls_input_path,
        output_dir=cls_save_path,
        batch_size=classify_b_size,
        copy_files=True,       # copy files to classifications dir for easy reference
        save_debug_viz=True,   # set True to save YOLO-annotated images to debug_viz/
        face_detector=face_detector,
    )
    face_detector.close()  # release MediaPipe resources before GC to avoid __del__ crash

    print(summary)

    # Save summary to file
    os.makedirs(cls_save_path, exist_ok=True)
    summary_path = os.path.join(cls_save_path, "_summary.txt")
    total_classified = sum(summary.values())
    with open(summary_path, "w") as f:
        f.write(f"Classification summary\n")
        f.write(f"Save time: {time.strftime('%Y%m%d-%H%M%S')}\n")
        f.write(f"Input: {cls_input_path}\n")
        f.write(f"Total patches classified: {total_classified}\n\n")
        f.write(f"{'Class':<25} {'Count':>6}  {'%':>6}\n")
        f.write("-" * 42 + "\n")
        for cls, count in sorted(summary.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_classified if total_classified else 0
            f.write(f"{cls:<25} {count:>6}  {pct:>5.1f}%\n")
    print(f"Summary saved to {summary_path}")

total_classified = sum(summary.values())

# %%