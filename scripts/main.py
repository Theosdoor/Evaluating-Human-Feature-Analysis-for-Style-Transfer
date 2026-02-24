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
RELOAD_EXTRACT_PATH = SAVE_DIR + "/extracted_humans/" + "20260224-221543"
# RELOAD_EXTRACT_PATH = extract_save_path

if RELOAD_EXTRACT_PATH:
    import re
    extract_save_path = RELOAD_EXTRACT_PATH

    # Map source tag back to full video path
    source_tag_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in TRAIN_PATHS}

    patch_files = [f for f in os.listdir(extract_save_path) if f.endswith('.jpg')]
    _pat = re.compile(r'^human_\d+_(.+)_f(\d+)_conf([\d.]+)_score([\d.]+)\.jpg$')
    for fname in patch_files:
        m = _pat.match(fname)
        if m:
            source_tag, frame_num, conf, score = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
            video_path = source_tag_to_path.get(source_tag, source_tag)
            selected_detections.append({
                'video_path': video_path,
                'frame_num': frame_num,
                'confidence': conf,
                'score': score,
            })

    print(f"Reloaded {len(patch_files)} patches from {extract_save_path}")
    print(f"  Parsed {len(selected_detections)} detections from filenames")
else:
    print(f"Using freshly-extracted patches from {extract_save_path}")

# %%
# 1.2. Classification
cls_input_path = extract_save_path
cls_save_path = f"{SAVE_DIR}/classifications/{SAVE_NAME}"
classify_b_size = 32

pose_model = YOLO(os.path.join(PROJECT_ROOT, 'models/yolo26l-pose.pt'))
pose_model.to(DEVICE)

results, summary = classify_directory(
    pose_model,
    input_dir=cls_input_path,
    output_dir=cls_save_path,
    batch_size=classify_b_size,
    copy_files=True, # copy files to classifications dir for easy reference next to pose results
)

print(summary)

# Save summary to file
os.makedirs(cls_save_path, exist_ok=True)
summary_path = os.path.join(cls_save_path, "summary.txt")
total_classified = sum(summary.values())
with open(summary_path, "w") as f:
    f.write(f"Classification summary\n")
    f.write(f"Run: {SAVE_NAME}\n")
    f.write(f"Input: {cls_input_path}\n")
    f.write(f"Total patches classified: {total_classified}\n\n")
    f.write(f"{'Class':<25} {'Count':>6}  {'%':>6}\n")
    f.write("-" * 42 + "\n")
    for cls, count in sorted(summary.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_classified if total_classified else 0
        f.write(f"{cls:<25} {count:>6}  {pct:>5.1f}%\n")
print(f"Summary saved to {summary_path}")

# %%
# Diagnostics
from collections import Counter

print("\n=== Extraction Diagnostics ===")
if len(detections) > 0:
    print(f"Videos processed:        {len(TRAIN_PATHS)}")
    print(f"Total raw detections:    {len(detections)}")
    print(f"After diverse_sampling:  {len(selected_detections)}")

    source_counts = Counter(os.path.basename(d['video_path']) for d in detections)
    selected_counts = Counter(os.path.basename(d['video_path']) for d in selected_detections)
    print(f"\n{'Video':<30} {'Raw':>6}  {'Selected':>8}")
    print("-" * 48)
    for src, raw in source_counts.most_common():
        sel = selected_counts.get(src, 0)
        print(f"{src:<30} {raw:>6}  {sel:>8}")

    score_vals = [d['score'] for d in detections]
    if score_vals:
        print(f"\nDetection score stats:")
        print(f"  min:  {min(score_vals):.3f}")
        print(f"  max:  {max(score_vals):.3f}")
        print(f"  mean: {sum(score_vals)/len(score_vals):.3f}")

    blur_vals = [d['blur_score'] for d in detections]
    if blur_vals:
        print(f"\nBlur score stats (higher = sharper):")
        print(f"  min:  {min(blur_vals):.1f}")
        print(f"  max:  {max(blur_vals):.1f}")
        print(f"  mean: {sum(blur_vals)/len(blur_vals):.1f}")
elif len(selected_detections) > 0:
    print("(Extraction skipped; stats from reloaded patch filenames)")
    print(f"Reloaded patches: {len(selected_detections)}")

    selected_counts = Counter(os.path.basename(d['video_path']) for d in selected_detections)
    print(f"\n{'Video':<30} {'Patches':>8}")
    print("-" * 40)
    for src, count in selected_counts.most_common():
        print(f"{src:<30} {count:>8}")

    score_vals = [d['score'] for d in selected_detections]
    if score_vals:
        print(f"\nScore stats (selected patches):")
        print(f"  min:  {min(score_vals):.3f}")
        print(f"  max:  {max(score_vals):.3f}")
        print(f"  mean: {sum(score_vals)/len(score_vals):.3f}")
else:
    print("(No extraction data available)")

print("\n=== Classification Diagnostics ===")
print(f"Total classified: {total_classified}")
print(f"\n{'Class':<25} {'Count':>6}  {'%':>6}")
print("-" * 42)
for cls, count in sorted(summary.items(), key=lambda x: -x[1]):
    pct = 100 * count / total_classified if total_classified else 0
    print(f"{cls:<25} {count:>6}  {pct:>5.1f}%")


# %%
