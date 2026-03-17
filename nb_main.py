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
from src.utils import *

DATA_DIR = os.path.join(PROJECT_ROOT, "downloaded_data") # name of dir where downloaded videos are
TRAIN_PATHS = [
    os.path.join(DATA_DIR,"Train/game/MafiaVideogame.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheGodfather.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheIrishman.mp4"),
    os.path.join(DATA_DIR,"Train/movie/TheSopranos.mp4"),
]
# Video durations in seconds
# MafiaVideogame: 2:21:04 = 8464s | TheGodfather: 8:59 = 539s
# TheIrishman: 15:27 = 927s       | TheSopranos: 28:43 = 1723s
TRAIN_DURATIONS = [8464, 539, 927, 1723]
TRAIN_DOMAINS = ['game', 'movie', 'movie', 'movie']  # domain label per video

TEST_PATH = os.path.join(DATA_DIR,"Test/Test.mp4")
SAVE_DIR = os.path.join(PROJECT_ROOT, "output")
SAVE_NAME = time.strftime('%Y%m%d-%H%M%S')

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# download external repos if necessary
if not os.path.exists(os.path.join(PROJECT_ROOT, "contrastive-unpaired-translation")):
    url = "https://github.com/Theosdoor/contrastive-unpaired-translation.git"
    subprocess.run(["git", "clone", url], check=True)

# -- 1.1 --
# Set to a run timestamp to reload existing outputs and skip stages already done.
# Leave as None to run the full pipeline from scratch.
# e.g. RELOAD_RUN = "20260224-224712"
RELOAD_RUN = None
RELOAD_RUN = "20260314-195748"

# -- 1.2 --
# Set True to re-run classification even when RELOAD_RUN is set.
RUN_CLASSIFICATION = False # need RELOAD_RUN != None to use this

# -- 1.3 --

# -- 2.1 --

# -- 2.2 --

# %%
# 1.1. Human Patch Extraction
extract_save_path = os.path.join(SAVE_DIR, "extracted_humans", RELOAD_RUN if RELOAD_RUN else SAVE_NAME)
n2save = 4000

# extraction params
detection_b_size = 32
yolo_interval=10
scene_change_threshold=8.0
blur_threshold_film=40.0
blur_threshold_game=100.0

detections = []         # all raw detections (for diagnostics)
selected_detections = []

if RELOAD_RUN:
    # Reload existing extracted patches
    selected_detections = reload_extracted_patches(extract_save_path, TRAIN_PATHS)
else:
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
        video_dets = extract_humans_from_video(model, path, 
                                                yolo_batch_size=detection_b_size,
                                                yolo_interval=yolo_interval,
                                                scene_change_threshold=scene_change_threshold,
                                                blur_threshold_film=blur_threshold_film,
                                                blur_threshold_game=blur_threshold_game
                                               )

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
cls_base_dir = os.path.join(SAVE_DIR, "classifications")
if RELOAD_RUN and RUN_CLASSIFICATION:
    cls_save_path = get_next_reclassify_dir(cls_base_dir, os.path.basename(extract_save_path))
else:
    cls_save_path = os.path.join(cls_base_dir, os.path.basename(extract_save_path))
classify_b_size = 32

if RELOAD_RUN and not RUN_CLASSIFICATION:
    # Reload existing classification results
    results, summary = reload_classification_results(cls_save_path)
else:
    pose_model = YOLO(os.path.join(PROJECT_ROOT, 'models/yolo26m-pose.pt'))
    pose_model.to(DEVICE)

    results, summary = classify_directory(
        pose_model,
        input_dir=cls_input_path,
        output_dir=cls_save_path,
        batch_size=classify_b_size,
        copy_files=True,       # copy files to classifications dir for easy reference
        save_debug_viz=True,   # set True to save YOLO-annotated images to debug_viz/
    )

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
# 1.3 Training Data Selection

# ... DONT TOUCH THIS



# %% [markdown]
# ## 2.1 Image Model Deployment (baseline model)

# %%
# 1. using contrastive-unpaired-translation, apply it to output/extracted_humans/20260314-195748 to convert images between game and movie domains.
# 2. analyse performance in both directions using appropriate metrics (e.g. FID, KID, LPIPS) and visualizations (e.g. UMAPs).
# 3. Transfer the style humans in downloaded_data/Test/Test.mp4 and save the video to output/baseline_model.

# %%
# Dataset preparation + CUT training
from src.baseline_model import build_frame_dataset, train_cut, translate_test_video

CUT_DIR        = os.path.join(PROJECT_ROOT, "contrastive-unpaired-translation")
DATA_2_1       = os.path.join(SAVE_DIR, "cut_data")
EXP_GAME2MOVIE = "cut_game2movie"
EXP_MOVIE2GAME = "cut_movie2game"

# Hyperparams — reduce epochs or batch size if time / VRAM constrained
N_FRAMES_PER_DOMAIN = 500
N_EPOCHS            = 20
N_EPOCHS_DECAY      = 5
BATCH_SIZE          = 4      # use 2 if VRAM < 8 GB

trainA, trainB, testA, testB = build_frame_dataset(
    selected_detections, TRAIN_PATHS, DATA_2_1, N_FRAMES_PER_DOMAIN
)

train_cut(CUT_DIR, DATA_2_1, EXP_GAME2MOVIE, "AtoB", DEVICE, N_EPOCHS, N_EPOCHS_DECAY, BATCH_SIZE)
train_cut(CUT_DIR, DATA_2_1, EXP_MOVIE2GAME, "BtoA", DEVICE, N_EPOCHS, N_EPOCHS_DECAY, BATCH_SIZE)

baseline_video = translate_test_video(CUT_DIR, EXP_GAME2MOVIE, TEST_PATH, SAVE_DIR, DEVICE)
print(f"Baseline video → {baseline_video}")

# %%
# Metrics (FID, KID, LPIPS)
import glob, json
from src.baseline_model import run_inference, make_inference_dataroot, compute_metrics

RESULTS_DIR = os.path.join(SAVE_DIR, "2_1_results")

g2m_fakes = run_inference(CUT_DIR, EXP_GAME2MOVIE,
                           make_inference_dataroot(testA, testB),
                           os.path.join(RESULTS_DIR, "g2m"), "AtoB", DEVICE)
m2g_fakes = run_inference(CUT_DIR, EXP_MOVIE2GAME,
                           make_inference_dataroot(testB, testA),
                           os.path.join(RESULTS_DIR, "m2g"), "AtoB", DEVICE)

metrics = {
    "game→movie": compute_metrics(testB, os.path.join(RESULTS_DIR, "g2m", "fake"),
                                  glob.glob(os.path.join(testA, "*.jpg")), g2m_fakes, DEVICE),
    "movie→game": compute_metrics(testA, os.path.join(RESULTS_DIR, "m2g", "fake"),
                                  glob.glob(os.path.join(testB, "*.jpg")), m2g_fakes, DEVICE),
}
for direction, vals in metrics.items():
    print(f"{direction}:  " + "  ".join(f"{k}: {v:.4f}" for k, v in vals.items()))

VIZ_DIR = os.path.join(SAVE_DIR, "2_1_viz")
os.makedirs(VIZ_DIR, exist_ok=True)
with open(os.path.join(VIZ_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# %%
# Visualisation — comparison grids + UMAPs
from src.baseline_model import save_comparison_grid, save_umap

save_comparison_grid(glob.glob(os.path.join(testA, "*.jpg")), g2m_fakes,
                     "game → movie (CUT)", os.path.join(VIZ_DIR, "comparison_game2movie.png"))
save_comparison_grid(glob.glob(os.path.join(testB, "*.jpg")), m2g_fakes,
                     "movie → game (CUT)", os.path.join(VIZ_DIR, "comparison_movie2game.png"))

save_umap(
    [glob.glob(os.path.join(testA, "*.jpg")),
     glob.glob(os.path.join(testB, "*.jpg")),
     g2m_fakes],
    ["game (real)", "movie (real)", "game→movie (fake)"],
    ["steelblue", "tomato", "mediumpurple"],
    "VGG feature UMAP: game→movie",
    os.path.join(VIZ_DIR, "umap_game2movie.png"),
    device=DEVICE,
)
save_umap(
    [glob.glob(os.path.join(testA, "*.jpg")),
     glob.glob(os.path.join(testB, "*.jpg")),
     m2g_fakes],
    ["game (real)", "movie (real)", "movie→game (fake)"],
    ["steelblue", "tomato", "seagreen"],
    "VGG feature UMAP: movie→game",
    os.path.join(VIZ_DIR, "umap_movie2game.png"),
    device=DEVICE,
)