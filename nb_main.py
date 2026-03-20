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

# Install required external model/repositories
install_script = os.path.join(PROJECT_ROOT, "scripts/install_externals.sh")
subprocess.run([install_script], check=True)

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
TRAIN_DATA = [
    {
        "path": os.path.join(DATA_DIR, "Train/game/MafiaVideogame.mp4"),
        "duration": 8464,  # 2:21:04
        "domain": "game"
    },
    {
        "path": os.path.join(DATA_DIR, "Train/movie/TheGodfather.mp4"),
        "duration": 539,  # 8:59
        "domain": "movie"
    },
    {
        "path": os.path.join(DATA_DIR, "Train/movie/TheIrishman.mp4"),
        "duration": 927,  # 15:27
        "domain": "movie"
    },
    {
        "path": os.path.join(DATA_DIR, "Train/movie/TheSopranos.mp4"),
        "duration": 1723,  # 28:43
        "domain": "movie"
    },
]

TEST_PATH = os.path.join(DATA_DIR,"Test/Test.mp4")
SAVE_DIR = os.path.join(PROJECT_ROOT, "output")
SAVE_NAME = time.strftime('%Y%m%d-%H%M%S')

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# %%
# Which parts of pipeline to run?

RUN_FULL_PIPELINE = False # True = run all stages ignoring reload flags. False = skip stages with reload flags set (see below).

# -- 1.1 --
RELOAD_EXTRACT = None
RELOAD_EXTRACT = "20260314-195748"

# -- 1.2 rule-based --
# Points to a run name under output/init_classifications/.
# Set to None to run rule-based classification from scratch.
# RELOAD_INIT_CLS = None
RELOAD_INIT_CLS = "20260314-195748-6"

# -- 1.2b manual annotations --
# Path to annotations.json from scripts/annotate.py.
# Used as training labels for the GCN.
RELOAD_ANNOTATIONS = None
RELOAD_ANNOTATIONS = "20260314-195748-6"

# -- 1.2c GCN --
# "rule"   — train GCN on rule-based labels (RELOAD_INIT_CLS dir)
# "manual" — train GCN on manual annotations (RELOAD_ANNOTATIONS)
GCN_LABEL_SOURCE = "manual"

# Set to a run name under output/gcn_results/ to skip GCN training+inference.
RELOAD_GCN = None
# RELOAD_GCN = "20260314-195748"

# -- 1.3 --
RELOAD_TRAIN_SELECT = None

# -- 2.1 --
RUN_TRAIN_CUT = False
RUN_TRANSLATE_VIDEO = False

# Effective reload controls (RUN_FULL_PIPELINE overrides per-stage reload flags)
reload_extract   = None if RUN_FULL_PIPELINE else RELOAD_EXTRACT
reload_init_cls  = None if RUN_FULL_PIPELINE else RELOAD_INIT_CLS
reload_gcn       = None if RUN_FULL_PIPELINE else RELOAD_GCN

# %% [markdown]
# ## 1.1. Human Patch Extraction

# %%
extract_save_path = os.path.join(SAVE_DIR, "extracted_humans", reload_extract if reload_extract else SAVE_NAME)
n2save = 4000

# extraction params
detection_b_size = 32
yolo_interval=10
scene_change_threshold=8.0
blur_threshold_film=40.0
blur_threshold_game=100.0

detections = []
selected_detections = []

if reload_extract:
    selected_detections = reload_extracted_patches(extract_save_path, [d["path"] for d in TRAIN_DATA])
else:
    model = YOLO(os.path.join(PROJECT_ROOT, 'models/yolov8m.pt'))
    model.to(DEVICE)

    domain_budget = n2save // 2
    targets = []
    movie_durations = [d["duration"] for d in TRAIN_DATA if d["domain"] == 'movie']
    movie_total = sum(movie_durations)
    for data in TRAIN_DATA:
        if data["domain"] == 'game':
            targets.append(domain_budget)
        else:
            targets.append(int(domain_budget * data["duration"] / movie_total))
    targets[-1] += n2save - sum(targets)

    for data, target in tqdm(zip(TRAIN_DATA, targets), desc="Processing training videos", unit="video", total=len(TRAIN_DATA)):
        video_dets = extract_humans_from_video(model, data["path"],
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

    print(f"Using freshly-extracted patches from {extract_save_path}")
    save_extraction_summary(extract_save_path, detections, selected_detections)

# %% [markdown]
# ## 1.2. Rule-Based Classification

# %%
init_cls_base_dir = os.path.join(SAVE_DIR, "init_classifications")
if reload_init_cls:
    init_cls_save_path = os.path.join(init_cls_base_dir, reload_init_cls)
else:
    init_cls_save_path = get_next_reclassify_dir(init_cls_base_dir, os.path.basename(extract_save_path))

classify_b_size = 32

if reload_init_cls:
    init_results, init_summary = reload_classification_results(init_cls_save_path)
else:
    pose_model = YOLO(os.path.join(PROJECT_ROOT, 'models/yolo26m-pose.pt'))
    pose_model.to(DEVICE)

    init_results, init_summary = classify_directory(
        pose_model,
        input_dir=extract_save_path,      # patches live here
        output_dir=init_cls_save_path,
        batch_size=classify_b_size,
        copy_files=True,
        save_debug_viz=False,
        save_keypoints=True,              # writes _keypoints.npz to extract_save_path
    )

    total_classified = sum(init_summary.values())
    os.makedirs(init_cls_save_path, exist_ok=True)
    summary_path = os.path.join(init_cls_save_path, "_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Rule-based classification summary\n")
        f.write(f"Save time: {time.strftime('%Y%m%d-%H%M%S')}\n")
        f.write(f"Input: {extract_save_path}\n")
        f.write(f"Total patches classified: {total_classified}\n\n")
        f.write(f"{'Class':<25} {'Count':>6}  {'%':>6}\n")
        f.write("-" * 42 + "\n")
        for cls, count in sorted(init_summary.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_classified if total_classified else 0
            f.write(f"{cls:<25} {count:>6}  {pct:>5.1f}%\n")
    print(f"Rule-based summary saved to {summary_path}")

# %% [markdown]
# ## 1.2b. Manual Annotation (optional — run scripts/annotate.py separately)
#
# Run annotate.py against init_cls_save_path to produce an annotations.json:
#
#     python3 scripts/annotate.py --cls-dir output/init_classifications/<run>
#
# Then set RELOAD_ANNOTATIONS above to the resulting annotations.json path.

# %% [markdown]
# ## 1.2c. GCN Classification

# %%
from src.gcn import run_gcn_pipeline, reload_gcn_results

gcn_run_name  = reload_gcn if reload_gcn else SAVE_NAME
gcn_save_path = os.path.join(SAVE_DIR, "gcn_results", gcn_run_name)

if reload_gcn:
    results, summary = reload_gcn_results(gcn_save_path)
else:
    # Decide which labels to use for GCN training
    if GCN_LABEL_SOURCE == "manual":
        if not RELOAD_ANNOTATIONS:
            raise ValueError("GCN_LABEL_SOURCE='manual' but RELOAD_ANNOTATIONS is not set.")
        if os.sep in RELOAD_ANNOTATIONS or "/" in RELOAD_ANNOTATIONS:
            # treat as explicit path
            ann_path = (
                os.path.join(PROJECT_ROOT, RELOAD_ANNOTATIONS)
                if not os.path.isabs(RELOAD_ANNOTATIONS)
                else RELOAD_ANNOTATIONS
            )
        else:
            # treat as run name — look up under output/manual_annotated/
            ann_path = os.path.join(SAVE_DIR, "manual_annotated", RELOAD_ANNOTATIONS, "annotations.json")
        labelled_dir = os.path.dirname(ann_path)

    results, summary = run_gcn_pipeline(
        labelled_dir    = labelled_dir,
        cls_source      = GCN_LABEL_SOURCE,
        all_patches_dir = extract_save_path,
        save_dir        = gcn_save_path,
        pose_model_path = os.path.join(PROJECT_ROOT, "models/yolo26m-pose.pt"),
        device          = DEVICE,
        lr              = 3e-4,
        epochs          = 300,
        hidden          = 128,
        dropout         = 0.1,
        batch_size      = 128,
        exclude_classes   = ['others'],
    )

total_classified = sum(summary.values())

# %% [markdown]
# ## 1.3 Training Data Selection

# %%
from src.data import get_data_split, flat_paths, flat_paths_by_domain

split = get_data_split(
    gcn_save_path,                  # use GCN results as the classification source
    train_split=1.0,
    exclude_classes=['others']
)

train_game, train_movie = flat_paths_by_domain(split['train'])
val_game, val_movie     = flat_paths_by_domain(split['val'])


# %% [markdown]
# ## 2.1 Image Model Deployment (baseline model)

# %%
from src.baseline_model import build_frame_dataset, train_cut, translate_test_video

cut_dir = os.path.join(PROJECT_ROOT, "external/contrastive-unpaired-translation")
data_2_1 = os.path.join(SAVE_DIR, "cut_data")
exp_game2movie = "cut_game2movie"
exp_movie2game = "cut_movie2game"

n_frames_per_domain = 500
n_epochs            = 20
n_epochs_decay      = 5
batch_size          = 4

trainA, trainB, testA, testB = build_frame_dataset(
    selected_detections, [d["path"] for d in TRAIN_DATA], data_2_1, n_frames_per_domain
)

if RUN_TRAIN_CUT:
    train_cut(cut_dir, data_2_1, exp_game2movie, "AtoB", DEVICE, n_epochs, n_epochs_decay, batch_size)
    train_cut(cut_dir, data_2_1, exp_movie2game, "BtoA", DEVICE, n_epochs, n_epochs_decay, batch_size)
else:
    print("Skipping CUT training (RUN_TRAIN_CUT=False).")

if RUN_TRANSLATE_VIDEO:
    baseline_video = translate_test_video(cut_dir, exp_game2movie, TEST_PATH, SAVE_DIR, DEVICE)
    print(f"Baseline video → {baseline_video}")
else:
    baseline_video = None
    print("Skipping video translation (RUN_TRANSLATE_VIDEO=False).")

# %%
import glob, json
from src.baseline_model import run_inference, make_inference_dataroot, compute_metrics

results_dir = os.path.join(SAVE_DIR, "2_1_results")

def checkpoint_exists(cut_dir, exp_name):
    return os.path.exists(os.path.join(cut_dir, "checkpoints", exp_name, "latest_net_G.pth"))

g2m_ready = checkpoint_exists(cut_dir, exp_game2movie)
m2g_ready = checkpoint_exists(cut_dir, exp_movie2game)

if g2m_ready and m2g_ready:
    g2m_fakes = run_inference(cut_dir, exp_game2movie,
                              make_inference_dataroot(testA, testB),
                              os.path.join(results_dir, "g2m"), "AtoB", DEVICE)
    m2g_fakes = run_inference(cut_dir, exp_movie2game,
                              make_inference_dataroot(testB, testA),
                              os.path.join(results_dir, "m2g"), "AtoB", DEVICE)

    metrics = {
        "game→movie": compute_metrics(testB, os.path.join(results_dir, "g2m", "fake"),
                                      glob.glob(os.path.join(testA, "*.jpg")), g2m_fakes, DEVICE),
        "movie→game": compute_metrics(testA, os.path.join(results_dir, "m2g", "fake"),
                                      glob.glob(os.path.join(testB, "*.jpg")), m2g_fakes, DEVICE),
    }
    for direction, vals in metrics.items():
        print(f"{direction}:  " + "  ".join(f"{k}: {v:.4f}" for k, v in vals.items()))

    viz_dir = os.path.join(SAVE_DIR, "2_1_viz")
    os.makedirs(viz_dir, exist_ok=True)
    with open(os.path.join(viz_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    from src.baseline_model import save_comparison_grid, save_umap

    save_comparison_grid(glob.glob(os.path.join(testA, "*.jpg")), g2m_fakes,
                         "game → movie (CUT)", os.path.join(viz_dir, "comparison_game2movie.png"))
    save_comparison_grid(glob.glob(os.path.join(testB, "*.jpg")), m2g_fakes,
                         "movie → game (CUT)", os.path.join(viz_dir, "comparison_movie2game.png"))

    save_umap(
        [glob.glob(os.path.join(testA, "*.jpg")),
         glob.glob(os.path.join(testB, "*.jpg")),
         g2m_fakes],
        ["game (real)", "movie (real)", "game→movie (fake)"],
        ["steelblue", "tomato", "mediumpurple"],
        "VGG feature UMAP: game→movie",
        os.path.join(viz_dir, "umap_game2movie.png"),
        device=DEVICE,
    )
    save_umap(
        [glob.glob(os.path.join(testA, "*.jpg")),
         glob.glob(os.path.join(testB, "*.jpg")),
         m2g_fakes],
        ["game (real)", "movie (real)", "movie→game (fake)"],
        ["steelblue", "tomato", "seagreen"],
        "VGG feature UMAP: movie→game",
        os.path.join(viz_dir, "umap_movie2game.png"),
        device=DEVICE,
    )
else:
    print("CUT checkpoints not found — skipping inference. Set RUN_TRAIN_CUT=True and re-run.")


# %% [markdown]
# ## 2.2 Enhanced model

# %%