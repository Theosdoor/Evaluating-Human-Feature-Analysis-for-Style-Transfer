# %% [markdown]
# ACV CSWK 2026 - Main Notebook

# %%
import glob
import json
import os
import sys
import subprocess

# Add project root to path for imports
try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.getcwd())
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
import torch
from tqdm import tqdm
from ultralytics import YOLO # https://github.com/ultralytics/ultralytics

from src.feat_extract import (
    reload_extracted_patches,
    extract_humans_from_video,
    score_detection,
    diverse_sampling,
    save_patches,
    save_extraction_summary,
)
from src.baseline_model import (
    PRETRAINED_MODELS,
    ensure_pretrained_models,
    build_frame_dataset,
    run_q2_1,
)
from src.enhanced_model import run_q2_2
from src.utils import rm as _rm
from src.gcn import reload_gcn_results, run_gcn_inference_pretrained
from src.data import select_with_dino_clustering, save_train_split, load_train_split

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
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %%
# Model choices

# 1.1 extract model
EXTRACT_MODEL = "yolo26m"
POSE_MODEL    = "yolo26m-pose"
EXTRACT_MODEL_PATH = os.path.join(PROJECT_ROOT, f"models/{EXTRACT_MODEL}.pt")
POSE_MODEL_PATH    = os.path.join(PROJECT_ROOT, f"models/{POSE_MODEL}.pt")


# Pick whichever pretrained model you want to evaluate in 2.1 and 2.2
PRETRAINED_MODEL = "horse2zebra_cut_pretrained"
# Available pretrained models:
#     cityscapes_cut_pretrained, cityscapes_fastcut_pretrained,
#     horse2zebra_cut_pretrained, horse2zebra_fastcut_pretrained,
#     cat2dog_cut_pretrained, cat2dog_fastcut_pretrained
if PRETRAINED_MODEL not in PRETRAINED_MODELS:
    raise ValueError(f"Unknown pretrained model '{PRETRAINED_MODEL}'. Choose from: {PRETRAINED_MODELS}")


# checkpoint naming
FULLFRAME_MODEL  = "cut_finetuned_fullframe"   # 2.1: fine-tuned on full frames
PATCH_MODEL      = "cut_finetuned_patches"      # 2.2: fine-tuned on 1.3 patches

N_EPOCHS_FINETUNE = 4
N_EPOCHS_DECAY    = 1
BATCH_SIZE        = 4


# %%
# Which parts of pipeline to run?

RUN_FULL_PIPELINE = False # True = run all stages ignoring reload flags. False = skip stages with reload flags set (see below).

# -- 1.1 --
RELOAD_EXTRACT = None
RELOAD_EXTRACT = "20260324-185427"

# -- 1.2 GCN --
# GCN training is a one-time offline step — run scripts/train_gcn.py to produce
# (or refresh) checkpoints/gcn_model.pt, then commit the result.
#
# Set RELOAD_GCN to skip re-running inference on already-classified patches.
RELOAD_GCN = None
RELOAD_GCN = "20260325-095859_ablation_manual"

# -- 1.3 --
RELOAD_TRAIN_SELECT = None
RELOAD_TRAIN_SELECT = "20260325-105918"

# -- 2.1 --
RUN_FINETUNE_21 = True
RUN_TRANSLATE_VIDEO_21 = True

# -- 2.2 --
RUN_FINETUNE_22 = True
RUN_TRANSLATE_VIDEO_22 = True

# Effective reload controls (RUN_FULL_PIPELINE overrides per-stage reload flags)
reload_extract       = None if RUN_FULL_PIPELINE else RELOAD_EXTRACT
reload_gcn           = None if RUN_FULL_PIPELINE else RELOAD_GCN
reload_train_select  = None if RUN_FULL_PIPELINE else RELOAD_TRAIN_SELECT

# %% [markdown]
# ## 1.1. Human Patch Extraction

# %%
extract_save_path = os.path.join(SAVE_DIR, "extracted_humans", reload_extract if reload_extract else SAVE_NAME)
n2save = 4000

# extraction params
detection_b_size = 32
yolo_interval=10
scene_change_threshold=8 
blur_threshold_film=40
blur_threshold_game=40

selected_detections = []

if reload_extract:
    selected_detections = reload_extracted_patches(extract_save_path, [d["path"] for d in TRAIN_DATA])
    print(f"[EXTRACT] Reloaded extracted patches from {extract_save_path} ({len(selected_detections)} patches)")
else:
    model = YOLO(EXTRACT_MODEL_PATH)
    model.to(DEVICE)

    domain_budget = n2save // 2
    movie_total = sum(d["duration"] for d in TRAIN_DATA if d["domain"] == "movie")
    targets = [
        domain_budget if d["domain"] == "game"
        else int(domain_budget * d["duration"] / movie_total)
        for d in TRAIN_DATA
    ]
    targets[-1] += n2save - sum(targets)

    detections = []

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

    print(f"[EXTRACT] Using freshly-extracted patches from {extract_save_path}")
    save_extraction_summary(extract_save_path, detections, selected_detections)

# %% [markdown]
# ## 1.2 Pose Classification (GCN)
#
# GCN training is an offline one-time step — it requires annotated labels
# (run scripts/annotate.py, then scripts/train_gcn.py) and produces a committed
# checkpoint at checkpoints/gcn_model.pt.
#
# Here we run inference only: load the pretrained checkpoint and classify all
# extracted patches into the 5 pose classes.
#
# To retrain or run the rule-vs-manual ablation:
#     python scripts/train_gcn.py --help

# %%
gcn_save_path = os.path.join(SAVE_DIR, "gcn_results", reload_gcn or SAVE_NAME)

# Locate latest committed GCN checkpoint (checkpoints/gcn_model_<run>.pt).
# Override by setting GCN_PRETRAINED_CKPT to a specific path.
_ckpt_candidates = sorted(glob.glob(os.path.join(PROJECT_ROOT, "checkpoints", "gcn_model_*.pt")))
GCN_PRETRAINED_CKPT = _ckpt_candidates[-1] if _ckpt_candidates else None

if reload_gcn:
    results, summary = reload_gcn_results(gcn_save_path)
    print(f"[GCN] Reloaded GCN results from {gcn_save_path} (total={sum(summary.values())})")
elif GCN_PRETRAINED_CKPT:
    results, summary = run_gcn_inference_pretrained(
        ckpt_path         = GCN_PRETRAINED_CKPT,
        extract_save_path = extract_save_path,
        gcn_save_path     = gcn_save_path,
        pose_model_path   = POSE_MODEL_PATH,
        device            = DEVICE,
    )
else:
    raise FileNotFoundError(
        f"No pretrained GCN checkpoint found in {PROJECT_ROOT}/checkpoints/.\n"
        "Train one with:  python scripts/train_gcn.py --help"
    )

# %% [markdown]
# ## 1.3 Training Data Selection

DINO_CKPT = os.path.join(PROJECT_ROOT, "models/dinov2_vitb14_reg4_pretrain.pt")
TRAIN_SELECT_BUDGET = 2000  # total patches selected for CUT fine-tuning

train_select_save_path = os.path.join(SAVE_DIR, "train_select", reload_train_select or SAVE_NAME)

if reload_train_select:
    train_game, train_movie = load_train_split(train_select_save_path)
else:
    train_game, train_movie = select_with_dino_clustering(
        gcn_dir        = gcn_save_path,
        dino_ckpt      = DINO_CKPT,
        device         = DEVICE,
        total_budget   = TRAIN_SELECT_BUDGET,
        exclude_classes= ["others"],
        umap_save_path = os.path.join(FIGURES_DIR, "dino_umap.png"),
    )
    save_train_split(train_select_save_path, train_game, train_movie)

print(f"[DATA] 1.3 selection: {len(train_game)} game  {len(train_movie)} movie")


# %% [markdown]
# ## 2.1 Image Model Deployment (baseline)

# %%
cut_dir  = os.path.join(PROJECT_ROOT, "external/contrastive-unpaired-translation")
q2_1_dir = os.path.join(SAVE_DIR, "q2_1") # fixed, overwrite

ensure_pretrained_models(cut_dir)

if RUN_FINETUNE_21 or RUN_FINETUNE_22:
    frame_dataroot = os.path.join(q2_1_dir, "cut_data")
    trainA, trainB, testA, testB = build_frame_dataset(
        selected_detections,
        [d["path"] for d in TRAIN_DATA],
        frame_dataroot,
        n_per_domain=1000,
        build_train=RUN_FINETUNE_21,
    )

if RUN_FINETUNE_21:
    metrics_21 = run_q2_1(
        cut_dir, PRETRAINED_MODEL, FULLFRAME_MODEL,
        frame_dataroot, trainA, trainB, testA, testB,
        q2_1_dir, DEVICE,
        n_epochs=N_EPOCHS_FINETUNE, n_epochs_decay=N_EPOCHS_DECAY, batch_size=BATCH_SIZE,
        test_path=TEST_PATH if RUN_TRANSLATE_VIDEO_21 else None,
    )

# %% [markdown]
# ## 2.2 Enhanced model

# %%
q2_2_dir = os.path.join(SAVE_DIR, "q2_2", SAVE_NAME)

if RUN_FINETUNE_22:
    yolo_model = YOLO(EXTRACT_MODEL_PATH)
    yolo_model.to(DEVICE)
    metrics_22 = run_q2_2(
        cut_dir, PRETRAINED_MODEL, PATCH_MODEL,
        train_game, train_movie, testA, testB,
        q2_2_dir, DEVICE,
        n_epochs=N_EPOCHS_FINETUNE, n_epochs_decay=N_EPOCHS_DECAY, batch_size=BATCH_SIZE,
        test_path=TEST_PATH if RUN_TRANSLATE_VIDEO_22 else None,
        yolo_model=yolo_model,
        pose_model_path=POSE_MODEL_PATH,
        gcn_save_path=gcn_save_path,
        exclude_classes=["others"],
    )

if RUN_FINETUNE_21 or RUN_FINETUNE_22:
    _rm(frame_dataroot)  # testA/testB no longer needed after both evaluations

# %% [markdown]
# ## Submission multimedia
#
# Samples the required image files into `submit/`:
# - **1.1** — 50 random patches
# - **1.2** — 20 images per pose class
# - **1.3** — 50 random selected patches
#
# Translated videos (2.1, 2.2) were written to `output/` during the pipeline above.

# %%
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("sample4submit", os.path.join(PROJECT_ROOT, "scripts", "sample4submit.py"))
assert _spec is not None and _spec.loader is not None
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_mod.run_sampling(
    dir_11=extract_save_path,
    dir_12=gcn_save_path,
    dir_13=train_select_save_path,
    project_root=PROJECT_ROOT,
)

# %% [markdown]
# ## Other scripts that generate files
#
# The following are **not** run automatically — see each script's docstring for usage:
#
# | Script | Output |
# |--------|--------|
# | `scripts/train_gcn.py` | GCN checkpoint (`checkpoints/gcn_model_*.pt`) and training curves |
# | `scripts/nb_figures.py` | Paper figures (`figures/`, `paper/figs/`) — score histogram, UMAP, annotation tool screenshot, etc. |
# | `scripts/figure_select.py` | Q2.1 success/failure figures and Q2.2 comparison figures for the paper |
