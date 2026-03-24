# %%
# ACV CSWK 2026 - Main Notebook

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
    evaluate_translation,
    translate_test_video,
)
from src.enhanced_model import (
    finetune_cut_patches,
    translate_test_video_enhanced,
    compute_bbox_fid,
)
from src.utils import finetune_cut
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

N_EPOCHS_FINETUNE = 5
N_EPOCHS_DECAY    = 3

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
RELOAD_GCN = "20260324-195802_manual"

# -- 1.3 --
RELOAD_TRAIN_SELECT = None
RELOAD_TRAIN_SELECT = "20260324-195936"

# -- 2.1 --
RUN_FINETUNE_21 = False
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
# GCN training is an offline one-time step — it requires manually annotated labels
# (run scripts/annotate.py, then scripts/train_gcn.py) and produces a committed
# checkpoint at checkpoints/gcn_model.pt.
#
# Here we run inference only: load the pretrained checkpoint and classify all
# extracted patches into the 5 pose classes.
#
# To retrain or run the rule-vs-manual ablation:
#     python scripts/train_gcn.py --help

# %%
import glob as _glob

gcn_save_path = os.path.join(SAVE_DIR, "gcn_results", reload_gcn or SAVE_NAME)

# Locate latest committed GCN checkpoint (checkpoints/gcn_model_<run>.pt).
# Override by setting GCN_PRETRAINED_CKPT to a specific path.
_ckpt_candidates = sorted(_glob.glob(os.path.join(PROJECT_ROOT, "checkpoints", "gcn_model_*.pt")))
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

# %%
# 1.3 — DINOv2 cluster-then-select
# Embed patches with DINOv2 ViT-B/14, cluster per (class × domain) group,
# select the centroid-nearest patch from each cluster.
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
#
# Pipeline:
#   1. Fine-tune pretrained CUT on full game/movie frames from 1.1 detection timestamps
#   2. Apply to test video frame-by-frame
#   3. Evaluate with FID, KID, LPIPS in both translation directions
 
# %%
cut_dir  = os.path.join(PROJECT_ROOT, "external/contrastive-unpaired-translation")
q2_1_dir = os.path.join(SAVE_DIR, "q2_1", SAVE_NAME)

ensure_pretrained_models(cut_dir)

# Build full-frame dataset — always run (testA/testB also needed by 2.2 evaluate)
frame_dataroot = os.path.join(q2_1_dir, "cut_data")
trainA, trainB, testA, testB = build_frame_dataset(
    selected_detections,
    [d["path"] for d in TRAIN_DATA],
    frame_dataroot,
    n_per_domain=500,
)

if RUN_FINETUNE_21:
    # Fine-tune from pretrained weights — produces FULLFRAME_MODEL checkpoint
    # (copies PRETRAINED_MODEL first, so original weights are never modified)
    finetune_cut(
        cut_dir, PRETRAINED_MODEL, FULLFRAME_MODEL, frame_dataroot, DEVICE,
        n_epochs=N_EPOCHS_FINETUNE, n_epochs_decay=N_EPOCHS_DECAY,
    )

    # Evaluate in both directions
    metrics_2_1 = evaluate_translation(cut_dir, FULLFRAME_MODEL, testA, testB, q2_1_dir, DEVICE, tag="2.1")

    if RUN_TRANSLATE_VIDEO_21:
        baseline_video = translate_test_video(cut_dir, FULLFRAME_MODEL, TEST_PATH, q2_1_dir, DEVICE)
        print(f"[2.1] Baseline video → {baseline_video}")

# %% [markdown]
# ## 2.2 Enhanced model
# 
# 1. use selected patch data from 1.3
# 2. keep the same pretrained CUT model as 2.1 (no retraining)
# 3. use temporal enhancement (deferred for now)
# 
# Pipeline:
#   1. Fine-tune from the same pretrained weights as 2.1 (NOT from 2.1's checkpoint),
#      but on 1.3-selected human patches — tighter domain match to the human regions
#      we actually care about translating
#   2. For each test frame: detect humans (1.1), crop, translate with PATCH_MODEL,
#      composite back onto the original frame (background untouched)
#   3. EMA temporal blending between consecutive frames reduces flicker
#   4. Evaluate with same metrics as 2.1 for a fair comparison

# %%
q2_2_dir = os.path.join(SAVE_DIR, "q2_2", SAVE_NAME)

if RUN_FINETUNE_22:
    # YOLO for human detection in the test video (reuse same weights as 1.1)
    yolo_model = YOLO(EXTRACT_MODEL_PATH)
    yolo_model.to(DEVICE)

    # Fine-tune from the same pretrained base as 2.1 — produces PATCH_MODEL checkpoint
    finetune_cut_patches(
        cut_dir, PRETRAINED_MODEL, PATCH_MODEL,
        train_game, train_movie,
        q2_2_dir, DEVICE,
        n_epochs=N_EPOCHS_FINETUNE, n_epochs_decay=N_EPOCHS_DECAY,
    )

    # Evaluate patch model in both directions using same test splits as 2.1
    metrics_2_2 = evaluate_translation(cut_dir, PATCH_MODEL, testA, testB, q2_2_dir, DEVICE, tag="2.2")

    # Translate test video with patch-level compositing + GCN filtering + temporal blending
    if RUN_TRANSLATE_VIDEO_22:
        enhanced_video = translate_test_video_enhanced(
            cut_dir, PATCH_MODEL, TEST_PATH, q2_2_dir, DEVICE,
            yolo_model=yolo_model,
            pose_model_path=POSE_MODEL_PATH,
            gcn_save_path=gcn_save_path,
            exclude_classes=["others"],   # "no_pose" kept — YOLO confirmed human
            blend_alpha=0.3,
            blur_threshold=10.0,
        )
        print(f"[2.2] Enhanced video → {enhanced_video}")

        bbox_fid_22 = compute_bbox_fid(
            real_patch_dir=os.path.join(q2_2_dir, "patch_dataroot", "trainB"),
            translated_crop_dir=os.path.join(q2_2_dir, "crops_translated"),
            device=DEVICE,
        )
        print(f"[2.2] Bbox FID: {bbox_fid_22:.4f}")
 