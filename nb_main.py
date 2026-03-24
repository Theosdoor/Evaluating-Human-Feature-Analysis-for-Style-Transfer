# %%
# ACV CSWK 2026 - Main Notebook

# %%
import os
import sys
import subprocess
import shutil

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

import glob
import json
import time
import numpy as np
import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO # https://github.com/ultralytics/ultralytics

from src.feat_extract import *
from src.classification import *
from src.baseline_model import *
from src.enhanced_model import *
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
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %%
# Model choices

# 1.1 extract model
EXTRACT_MODEL = "yolov8m"
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
RELOAD_EXTRACT = "20260314-195748"

# -- 1.2 rule-based --
# Points to a run name under output/init_classifications/.
# Set to None to run rule-based classification from scratch.
RELOAD_INIT_CLS = None
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
RELOAD_GCN = "20260320-162455"
# Val accuracy per class for manual annotations:
    # full_body_front           52/72  (72.2%)
    # full_body_back            18/19  (94.7%)
    # head_shoulder_front       49/75  (65.3%)
    # head_shoulder_back        19/27  (70.4%)
    # others                    5/51  (9.8%)

# -- 1.3 --
# RELOAD_TRAIN_SELECT = None

# -- 2.1 --
RUN_FINETUNE_21 = False
RUN_TRANSLATE_VIDEO_21 = True

# -- 2.2 --
RUN_FINETUNE_22 = True
RUN_TRANSLATE_VIDEO_22 = True

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
# ## 1.2 Classification

# %%
# initially do it based on rules applied to keypoints
init_cls_base_dir = os.path.join(SAVE_DIR, "init_classifications")
if reload_init_cls:
    init_cls_save_path = os.path.join(init_cls_base_dir, reload_init_cls)
else:
    init_cls_save_path = get_next_reclassify_dir(init_cls_base_dir, os.path.basename(extract_save_path))

classify_b_size = 32

if reload_init_cls:
    init_results, init_summary = reload_classification_results(init_cls_save_path)
    print(f"[CLS] Reloaded rule-based classification from {init_cls_save_path} (total={sum(init_summary.values())})")
else:
    pose_model = YOLO(POSE_MODEL_PATH)
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
    print(f"[CLS] Rule-based summary saved to {summary_path}")

# %% [markdown]
# Manual Annotation (optional — run scripts/annotate.py separately)
#
# Run annotate.py against init_cls_save_path to produce an annotations.json:
#
#     python3 scripts/annotate.py --cls-dir output/init_classifications/<run>
#
# Then set RELOAD_ANNOTATIONS above to the resulting annotations.json path.


# %%
# GCN Classification
from src.gcn import run_gcn_pipeline, reload_gcn_results, plot_annotation_ablation

gcn_save_path = os.path.join(SAVE_DIR, "gcn_results", reload_gcn or SAVE_NAME)

gcn_params = dict(
    all_patches_dir = extract_save_path,
    pose_model_path = POSE_MODEL_PATH,
    device          = DEVICE,
    lr              = 3e-4,
    epochs          = 300,
    hidden          = 128,
    dropout         = 0.1,
    batch_size      = 256,
    # exclude_classes = ['others'],  # including 'others' is actually really important for generalising!!
    save_plots      = True,
)

if reload_gcn:
    results, summary = reload_gcn_results(gcn_save_path)
    print(f"[GCN] Reloaded GCN results from {gcn_save_path} (total={sum(summary.values())})")
else:
    if GCN_LABEL_SOURCE == "manual":
        if not RELOAD_ANNOTATIONS:
            raise ValueError("GCN_LABEL_SOURCE='manual' but RELOAD_ANNOTATIONS is not set.")
        if "/" in RELOAD_ANNOTATIONS:
            ann_path = RELOAD_ANNOTATIONS if os.path.isabs(RELOAD_ANNOTATIONS) else os.path.join(PROJECT_ROOT, RELOAD_ANNOTATIONS)
        else:
            ann_path = os.path.join(SAVE_DIR, "manual_annotated", RELOAD_ANNOTATIONS, "annotations.json")
        labelled_dir = os.path.dirname(ann_path)
    elif GCN_LABEL_SOURCE == "rule":
        labelled_dir = init_cls_save_path
    else:
        raise ValueError(f"Unknown GCN_LABEL_SOURCE '{GCN_LABEL_SOURCE}'. Use 'manual' or 'rule'.")

    results, summary, gcn_per_class_val_acc = run_gcn_pipeline(
        labelled_dir = labelled_dir,
        cls_source   = GCN_LABEL_SOURCE,
        save_dir     = gcn_save_path,
        **gcn_params,  # type: ignore[arg-type]
    )

# %%
# GCN Annotation Ablation (rule-based vs manual labels)
# Run both label sources and compare per-class val accuracy.
# Requires both RELOAD_INIT_CLS (rule labels) and RELOAD_ANNOTATIONS (manual labels).
RUN_GCN_ABLATION = False

if RUN_GCN_ABLATION:
    _rule_dir   = os.path.join(SAVE_DIR, "init_classifications", RELOAD_INIT_CLS)
    _manual_dir = os.path.join(SAVE_DIR, "manual_annotated", RELOAD_ANNOTATIONS)
    _, _, _rule_acc = run_gcn_pipeline(
        labelled_dir = _rule_dir,
        cls_source   = "rule",
        save_dir     = os.path.join(SAVE_DIR, "gcn_results", SAVE_NAME + "_ablation_rule"),
        **gcn_params,  # type: ignore[arg-type]
    )
    _, _, _manual_acc = run_gcn_pipeline(
        labelled_dir = _manual_dir,
        cls_source   = "manual",
        save_dir     = os.path.join(SAVE_DIR, "gcn_results", SAVE_NAME + "_ablation_manual"),
        **gcn_params,  # type: ignore[arg-type]
    )
    plot_annotation_ablation(
        rule_per_class_val_acc   = _rule_acc,
        manual_per_class_val_acc = _manual_acc,
        save_path = os.path.join(FIGURES_DIR, "gcn_annotation_ablation.png"),
    )

# %% [markdown]
# ## 1.3 Training Data Selection

# %%
# For now, just use all data from the gcn results directory
from src.data import get_data_split, flat_paths, flat_paths_by_domain

split = get_data_split(
    gcn_save_path,
    train_split=1.0,
    exclude_classes=['others']
)

train_game, train_movie = flat_paths_by_domain(split['train'])
val_game, val_movie     = flat_paths_by_domain(split['val'])


# %% [markdown]
# ## 2.1 Image Model Deployment (baseline)
#
# Pipeline:
#   1. Fine-tune pretrained CUT on full game/movie frames from 1.1 detection timestamps
#   2. Apply to test video frame-by-frame
#   3. Evaluate with FID, KID, LPIPS in both translation directions
 
# %%
cut_dir  = os.path.join(PROJECT_ROOT, "external/contrastive-unpaired-translation")
q2_1_dir = os.path.join(SAVE_DIR, "q2_1")

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
    finetune_cut_fullframe(
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
            gcn_hidden=128,
            exclude_classes=["others"],
            blend_alpha=0.3,
            blur_threshold=10.0,
        )
        print(f"[2.2] Enhanced video → {enhanced_video}")
 