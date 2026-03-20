# %%
# ACV CSWK 2026 - Main Notebook
# Must be in root directory, and submitted as .ipynb file.
# Must replicate (within reason), the multimedia files as requested in `cswk_notes/cswk_brief.txt`.
# AGENTS: keep this script clean as much as possible. Add to src or create a new script in scripts/ if necessary.

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
from src.baseline_model import (
    ensure_pretrained_models,
    build_frame_dataset,
    run_inference,
    make_inference_dataroot,
    compute_metrics,
    save_comparison_grid,
    save_umap,
    PRETRAINED_MODELS,
    translate_test_video,
)

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

# Pick whichever pretrained model you want to evaluate in 2.1 and 2.2
EXP_NAME = "horse2zebra_cut_pretrained"  # or any from PRETRAINED_MODELS (TODO - list options here)

if EXP_NAME not in PRETRAINED_MODELS:
    raise ValueError(f"Unknown pretrained model '{EXP_NAME}'. Choose from: {PRETRAINED_MODELS}")


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

# -- 2.1 & 2.2 --
RUN_TRANSLATE_VIDEO = True

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
    print(f"[EXTRACT] Reloaded extracted patches from {extract_save_path} ({len(selected_detections)} patches)")
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

gcn_run_name  = reload_gcn if reload_gcn else SAVE_NAME
gcn_save_path = os.path.join(SAVE_DIR, "gcn_results", gcn_run_name)

gcn_params = dict(
    all_patches_dir = extract_save_path,
    pose_model_path = os.path.join(PROJECT_ROOT, "models/yolo26m-pose.pt"),
    device          = DEVICE,
    lr              = 3e-4,
    epochs          = 300,
    hidden          = 128,
    dropout         = 0.1,
    batch_size      = 256,
    # exclude_classes = ['others'],  # including 'others' is actually really important for generalising!!
    save_plots      = False,
)

if reload_gcn:
    results, summary = reload_gcn_results(gcn_save_path)
    print(f"[GCN] Reloaded GCN results from {gcn_save_path} (total={sum(summary.values())})")
else:
    if GCN_LABEL_SOURCE == "manual":
        if not RELOAD_ANNOTATIONS:
            raise ValueError("GCN_LABEL_SOURCE='manual' but RELOAD_ANNOTATIONS is not set.")
        if os.sep in RELOAD_ANNOTATIONS or "/" in RELOAD_ANNOTATIONS:
            ann_path = (
                os.path.join(PROJECT_ROOT, RELOAD_ANNOTATIONS)
                if not os.path.isabs(RELOAD_ANNOTATIONS)
                else RELOAD_ANNOTATIONS
            )
        else:
            ann_path = os.path.join(SAVE_DIR, "manual_annotated", RELOAD_ANNOTATIONS, "annotations.json")
        labelled_dir = os.path.dirname(ann_path)

    results, summary, gcn_per_class_val_acc = run_gcn_pipeline(
        labelled_dir = labelled_dir,
        cls_source   = GCN_LABEL_SOURCE,
        save_dir     = gcn_save_path,
        **gcn_params,
    )

total_classified = sum(summary.values())

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
        **gcn_params,
    )
    _, _, _manual_acc = run_gcn_pipeline(
        labelled_dir = _manual_dir,
        cls_source   = "manual",
        save_dir     = os.path.join(SAVE_DIR, "gcn_results", SAVE_NAME + "_ablation_manual"),
        **gcn_params,
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
# ## 2.1 Image Model Deployment (baseline model)

# %%


cut_dir = os.path.join(PROJECT_ROOT, "external/contrastive-unpaired-translation")
q2_1_dir = os.path.join(SAVE_DIR, "q2_1")

ensure_pretrained_models(cut_dir)

n_frames_per_domain = 500

trainA, trainB, testA, testB = build_frame_dataset(
    selected_detections,
    [d["path"] for d in TRAIN_DATA],
    os.path.join(q2_1_dir, "cut_data"),
    n_frames_per_domain,
)

results_dir = os.path.join(q2_1_dir, "results")

g2m_fakes = run_inference(
    cut_dir, EXP_NAME,
    make_inference_dataroot(testA, testB),
    os.path.join(results_dir, "g2m"),
    "AtoB", DEVICE,
)
m2g_fakes = run_inference(
    cut_dir, EXP_NAME,
    make_inference_dataroot(testB, testA),
    os.path.join(results_dir, "m2g"),
    "BtoA", DEVICE,
)

if RUN_TRANSLATE_VIDEO:
    baseline_video = translate_test_video(cut_dir, EXP_NAME, TEST_PATH, q2_1_dir, DEVICE)
    print(f"[CUT] Baseline video → {baseline_video}")
else:
    baseline_video = None
    print("[CUT] Skipping video translation (RUN_TRANSLATE_VIDEO=False).")

# %%
metrics = {
    "game→movie": compute_metrics(
        testB,
        os.path.join(results_dir, "g2m", "fake"),
        glob.glob(os.path.join(testA, "*.jpg")),
        g2m_fakes,
        DEVICE,
    ),
    "movie→game": compute_metrics(
        testA,
        os.path.join(results_dir, "m2g", "fake"),
        glob.glob(os.path.join(testB, "*.jpg")),
        m2g_fakes,
        DEVICE,
    ),
}
for direction, vals in metrics.items():
    print(f"[CUT] {direction}:  " + "  ".join(f"{k}: {v:.4f}" for k, v in vals.items()))

viz_dir = os.path.join(q2_1_dir, "viz")
os.makedirs(viz_dir, exist_ok=True)
with open(os.path.join(viz_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

save_comparison_grid(
    glob.glob(os.path.join(testA, "*.jpg")),
    g2m_fakes,
    "game → movie (CUT)",
    os.path.join(viz_dir, "comparison_game2movie.png"),
)
save_comparison_grid(
    glob.glob(os.path.join(testB, "*.jpg")),
    m2g_fakes,
    "movie → game (CUT)",
    os.path.join(viz_dir, "comparison_movie2game.png"),
)

save_umap(
    [
        glob.glob(os.path.join(testA, "*.jpg")),
        glob.glob(os.path.join(testB, "*.jpg")),
        g2m_fakes,
    ],
    ["game (real)", "movie (real)", "game→movie (fake)"],
    ["steelblue", "tomato", "mediumpurple"],
    "VGG feature UMAP: game→movie",
    os.path.join(viz_dir, "umap_game2movie.png"),
    device=DEVICE,
)
save_umap(
    [
        glob.glob(os.path.join(testA, "*.jpg")),
        glob.glob(os.path.join(testB, "*.jpg")),
        m2g_fakes,
    ],
    ["game (real)", "movie (real)", "movie→game (fake)"],
    ["steelblue", "tomato", "seagreen"],
    "VGG feature UMAP: movie→game",
    os.path.join(viz_dir, "umap_movie2game.png"),
    device=DEVICE,
)


# %% [markdown]
# ## 2.2 Enhanced model

# %%
# 1. use selected patch data from 1.3
# 2. keep the same pretrained CUT model as 2.1 (no retraining)
# 3. use temporal enhancement (deferred for now)

q2_2_dir = os.path.join(SAVE_DIR, "q2_2")
data_2_2 = os.path.join(q2_2_dir, "data")
trainA = os.path.join(data_2_2, "trainA")
trainB = os.path.join(data_2_2, "trainB")
testA = os.path.join(data_2_2, "testA")
testB = os.path.join(data_2_2, "testB")

for d in [trainA, trainB, testA, testB]:
    os.makedirs(d, exist_ok=True)

def _stage_paths(paths, out_dir):
    staged = 0
    for src in paths:
        dst = os.path.join(out_dir, os.path.basename(src))
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(src), dst)
            except FileExistsError:
                pass
            except OSError:
                # Fallback when symlinks are unavailable.
                shutil.copy2(src, dst)
            staged += 1
    return staged

print("[CUT] Staging 1.3-selected data for 2.2...")
print(f"[CUT] trainA (game):  +{_stage_paths(train_game, trainA)}")
print(f"[CUT] trainB (movie): +{_stage_paths(train_movie, trainB)}")
print(f"[CUT] testA (game):   +{_stage_paths(val_game if val_game else train_game[:200], testA)}")
print(f"[CUT] testB (movie):  +{_stage_paths(val_movie if val_movie else train_movie[:200], testB)}")

results_dir = os.path.join(q2_2_dir, "results")

g2m_fakes = run_inference(
    cut_dir, EXP_NAME,
    make_inference_dataroot(testA, testB),
    os.path.join(results_dir, "g2m"),
    "AtoB", DEVICE,
)
m2g_fakes = run_inference(
    cut_dir, EXP_NAME,
    make_inference_dataroot(testB, testA),
    os.path.join(results_dir, "m2g"),
    "BtoA", DEVICE,
)

if RUN_TRANSLATE_VIDEO:
    enhanced_video = translate_test_video(
        cut_dir,
        EXP_NAME,
        TEST_PATH,
        q2_2_dir,
        DEVICE,
        output_name="enhanced_model.mp4",
    )
    print(f"[CUT] Enhanced video → {enhanced_video}")
else:
    enhanced_video = None
    print("[CUT] Skipping video translation (RUN_TRANSLATE_VIDEO=False).")

# %%
metrics = {
    "game→movie": compute_metrics(
        testB,
        os.path.join(results_dir, "g2m", "fake"),
        glob.glob(os.path.join(testA, "*.jpg")),
        g2m_fakes,
        DEVICE,
    ),
    "movie→game": compute_metrics(
        testA,
        os.path.join(results_dir, "m2g", "fake"),
        glob.glob(os.path.join(testB, "*.jpg")),
        m2g_fakes,
        DEVICE,
    ),
}
for direction, vals in metrics.items():
    print(f"[CUT] {direction}:  " + "  ".join(f"{k}: {v:.4f}" for k, v in vals.items()))

viz_dir = os.path.join(q2_2_dir, "viz")
os.makedirs(viz_dir, exist_ok=True)
with open(os.path.join(viz_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

save_comparison_grid(
    glob.glob(os.path.join(testA, "*.jpg")),
    g2m_fakes,
    "game → movie (CUT)",
    os.path.join(viz_dir, "comparison_game2movie.png"),
)
save_comparison_grid(
    glob.glob(os.path.join(testB, "*.jpg")),
    m2g_fakes,
    "movie → game (CUT)",
    os.path.join(viz_dir, "comparison_movie2game.png"),
)

save_umap(
    [
        glob.glob(os.path.join(testA, "*.jpg")),
        glob.glob(os.path.join(testB, "*.jpg")),
        g2m_fakes,
    ],
    ["game (real)", "movie (real)", "game→movie (fake)"],
    ["steelblue", "tomato", "mediumpurple"],
    "VGG feature UMAP: game→movie",
    os.path.join(viz_dir, "umap_game2movie.png"),
    device=DEVICE,
)
save_umap(
    [
        glob.glob(os.path.join(testA, "*.jpg")),
        glob.glob(os.path.join(testB, "*.jpg")),
        m2g_fakes,
    ],
    ["game (real)", "movie (real)", "movie→game (fake)"],
    ["steelblue", "tomato", "seagreen"],
    "VGG feature UMAP: movie→game",
    os.path.join(viz_dir, "umap_movie2game.png"),
    device=DEVICE,
)