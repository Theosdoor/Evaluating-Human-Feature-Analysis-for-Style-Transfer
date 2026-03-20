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
# Set to a run timestamp to reload existing outputs and skip stages already done.
# Leave as None to run the full pipeline from scratch.
# e.g. RELOAD_RUN = "20260224-224712"
RELOAD_EXTRACT = None # skip 1.1
RELOAD_EXTRACT = "20260314-195748"

# -- 1.2 --
# Set to None to run classification.
# Set to a folder name under output/classifications to reload existing results.
RELOAD_CLS = None
# RELOAD_CLS = "20260314-195748-3"

# -- 1.2b --
# Set to a path (relative to project root, or absolute) to an annotations.json produced
# by scripts/annotate.py. Those manual labels will override the rule-based classification.
# e.g. RELOAD_ANNOTATIONS = "output/manual_annotated/20260314-195748-5/annotations.json"
RELOAD_ANNOTATIONS = None

# -- 1.3 --
RELOAD_TRAIN_SELECT = None # skip 1.3

# -- 2.1 --
# toggles for quick checkpoint-only runs
RUN_TRAIN_CUT = False
RUN_TRANSLATE_VIDEO = False

# -- 2.2 --

# Effective reload controls (RUN_FULL_PIPELINE overrides per-stage reload flags)
reload_extract = None if RUN_FULL_PIPELINE else RELOAD_EXTRACT
reload_cls = None if RUN_FULL_PIPELINE else RELOAD_CLS
reload_annotations = None if RUN_FULL_PIPELINE else RELOAD_ANNOTATIONS

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

detections = []         # all raw detections (for diagnostics)
selected_detections = []

if reload_extract:
    # Reload existing extracted patches
    selected_detections = reload_extracted_patches(extract_save_path, [d["path"] for d in TRAIN_DATA])
else:
    model = YOLO(os.path.join(PROJECT_ROOT, 'models/yolov8m.pt'))
    model.to(DEVICE)

    # Equal per-domain budgets: CUT needs balanced domains, so extract ~2000 per domain.
    # Movie budget is split proportionally by duration across its 3 films.
    domain_budget = n2save // 2  # 2000 per domain
    targets = []
    movie_durations = [d["duration"] for d in TRAIN_DATA if d["domain"] == 'movie']
    movie_total = sum(movie_durations)
    for data in TRAIN_DATA:
        if data["domain"] == 'game':
            targets.append(domain_budget)
        else:
            targets.append(int(domain_budget * data["duration"] / movie_total))
    targets[-1] += n2save - sum(targets)  # absorb rounding remainder

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
    # 50 to submit (do once we've got good results)
    print(f"Using freshly-extracted patches from {extract_save_path}")
    save_extraction_summary(extract_save_path, detections, selected_detections)

# %% [markdown]
# ## 1.2. Classification

# %%
cls_input_path = extract_save_path
cls_base_dir = os.path.join(SAVE_DIR, "classifications")
if reload_cls:
    cls_save_path = os.path.join(cls_base_dir, reload_cls)
else:
    cls_save_path = get_next_reclassify_dir(cls_base_dir, os.path.basename(extract_save_path))

classify_b_size = 32

if reload_cls:
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
        save_debug_viz=False,   # set True to save YOLO-annotated images to debug_viz/
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

# %% [markdown]
# ## 1.2b. Apply Manual Annotations (optional)

# %%
# If a manual annotations.json exists, merge those labels into `results` and `summary`.
# Patches annotated manually override the rule-based classification.
if reload_annotations:
    ann_path = os.path.join(PROJECT_ROOT, reload_annotations) if not os.path.isabs(reload_annotations) else reload_annotations
    if not os.path.exists(ann_path):
        print(f"[WARN] annotations file not found: {ann_path}")
    else:
        import json
        with open(ann_path) as _f:
            _raw_annotations = json.load(_f)

        # _raw_annotations: {fname: {"label": ..., "source": ...}}
        # Rebuild summary from scratch so counts stay accurate.
        _overridden = 0
        ALL_LABELS_SET = set(CLASSES) | {"bad_extraction"}
        for _fname, _entry in _raw_annotations.items():
            _label = _entry.get("label")
            if _label not in ALL_LABELS_SET:
                print(f"  [WARN] unknown label '{_label}' for {_fname}, skipping")
                continue
            _old = results.get(_fname)
            if _old and _old != _label:
                summary[_old] = summary.get(_old, 0) - 1
                _overridden += 1
            if _label in summary:
                summary[_label] = summary.get(_label, 0) + (0 if _old == _label else 1)
            results[_fname] = _label

        total_classified = sum(v for v in summary.values() if v > 0)
        print(f"Loaded {len(_raw_annotations)} manual annotations from {ann_path}")
        print(f"  {_overridden} patches had their rule-based label overridden")
        print("  " + "  ".join(f"{k}: {v}" for k, v in summary.items() if v > 0))
else:
    print("No manual annotations loaded (RELOAD_ANNOTATIONS=None).")

# %% [markdown]
# ## 1.3 Training Data Selection

# %%
save_dir = os.path.join(SAVE_DIR, "training_data_selection", os.path.basename(cls_save_path))

from src.data import get_data_split, flat_paths, flat_paths_by_domain

split = get_data_split(
    cls_save_path,
    train_split=1.0,
    exclude_classes=['others'],  # drop ambiguous patches
)

train_game, train_movie = flat_paths_by_domain(split['train'])
val_game, val_movie = flat_paths_by_domain(split['val'])

# TODO - same splits at save_dir
# os.makedirs(save_dir, exist_ok=True)



# %% [markdown]
# ## 2.1 Image Model Deployment (baseline model)

# %%
# 1. using contrastive-unpaired-translation, apply it to output/extracted_humans/20260314-195748 to convert images between game and movie domains.
# 2. analyse performance in both directions using appropriate metrics (e.g. FID, KID, LPIPS) and visualizations (e.g. UMAPs).
# 3. Transfer the style humans in downloaded_data/Test/Test.mp4 and save the video to output/baseline_model.

# %%
# Dataset preparation + CUT training
from src.baseline_model import build_frame_dataset, train_cut, translate_test_video

cut_dir = os.path.join(PROJECT_ROOT, "external/contrastive-unpaired-translation")
data_2_1 = os.path.join(SAVE_DIR, "cut_data")
exp_game2movie = "cut_game2movie"
exp_movie2game = "cut_movie2game"

# Hyperparams — reduce epochs or batch size if time / VRAM constrained
n_frames_per_domain = 500
n_epochs            = 20
n_epochs_decay      = 5
batch_size          = 4      # use 2 if VRAM < 8 GB

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
# Metrics (FID, KID, LPIPS) & viz
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

    # Visualisation — comparison grids + UMAPs
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