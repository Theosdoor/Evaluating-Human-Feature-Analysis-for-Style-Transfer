# ACV Coursework

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Theosdoor/ACV_cswk/blob/main/main.ipynb)

Deep learning pipeline to enhance the visual quality of humans in game videos using movie footage as reference. Uses unpaired image-to-image translation (CUT) applied locally to human patches extracted via YOLO detection and pose-based classification.

## Getting Started

```bash
# 1. Clone the repo
git clone <repo-url> && cd features-for-style-transfer

# 2. Download external dependencies (CUT fork + DINOv2 checkpoint)
#    Must run BEFORE uv sync — CUT fork is a uv workspace member
bash scripts/install_externals.sh

# 3. Install Python dependencies
uv sync && source .venv/bin/activate

# 4. Run the full pipeline
python3 nb_main.py
```

## Pipeline

| Stage | Module | Description |
|-------|--------|-------------|
| **1.1** | `src/feat_extract.py` | Extract human patches from video via YOLO; score by blur + motion |
| **1.2** | `src/classification.py` + `src/gcn.py` | Rule-based + GCN pose classification into 5 classes |
| **1.3** | `src/data.py` | Stratified patch selection using DINOv2 + K-Means clustering |
| **2.1** | `src/baseline_model.py` | Fine-tune CUT on full frames; evaluate with FID/KID/LPIPS |
| **2.2** | `src/enhanced_model.py` | Patch-level CUT with ST-GCN filtering, feathering, and EMA temporal blending |

Each stage can be skipped by setting `RELOAD_*` variables in `nb_main.py` to a prior run's timestamp.

## Dataset

| File | Duration | Size | Resolution | FPS | Frames | Domain |
|------|----------|------|------------|-----|--------|--------|
| `downloaded_data/Train/game/MafiaVideogame.mp4` | 2:21:04 | 484 MB | 1280×720 | 30.00 | 253,944 | game |
| `downloaded_data/Train/movie/TheGodfather.mp4` | 0:08:59 | 70 MB | 1280×720 | 23.98 | 12,945 | movie |
| `downloaded_data/Train/movie/TheIrishman.mp4` | 0:15:27 | 114 MB | 1280×720 | 25.06 | 23,236 | movie |
| `downloaded_data/Train/movie/TheSopranos.mp4` | 0:28:43 | 121 MB | 1280×720 | 30.00 | 51,714 | movie |
| `downloaded_data/Test/Test.mp4` | 0:01:10 | 17 MB | 1280×720 | 30.00 | 2,114 | game |

Videos live in `downloaded_data/` (git-ignored).

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_gcn.py` | Offline GCN training (`--label-source manual\|rule`, `--merge-annotations`) |
| `scripts/annotate.py` | Flask web UI for manual patch labelling |
| `scripts/figure_select.py` | Interactive selection of best comparison images for the paper |
| `scripts/nb_figures.py` | Generate notebook figures (score histograms, GCN curves) |
| `scripts/run_m2g_inference.py` | Re-run CUT inference in movie→game direction using cached checkpoints |
| `scripts/sample4submit.py` | Extract sample images from output for submission |
| `scripts/install_externals.sh` | Download DINOv2 checkpoint + clone CUT fork |