# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

AI assistant guide for the ACV (Advanced Computer Vision) coursework repository.

> **Also read:** [`paper/AGENTS.md`](paper/AGENTS.md) for paper-writing rules. You may **read** files in `cswk_notes/` but do **not** edit them.

---

## Project Overview

Deep learning pipeline to enhance the visual quality of humans in game videos using movie footage as reference. The approach uses unpaired image-to-image translation (CycleGAN / CUT) applied **locally** to human patches extracted via YOLO detection and pose-based classification.

**Assignment structure (marks):**

| Q | Topic | Marks |
|---|-------|-------|
| 1.1 | Human patch extraction from video | 10% |
| 1.2 | Pose classification into 5 classes | 15% |
| 1.3 | Stratified training data selection | 15% |
| 2.1 | Baseline image model deployment (CUT/CycleGAN) | 20% |
| 2.2 | Local/temporal enhancement + video output | 30% |
| 3 | Report quality | 10% |

---

## Environment Setup

```bash
uv sync && source .venv/bin/activate
bash scripts/install_externals.sh   # downloads DINOv2 checkpoint + clones CUT fork
python3 nb_main.py                  # run the full pipeline
```

**Python version:** 3.12 (enforced via `.python-version`)
**Package manager:** `uv` (lock file: `uv.lock`)
**Device selection** (used across all modules):

```python
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
```

**WandB:** copy `.env.example` → `.env` and set `WANDB_API_KEY` / `WANDB_PROJECT`.

**Slurm (HPC):** partition `ug-gpu-small`, `--gres=gpu:turing:1`, 5 h, 28 GB RAM. Edit the last block of the target script before submitting. GPU used: **NVIDIA GeForce RTX 2080 Ti** (NCC HPC cluster). Always cite this in the report.

---

## Repository Layout

```
ACV_cswk/
├── nb_main.py                  # Main pipeline orchestrator (submit as .ipynb)
├── src/
│   ├── feat_extract.py         # Q1.1 — extract human patches from video
│   ├── classification.py       # Q1.2 — pose-based classifier (5 classes)
│   ├── data.py                 # Q1.3 — stratified train/val data selection
│   ├── baseline_model.py       # Q2.1 — CUT fine-tune on full frames
│   ├── enhanced_model.py       # Q2.2 — patch-level compositing + temporal blend
│   ├── gcn.py                  # GCN classifier (used in Q2.2 inference)
│   └── utils.py                # Shared: video I/O, CUT wrapper, dir naming
├── scripts/
│   ├── annotate.py             # Flask annotation tool for manual labelling
│   ├── debug_viz.py            # Debug visualisation helpers
│   └── install_externals.sh    # Download DINOv2 + clone CUT fork
├── paper/                      # LaTeX report (NeurIPS 2025 format)
│   ├── main.tex
│   ├── Makefile
│   ├── ref.bib
│   ├── sections/
│   │   ├── intro.tex
│   │   ├── human_feature_analysis.tex
│   │   └── real_world_application.tex
│   └── figs/                   # Figures for paper (PDF/PNG preferred)
├── external/
│   └── contrastive-unpaired-translation/   # Forked CUT repo (uv workspace member)
├── downloaded_data/            # Videos — NOT committed
│   ├── Train/game/MafiaVideogame.mp4
│   ├── Train/movie/{TheGodfather,TheIrishman,TheSopranos}.mp4
│   └── Test/Test.mp4
├── output/                     # All generated artefacts — NOT committed
├── figures/                    # Standalone figures (e.g. ablation plots)
├── cswk_notes/                 # Assignment notes — DO NOT EDIT
├── pyproject.toml
├── uv.lock
├── .env.example
└── AGENTS.md
```

---

## Source Modules

### `src/feat_extract.py` — Q1.1
Extracts human patches from video frames using YOLO detection. Scores patches by blur (Laplacian variance) and motion (frame difference). Outputs crops with metadata to `output/extracted_humans/<timestamp>/`.

### `src/classification.py` — Q1.2
Classifies extracted patches into 5 pose categories using MediaPipe/COCO keypoints + a lightweight MLP/SVM head. Produces per-class subdirectories and diagnostic renders. Uses `ClassifierConfig` dataclass for configuration.

### `src/gcn.py` — GCN classifier
`PoseGCN` model using `torch-geometric`. Uses COCO skeleton adjacency graph (`NUM_NODES`, `LOWER_KPS`, `CLASSES` constants). Trained with per-class validation; used in Q2.2 for inference-time patch classification.

### `src/data.py` — Q1.3
Stratified train/val split across both class and domain (game vs. movie). Outputs split manifests for CUT training.

### `src/baseline_model.py` — Q2.1
Fine-tunes the CUT model on full frames. Wraps CUT as a subprocess via `src/utils.py`. Evaluates with FID, KID, and LPIPS via `cleanfid` and `lpips` libraries.

### `src/enhanced_model.py` — Q2.2
Fine-tunes CUT on selected patches. Composites translated patches back onto source frames and applies EMA temporal blending across frames for video consistency.

### `src/utils.py`
- `run_cut(...)` — subprocess wrapper for CUT training/inference
- `get_next_reclassify_dir(base)` — **always use this** for new classification output dirs (never reimplement suffix logic)
- Video I/O helpers, timestamped directory creation

---

## Key Conventions

### Output directory naming

`SAVE_NAME = time.strftime('%Y%m%d-%H%M%S')` is set once at notebook startup and used as the base name for the current run.

| Stage | Output path |
| ----- | ----------- |
| 1.1 extraction | `output/extracted_humans/<SAVE_NAME>/` |
| 1.2 rule-based classification | `output/init_classifications/<SAVE_NAME>-<N>/` (via `get_next_reclassify_dir`) |
| 1.2 manual annotations | `output/manual_annotated/<SAVE_NAME>/annotations.json` |
| 1.2 GCN training | `output/gcn_results/<SAVE_NAME>/` |
| 2.1 full-frame model | `output/q2_1/<SAVE_NAME>/` |
| 2.2 enhanced model | `output/q2_2/<SAVE_NAME>/` |

- **Timestamped runs:** `YYYYMMDD-HHMMSS` prefix (e.g. `20260314-195748`)
- **Reclassification reruns:** use `get_next_reclassify_dir` from `src/utils.py`; never reimplement the suffix-increment logic elsewhere
- Expected pattern: if `20260314-195748-1` and `20260314-195748-3` exist, next dir is `20260314-195748-4`
- **RELOAD_* variables** in `nb_main.py` short-circuit each stage by pointing to a prior run's output directory name

### Print logging tags

All modules prefix log lines with bracketed tags:

| Tag | Module |
|-----|--------|
| `[EXTRACT]` | feat_extract.py |
| `[CLS]` | classification.py |
| `[GCN]` | gcn.py |
| `[CUT]` | utils.py (CUT subprocess wrapper) |
| `[VIDEO]` | utils.py (video I/O helpers) |
| `[DATA]` | data.py |
| `[Q2.1]` | baseline_model.py |
| `[ENH]` | enhanced_model.py |

### Code style

- **snake_case** for functions and variables
- **PascalCase** for classes
- **ALL_CAPS** for module-level constants (`CLASSES`, `NUM_NODES`, `LOWER_KPS`, etc.)
- Type hints used throughout; dataclasses for configuration objects
- Imports: stdlib → third-party → local (`from src...`)
- Section separators: `# -----------`

### Reload / checkpoint system

Each pipeline stage accepts a `reload` flag that skips recomputation and loads from the most recent timestamped output dir. Preserve this pattern when adding new stages.

### Notebook (`nb_main.py`)

- Must stay in the root directory; submitted as `.ipynb`
- **Keep `nb_main.py` clean** — move any non-pipeline logic into `src/` or `scripts/`
- All scripts beyond the main pipeline go in `scripts/`
- Include `wget` / `git clone` auto-download lines for external models

---

## Data

| File | Duration | Resolution | FPS | Domain |
|------|----------|------------|-----|--------|
| `Train/game/MafiaVideogame.mp4` | 2 h 21 min | 1280×720 | 30 | game |
| `Train/movie/TheGodfather.mp4` | 8 min 59 s | 1280×720 | 23.98 | movie |
| `Train/movie/TheIrishman.mp4` | 15 min 27 s | 1280×720 | 25.06 | movie |
| `Train/movie/TheSopranos.mp4` | 28 min 43 s | 1280×720 | 30 | movie |
| `Test/Test.mp4` | 1 min 10 s (2,114 frames) | 1280×720 | 30 | game |

Videos live in `downloaded_data/` which is git-ignored. Never commit `.mp4` files.

---

## External Dependencies

| Dependency | Location | How installed |
|------------|----------|---------------|
| DINOv2 ViT-B/14 checkpoint | `external/dinov2/` | `scripts/install_externals.sh` (wget) |
| Contrastive Unpaired Translation (CUT) | `external/contrastive-unpaired-translation/` | `scripts/install_externals.sh` (git clone) |

The CUT repo is a **uv workspace member** (`pyproject.toml → [tool.uv.workspace]`).

---

## Paper

Build with:

```bash
make -C paper          # → paper/build/main.pdf
make -C paper clean    # wipe artefacts
```

Always build after editing `.tex` files to catch errors. Do **not** commit `paper/build/`.

**Word limits (examiners stop reading at the limit):**

| Section | Max words |
|---------|-----------|
| 1.1 | 100 |
| 1.2 | 100 |
| 1.3 | 100 |
| 2.1 | 250 |
| 2.2 | 250 |

Included in count: prose, in-text citations, footnotes.
Excluded: diagrams, tables, equations, abstract, bibliography, appendices.

Figures go in `paper/figs/`; reference with relative paths (e.g. `figs/my_fig.pdf`).

**Marks require hardware and training time** — be explicit about what GPU was used and how long training took. Marks are not awarded for raw compute.

---

## What NOT to Do

- Do **not** edit files in `cswk_notes/`
- Do **not** commit `downloaded_data/`, `output/`, `wandb/`, or `paper/build/`
- Do **not** reimplement the suffix-increment logic for classification dirs — use `get_next_reclassify_dir`
- Do **not** push `.mp4` files or model checkpoints
- Do **not** modify `paper/neurips_2025.sty`

---

## Submission Checklist

- [ ] `nb_main.py` converted to `.ipynb` and placed in repo root
- [ ] Auto-download lines present for DINOv2 and CUT (`wget`, `git clone`)
- [ ] All media compressed; no raw `.mp4` files included
- [ ] Paper PDF compiled and included alongside code ZIP
- [ ] Hardware used and training time documented in report
- [ ] All external code, models, and datasets cited in report and notebook comments
