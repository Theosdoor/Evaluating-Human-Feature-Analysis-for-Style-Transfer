# ACV Coursework — Copilot Instructions

## Project Goal
Apply **movie visual style to game human footage** via unpaired image-to-image translation (e.g. CycleGAN). Humans are extracted and classified first so style transfer is applied locally (human patches only), not to the whole frame.

## Assignment Structure & Marks
| Q | Task | Mark | Status |
|---|------|------|--------|
| 1.1 | Human patch extraction from Train videos → 1,000+ patches | 10% | In progress |
| 1.2 | Classify patches into 5 pose classes | 15% | In progress |
| 1.3 | Select best patches for style transfer training | 15% | 🔲 TODO |
| 2.1 | Train/deploy unpaired image-to-image model (CycleGAN etc.) | 20% | 🔲 TODO |
| 2.2 | Apply model to Test video with temporal enhancement | 30% | 🔲 TODO |

All written up in final report (see `paper/` folder).

(Full cswk brief at cswk_notes/cswk_brief.txt)

## Pipeline Architecture

**Two-stage, reference-based design** — detections store `{video_path, frame_num, bbox}` not raw pixels; `save_patches` re-reads videos sequentially to avoid memory blowout.

**Frame filtering before YOLO** (in `extract_humans_from_video`):
1. Inter-frame diff < `scene_change_threshold` → skip duplicate frames  
2. Enforce `yolo_interval` minimum gap  
3. Buffer `yolo_batch_size=8` eligible frames → single GPU call

**Film vs game blur threshold** inferred from filename keywords (`movie`, `godfather`, etc.); film gets a lower Laplacian variance threshold.

**Classification is evidence-weighted** (`classify_orientation` in `src/classification.py`): accumulates `front_score`/`back_score` with weighted signals (face visibility = 3.0, geometry = 1.0, ears = 0.5, shoulder-hip ratio = 0.3); ambiguous detections → `others`.

## Models
Stored in `models/` (not committed, not downloaded automatically):
- `models/yolov8m.pt` — person detection (class 0 only)
- `models/yolo26m-pose.pt` — 17-keypoint COCO pose estimation

COCO keypoint index reference is at the top of `src/classification.py`.

## Environment & Execution

```bash
cd /path/to/dir
uv sync
source .venv/bin/activate
python3 main.py
```

**Slurm** (`submit_job.sh`): single GPU (RTX 2080ti), max. 28 GB RAM, CUDA auto-selected:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)
```

## Output Layout
```
output/
  extracted_humans/<timestamp>/     # 1,000+ scored patches (submit random 50)
  classifications/<timestamp>/
    full_body_front/ full_body_back/
    head_shoulder_front/ head_shoulder_back/ others/  # submit random 20/class
```

## Submission Requirements
- **Jupyter notebook** must stay in root dir and `import` from `src/`
- PDF report + ZIP (notebook + multimedia + optional `.py`/`.sh`)
- Do **not** include original `.mp4` data files; compress all media
- Include auto-download lines for any external models/datasets (`wget`, `git clone`)

## Conventions
- Use `uv add` / `uv remove` for dependencies; never bare `pip install`
- Use `tqdm` for all large-iterable loops
- `score_detection` weights: confidence 0.30, relative area 0.25, sharpness 0.20, centering 0.15, aspect ratio 0.10
- Async `imwrite` via `ThreadPoolExecutor` in `save_patches` is intentional — disk I/O only, no GPU contention

## Experiment Logging (Required)
Keep `EXPERIMENTS.md` up to date for reproducibility.
- When running one-off/manual experiments add a short entry to `EXPERIMENTS.md` with the command, outputs directory, and key results.
