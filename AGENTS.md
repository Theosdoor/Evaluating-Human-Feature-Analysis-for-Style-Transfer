# ACV Coursework — Copilot Instructions

## Project Goal
Apply **movie visual style to game human footage** via unpaired image-to-image translation (e.g. CycleGAN). Humans are extracted and classified first so style transfer is applied locally (human patches only), not to the whole frame.

See [cswk_notes/cswk_brief.txt](cswk_notes/cswk_brief.txt) for assignment structure, mark breakdown, and detailed requirements.

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
- **Jupyter notebook** must stay in root dir and `import` from `src/` (`nb_main.py` will be the submission notebook).
- PDF report + ZIP (notebook + multimedia + optional `.py`/`.sh`)
- Do **not** include original `.mp4` data files; compress all media
- Include auto-download lines for any external models/datasets (`wget`, `git clone`)
- Other scripts besides `nb_main.py` can go into `scripts/`.
