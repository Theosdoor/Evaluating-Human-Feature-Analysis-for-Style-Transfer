# AGENTS.md

Always load the `python` skill before **writing** Python code in this repo.

## Project Goal
Deep learning solution to enhance visual quality of humans in game videos using movie footage as reference. Unpaired image-to-image translation (CycleGAN) applied locally to human patches extracted via YOLO detection + pose classification.

See [cswk_notes/cswk_brief.md](cswk_notes/cswk_brief.md) for assignment structure and requirements. Don't edit files in `cswk_notes/`.

---

## Environment

```bash
cd /home2/nchw73/Year4/ACV_cswk
uv sync && source .venv/bin/activate
python3 nb_main.py   # or run cells interactively
```

**Slurm** (`slurm/submit_job.sh`): partition `ug-gpu-small`, `--gres=gpu:turing:1`, 5 h, 28 GB RAM. Currently runs `scripts/ablate_mediapipe.py` — edit the last block to change target.

```python
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
```

## Data

```
downloaded_data/
  Train/game/MafiaVideogame.mp4     # 141 min
  Train/movie/TheGodfather.mp4  TheIrishman.mp4  TheSopranos.mp4
  Test/Test.mp4
```



## Submission Requirements

- **Jupyter notebook** must stay in root dir; `nb_main.py` will be submitted as `.ipynb`
- Do **not** include original `.mp4` files; compress all media
- Include auto-download lines for external models (`wget`, `git clone`)
- Scripts other than `nb_main.py` go in `scripts/`
- Don't use type hints, or `from __future__ import annotations`

See [paper/AGENTS.md](paper/AGENTS.md) for paper build instructions.
