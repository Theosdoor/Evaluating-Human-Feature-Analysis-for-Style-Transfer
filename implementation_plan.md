# Implementation Plan — Stages 1.3, 2.1, 2.2

## Dataset Details

| File | Duration | Size | Resolution | FPS | Frames | Bitrate | Domain |
|------|----------|------|------------|-----|--------|---------|--------|
| `downloaded_data/Train/game/MafiaVideogame.mp4` | 2:21:04 | 484 MB | 1280×720 | 30.00 | 253,944 | ~0.5 Mbps | game |
| `downloaded_data/Train/movie/TheGodfather.mp4` | 0:08:59 | 70 MB | 1280×720 | 23.98 | 12,945 | ~1.1 Mbps | movie |
| `downloaded_data/Train/movie/TheIrishman.mp4` | 0:15:27 | 114 MB | 1280×720 | 25.06 | 23,236 | ~1.0 Mbps | movie |
| `downloaded_data/Train/movie/TheSopranos.mp4` | 0:28:43 | 121 MB | 1280×720 | 30.00 | 51,714 | ~0.6 Mbps | movie |
| `downloaded_data/Test/Test.mp4` | 0:01:10 | 17 MB | 1280×720 | 30.00 | 2,114 | ~2.0 Mbps | game |

**Total training:** ~53 min movie + ~141 min game. **Test:** 70 seconds, 2,114 frames @ 30 fps.

All sources are 720p. MafiaVideogame is heavily compressed (~0.5 Mbps) but resolution is adequate for 256×256 CycleGAN patches — `--crop_size 256` is fine.

---

## Current State

Stages 1.1 and 1.2 are complete. The `20260225-104100` run produced:

- **1,000 patches** (250 per source: MafiaVideogame, TheGodfather, TheIrishman, TheSopranos). Can run nb_main.py again for more patches if appropriate.
- **Score range**: 0.57–0.94 (mean 0.72)
- **Classification distribution**: 632 full\_body\_front (63.2%), 228 head\_shoulder\_front (22.8%), 65 full\_body\_back (6.5%), 38 head\_shoulder\_back (3.8%), 37 others (3.7%)
- Game patches come solely from MafiaVideogame; movie patches from the other three sources.
- **MafiaVideogame is massively undersampled**: 250 patches from a 2h21m video = ~1.8 patches/min vs ~14 patches/min for movie sources. This is a sampling config issue, not a data scarcity problem — there is ample game footage.

Dependencies already in `pyproject.toml`: torch, torchvision, ultralytics, opencv-python, scikit-learn, numpy, tqdm, **timm**, **umap-learn**, **clean-fid**. All managed via `uv`; to add further deps use `uv add <package>`, never edit `pyproject.toml` directly.

---

## 1.3 — Training Data Selection (15 marks)

### Evaluation of Proposed Plan

The plan is sound overall. Specific notes and adjustments:

**Class filtering** — Keep all five classes as specified in the brief: `full_body_front`, `full_body_back`, `head_shoulder_front`, `head_shoulder_back`, and `others`. The current distribution is heavily skewed (632 / 228 / 65 / 38 / 37), so we need to fix the upstream sampling to get more back-view and head-shoulder patches (see "Fixing class imbalance at extraction" below). After re-extraction, drop only the truly unusable patches (e.g. score < threshold, severe occlusion) rather than entire classes.

**Quality floor at score ≥ 0.5** — The current minimum is 0.57, so a 0.5 threshold removes nothing from this particular run. Either raise it to ~0.6 to actually trim the long tail, or leave it permissive and let DINO diversity do the heavy lifting. **Use 0.6** — it's a light trim that removes the worst ~10% while keeping enough volume.

**DINO diversity selection** — Good choice. Practical notes:

- `facebook/dino-vits8` (ViT-S/8, 21M params) fits easily in GPU memory alongside the rest. Use the CLS token output (384-dim) as the feature vector — no need for spatial tokens.
- k=50 clusters is reasonable for a pool in the 300–800 range. If post-filtering leaves fewer than 500 patches, reduce k proportionally (e.g. k = max(20, pool_size // 10)).
- Sample up to N/k patches per cluster, prioritising by extraction score within each cluster. This retains the quality signal while enforcing diversity.
- **Be careful with the game/movie split**: run k-means on each domain separately to ensure diversity within each domain. If you cluster the combined pool, game and movie patches may land in the same clusters and the per-cluster sampling won't preserve domain balance.

**Game/movie balance** — Currently you have 250 game + 750 movie patches. After class filtering 925 remain, roughly 240 game / 685 movie. This is not a data scarcity problem: MafiaVideogame has 2h21m of footage and should yield 1000–1500 raw patches easily. The fix is simply raising the extraction target, not augmentation or accepting imbalance.

**Action**: re-extract MafiaVideogame targeting **1000–1500 raw patches** (see "Fixing class imbalance at extraction" below). After DINO diversity selection, target **450 patches per domain** — this brings the training set much closer to the ~1000 images/domain used in the original CycleGAN horse2zebra experiments. 300/domain is unnecessarily conservative given the available source material.

### Fixing class imbalance at extraction

The current 63% full\_body\_front dominance is a sampling artefact compounded by undersampling MafiaVideogame. A 2h21m game video naturally contains abundant back-views, head-shoulder shots, and crowd scenes — they are simply being discarded by aggressive deduplication and a low `n2save` cap. Two changes fix it:

1. **Lower the `temporal_gap` in `diverse_sampling`** — the current gap of 30 frames aggressively deduplicates, which discards back-view and head-shoulder crops that appear briefly. Reduce to `temporal_gap=15` or even `10` to retain more short-lived poses. The DINO diversity step downstream will handle the actual deduplication more intelligently.

2. **Increase extraction volume** — set `n2save=1500` for MafiaVideogame (and raise movie sources to `n2save=600` each), then let the DINO diversity selection in 1.3 trim down to 450 per domain. More raw material means the rare classes (back views, head\_shoulder\_back) have a better chance of surviving quality filtering. Class-aware budgeting in `diverse_sampling` is a secondary fix — unnecessary if extraction volume is high enough.

The simplest approach: extract 2000 patches (option 3), then in `select_training_data` do stratified sampling — allocate a minimum quota per class (e.g. 15% of target) and fill the rest proportionally. This keeps the extraction code simple and handles balancing in one place.

**UMAP visualisation** — Good report figure. Use `umap-learn` with `n_neighbors=15, min_dist=0.1`, colour by domain and by k-means cluster. Two subplots: one coloured by domain (game vs movie), one by cluster ID. This shows both the domain separation and the diversity guarantee.

### Implementation

**New file**: `src/data_selection.py`

```
select_training_data(
    classifications_dir,       # output/classifications/<timestamp>
    output_dir,                # output/selected_for_training/
    keep_classes,              # ['full_body_front', 'full_body_back', 'head_shoulder_front']
    min_score,                 # 0.6
    n_clusters,                # 50 (per domain)
    target_per_domain,         # 450
    device,                    # 'cuda'
) -> dict                     # stats: per-domain counts, cluster info
```

Steps inside:
1. Glob all `.jpg` in each class subdir of `classifications_dir` (all five classes)
2. Parse score from filename; drop below `min_score`
3. Split by domain: game (`MafiaVideogame` in filename) vs movie (everything else)
4. For each domain:
   a. Load each patch, resize to 224×224, run through DINO ViT-S/8, extract CLS token → 384-dim
   b. k-means clustering (scikit-learn) with k = min(n_clusters, len(pool) // 5)
   c. **Stratified sampling**: allocate a minimum per-class quota (e.g. 15% of `target_per_domain` reserved across underrepresented classes: `full_body_back`, `head_shoulder_back`, `others`). Fill each quota from the relevant clusters by score, then fill remaining budget proportionally from all clusters.
   d. This ensures rare classes are represented without discarding them entirely
5. Copy selected patches to `output/selected_for_training/{game,movie}/`
6. Return stats dict; optionally save UMAP plot to output dir

**DINO loading** — use `timm` (already installed) to load the model:
```python
import timm
dino = timm.create_model('vit_small_patch8_224.dino', pretrained=True, num_classes=0)
dino.eval()
```
The `num_classes=0` call returns the CLS token (384-dim) directly. `timm` downloads weights on first call (~80 MB) and caches them in `~/.cache/huggingface/hub`. Alternatively, `torch.hub.load('facebookresearch/dino:main', 'dino_vits8')` works without `timm` but `timm` is already a project dependency so prefer it.

### Integration into nb\_main.py

Add a new `# %%` cell block after the classification section:

```python
# 1.3. Training Data Selection
from src.data_selection import select_training_data

selection_stats = select_training_data(
    classifications_dir=cls_save_path,
    output_dir=os.path.join(SAVE_DIR, "selected_for_training"),
    keep_classes=CLASSES,  # all five classes
    min_score=0.6,
    n_clusters=50,
    target_per_domain=450,
    device=DEVICE,
)
```

### Output

```
output/selected_for_training/
    game/       # ~450 patches
    movie/      # ~450 patches
    _selection_summary.txt
    umap_plot.png
```

---

## 2.1 — Image Model Deployment (20 marks)

### Evaluation of Proposed Plan

**Model choice (junyanz/pytorch-CycleGAN-and-pix2pix)** — Correct call. This repo is the canonical implementation, well tested, and has straightforward training CLI commands. The alternative (a lightweight CycleGAN implementation from scratch) would take significantly more time for questionable benefit.

**Pretrained horse2zebra then fine-tune vs train from scratch** — The plan suggests starting from a pretrained checkpoint. Worth noting:

- horse2zebra weights will transfer some texture-translation capability, but the domain gap is large. Fine-tuning will converge faster than random init but may inherit biases (e.g. zebra stripes bleeding into outputs).
- For a 50–100 epoch training run on 256×256 patches with ~300 images per domain, an RTX 2080 Ti will need approximately 3–6 hours. This is very achievable within a Slurm job.
- **Train from scratch for 100 epochs.** With only ~300 images per domain at 256×256, convergence is fast. Starting from scratch avoids pretrained-bias artifacts and makes the report narrative cleaner ("we trained a CycleGAN on our curated data" vs "we fine-tuned a horse2zebra model").

**Training specifics to pin down**:

- `--dataroot output/selected_for_training --name game2movie`
- `--model cycle_gan --pool_size 50 --no_dropout`
- `--load_size 286 --crop_size 256` (standard CycleGAN random-crop augmentation) — all sources confirmed 1280×720, so 256×256 patches are safe.
- `--batch_size 1` (default for CycleGAN; batch_size > 1 is possible but changes BatchNorm statistics)
- `--n_epochs 50 --n_epochs_decay 50` (linear LR decay in second half)
- `--gpu_ids 0`
- Save checkpoints every 10 epochs for comparison in the report

**Evaluation metrics**:

- **FID**: use `clean-fid`. Compute FID(generated\_game→movie, real\_movie) and FID(generated\_movie→game, real\_game). Lower is better. With ~300 images, FID will have high variance — note this in the report.
- **SSIM**: `skimage.metrics.structural_similarity` between each input and its translated output. This measures structure preservation, not style transfer quality. Report per-image SSIM with mean/std.
- **Show 10 success + 10 failure pairs per direction** as the brief requires. Select failures by lowest SSIM or visual inspection (faces with artifacts, mode collapse, background bleed).

**Potential issue**: the junyanz repo expects data in `trainA/` and `trainB/` subdirectories. We need to either structure the selected_for_training output to match, or create symlinks. Simplest: rename at training time or use a wrapper script.

### CycleGAN normalisation pipeline

The junyanz repo normalises all inputs to `[-1, 1]` (per-channel `mean=0.5, std=0.5` after dividing by 255). Patches from OpenCV are BGR `uint8`. Failing to match this will produce garbage output regardless of model quality. Every code path that touches the generator needs explicit preprocess/postprocess:

**Preprocess** (before feeding to generator):
```python
def preprocess_for_cyclegan(bgr_uint8: np.ndarray) -> torch.Tensor:
    """BGR uint8 numpy → [-1, 1] RGB float32 tensor, NCHW."""
    rgb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0  # [0, 1]
    tensor = (tensor - 0.5) / 0.5                                     # [-1, 1]
    return tensor.unsqueeze(0)                                         # [1, C, H, W]
```

**Postprocess** (after generator output):
```python
def postprocess_from_cyclegan(tensor: torch.Tensor) -> np.ndarray:
    """[-1, 1] RGB float32 tensor → BGR uint8 numpy."""
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = tensor * 0.5 + 0.5                   # [0, 1]
    tensor = tensor.clamp(0, 1)
    rgb = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
```

Put these in `src/style_transfer.py` and use them everywhere: in evaluation (2.1), in the video pipeline (2.2), and in any notebook visualisation cells. This is a common silent failure — the model runs without errors but produces colour-inverted or washed-out results.

### Loading the CycleGAN generator for inference (2.1 → 2.2)

The junyanz repo saves checkpoints as `{epoch}_net_G_A.pth` and `{epoch}_net_G_B.pth` (generators A→B and B→A) inside `checkpoints/{name}/`. To load a generator outside the repo's own test script:

```python
import sys
sys.path.insert(0, 'external/cyclegan')  # make repo importable

from models.networks import define_G

def load_cyclegan_generator(
    checkpoint_path: str,
    input_nc: int = 3,
    output_nc: int = 3,
    ngf: int = 64,
    netG: str = 'resnet_9blocks',
    norm: str = 'instance',
    device: str = 'cuda',
) -> torch.nn.Module:
    """Load a trained CycleGAN generator for inference."""
    net = define_G(input_nc, output_nc, ngf, netG, norm,
                   use_dropout=False, init_type='normal',
                   init_gain=0.02, gpu_ids=[])
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(state_dict)
    net.to(device).eval()
    return net

# All inference calls should use autocast for reduced VRAM and faster Tensor Core throughput:
# with torch.no_grad(), torch.amp.autocast('cuda'):
#     fake = generator(patch)

# Usage:
generator_g2m = load_cyclegan_generator(
    'external/cyclegan/checkpoints/game2movie/latest_net_G_A.pth',
    device=DEVICE,
)  # game → movie
generator_m2g = load_cyclegan_generator(
    'external/cyclegan/checkpoints/game2movie/latest_net_G_B.pth',
    device=DEVICE,
)  # movie → game
```

This uses the repo's own `define_G` to build the architecture (matching the training config exactly), then loads the state dict. Put `load_cyclegan_generator` in `src/style_transfer.py` so both 2.1 evaluation and 2.2 video pipeline can import it.

### Implementation

**Clone and setup** (in `scripts/setup_cyclegan.sh` or directly in notebook):

```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git external/cyclegan
# install the repo's extra deps into the project venv via uv
uv add dominate visdom  # junyanz requirements not already covered
```

Do **not** use `pip install` or `pip install -r` directly — always use `uv add` so deps stay tracked in `pyproject.toml`. Check `external/cyclegan/requirements.txt` first; most packages (torch, torchvision, numpy, Pillow) are already present and `uv` will skip them.

**New file**: `src/style_transfer.py`

```
prepare_cyclegan_data(
    selected_dir,              # output/selected_for_training/
    cyclegan_data_dir,         # external/cyclegan/datasets/game2movie/
) -> None                     # creates trainA/, trainB/ symlinks

train_cyclegan(
    cyclegan_root,             # external/cyclegan/
    data_name,                 # 'game2movie'
    n_epochs, n_epochs_decay,
    gpu_id,
) -> str                      # path to checkpoints dir

evaluate_cyclegan(
    cyclegan_root,
    checkpoint_dir,
    test_dir,                  # can be the training data itself for in-domain eval
    direction,                 # 'AtoB' or 'BtoA'
) -> dict                     # FID, per-image SSIM
```

**New file**: `scripts/train_cyclegan.sh` (Slurm wrapper)

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=06:00:00

cd external/cyclegan
python train.py --dataroot datasets/game2movie \
    --name game2movie --model cycle_gan \
    --n_epochs 50 --n_epochs_decay 50 \
    --load_size 286 --crop_size 256 \
    --batch_size 1 --pool_size 50 --no_dropout \
    --save_epoch_freq 10 --gpu_ids 0
```

**FID dependency** — use `clean-fid` (already installed, preferred over `pytorch-fid` as it corrects for JPEG compression artifacts in FID computation). No further install needed.

### Integration into nb\_main.py

```python
# 2.1. CycleGAN Training & Evaluation
from src.style_transfer import prepare_cyclegan_data, evaluate_cyclegan

prepare_cyclegan_data(
    selected_dir=os.path.join(SAVE_DIR, "selected_for_training"),
    cyclegan_data_dir="external/cyclegan/datasets/game2movie",
)
# Training is run via Slurm (scripts/train_cyclegan.sh) or inline:
# !cd external/cyclegan && python train.py ...

# Evaluation
eval_results = evaluate_cyclegan(...)
```

### Output

```
external/cyclegan/checkpoints/game2movie/    # model weights
output/cyclegan_results/
    game_to_movie/    # translated images
    movie_to_game/    # translated images
    fid_scores.txt
    ssim_scores.txt
    success_pairs/    # 10 best per direction
    failure_pairs/    # 10 worst per direction
```

---

## 2.2 — Local Temporal Enhancement (30 marks)

### Evaluation of Proposed Plan

This section carries the most marks (30%) so it needs the most attention.

**Stage 1 — Local patch application**: The approach is exactly what the brief asks for ("local methods from 1.1 to 1.3"). Detection → crop → translate → composite. A few practical concerns:

- **Seam artifacts at bbox boundaries**: Raw paste-back will be visible. Use Poisson blending (`cv2.seamlessClone`) or a soft alpha mask (Gaussian feathering at bbox edges, ~10px border) to smooth the transition. **`cv2.seamlessClone` caveat**: it requires the mask ROI to not touch the image border — if the bbox is at the frame edge (person walking in/out of shot), `seamlessClone` will either fail or produce artifacts. Handle this by:
  1. Checking if the bbox is within a margin (e.g. 5px) of the frame border
  2. If so, fall back to Gaussian feathering (alpha blend with a soft mask that tapers at the bbox edges)
  3. If not, use `seamlessClone` with `cv2.NORMAL_CLONE` for best quality
  
  Implement both blending modes and select per-detection based on border proximity.
- **Resize mismatch**: Detected bboxes won't be 256×256. Resize the crop to 256×256 before CycleGAN, then resize back to original bbox dimensions. Bilinear interpolation both ways is fine.
- **Multiple humans per frame**: Run YOLO, translate each detection independently, composite all back. Process detections from largest to smallest to handle overlaps (largest patch painted first = most important subject gets clean translation).
- **Non-human regions unchanged**: This is the whole point — only human patches get style-transferred. Background stays native. This contrast is a good report discussion point.

**Stage 2 — Temporal consistency with optical flow**: The plan is workable but has engineering complexity. Evaluation:

- **RAFT** is a good flow estimator and has pretrained weights in torchvision (`torchvision.models.optical_flow.raft_large` or `raft_small`). `raft_small` uses less VRAM — relevant on the 2080 Ti if CycleGAN is also loaded.
- **The blending formula** `output[t] = α * cyclegan(frame[t]) + (1-α) * warp(output[t-1], flow)` is correct for simple temporal smoothing. α=0.7 is a reasonable starting point. Lower α = smoother but more ghosting on fast motion.
- **Scene cut handling**: You already have inter-frame difference logic in `extract_humans_from_video`. Reuse the `frame_difference` function: when diff > threshold, reset the temporal buffer (output[t] = cyclegan(frame[t]), no warping).
- **Performance concern**: Running RAFT + CycleGAN + YOLO per frame will be slow. For a test video of reasonable length, budget ~1–2 seconds per frame on the 2080 Ti. A 30fps, 60-second clip = 1800 frames ≈ 30–60 minutes. This is manageable in a Slurm job but worth noting.
- **Warp implementation**: Use `torch.nn.functional.grid_sample` to warp the previous frame using the RAFT flow field. This is differentiable and GPU-accelerated.

### VRAM budget (RTX 2080 Ti — 11 GB)

All three models need to be resident during 2.2 video processing. Approximate VRAM per model:

| Model | Precision | Approx. VRAM |
|---|---|---|
| YOLOv8m (detection) | FP16 | ~0.8 GB |
| CycleGAN ResNet-9 generator | FP32 | ~0.5 GB |
| RAFT small | FP32 | ~1.0 GB |
| PyTorch CUDA overhead | — | ~0.8 GB |
| **Subtotal (persistent)** | | **~3.1 GB** |
| Per-frame activations (256×256 CycleGAN + 520×960 RAFT) | | ~2–4 GB |
| **Working total** | | **~5–7 GB** |

This leaves 4–6 GB headroom — comfortable, but only if models are loaded once and reused. Do **not** re-instantiate models per frame.

Precautions:
- Load CycleGAN generator with `torch.no_grad()` context for all inference — saves activation memory.
- Run YOLO with `half=True` (FP16 on CUDA) as already done in extraction.
- If OOM occurs, process RAFT and CycleGAN sequentially per frame with `torch.cuda.empty_cache()` between them, rather than keeping both live.
- Monitor with `torch.cuda.max_memory_allocated()` during the first 10 frames and log it.

**Alternative temporal approach (simpler, still effective)**: Instead of full optical-flow blending, apply a simple exponential moving average (EMA) on the translated patches only (not the full frame). For each tracked bbox across frames, blend the current translated patch with the previous frame's translated patch at the same location, weighted by IoU overlap. This is much cheaper than RAFT and still smooths flicker. Use RAFT as the "advanced" version described in the report.

**Suggested implementation order**: Build Stage 1 first, produce the "local application" video, then add Stage 2 on top. This way you always have a working output to submit, even if temporal blending has bugs.

### Implementation

**New file**: `src/video_pipeline.py`

```
apply_local_style_transfer(
    cyclegan_model,            # loaded CycleGAN generator
    yolo_model,                # YOLOv8 detector
    video_path,                # downloaded_data/Test/Test.mp4
    output_path,               # output/test_local.mp4
    device,
    blend_mode='poisson',      # 'paste' | 'feather' | 'poisson'
) -> None

apply_temporal_style_transfer(
    cyclegan_model,
    yolo_model,
    flow_model,                # RAFT
    video_path,
    output_path,               # output/test_temporal.mp4
    device,
    alpha=0.7,
    scene_change_threshold=30.0,
    blend_mode='poisson',
) -> None
```

**Per-frame processing loop** (inside `apply_local_style_transfer`):

```
for each frame:
    1. Run YOLO → list of (bbox, confidence)
    2. with torch.no_grad(), torch.amp.autocast('cuda'):
       For each bbox (sorted by area, largest first):
          a. Crop patch from frame
          b. Resize to 256×256
          c. Run through CycleGAN generator (game→movie direction)
          d. Resize translated patch back to bbox dimensions
          e. Composite onto frame using blend_mode
    3. Write frame to output video
```

**Per-frame processing loop** (inside `apply_temporal_style_transfer`):

```
prev_output = None
prev_frame = None

for each frame:
    1. Run YOLO → detections
    2. with torch.no_grad(), torch.amp.autocast('cuda'):
       For each detection: crop → resize → CycleGAN → resize back
    3. Composite all translated patches → current_stylized
    4. If prev_output is not None:
       a. with torch.no_grad(), torch.amp.autocast('cuda'):
          Compute flow = RAFT(prev_frame, frame)
       b. warped = grid_sample(prev_output, flow)
       c. Check frame_difference(prev_frame, frame)
       d. If scene_cut: output = current_stylized
          Else: output = α * current_stylized + (1-α) * warped
    5. Else:
       output = current_stylized
    6. Write output to video
    7. prev_output = output; prev_frame = frame
```

**Video I/O**: Use `cv2.VideoWriter` with the same fps/resolution as the input. Use the `mp4v` or `avc1` codec. Compress the final video with ffmpeg to keep the ZIP small.

**New file**: `scripts/run_test_video.sh` (Slurm wrapper for the full test pipeline)

### Integration into nb\_main.py

```python
# 2.2. Local + Temporal Enhancement
from src.video_pipeline import apply_local_style_transfer, apply_temporal_style_transfer
from src.style_transfer import load_cyclegan_generator

# Load CycleGAN generator (see "Loading the CycleGAN generator" in 2.1 section)
generator = load_cyclegan_generator(
    'external/cyclegan/checkpoints/game2movie/latest_net_G_A.pth',
    device=DEVICE,
)

# Stage 1: Local patch application
apply_local_style_transfer(
    generator, yolo_model, TEST_PATH,
    output_path=os.path.join(SAVE_DIR, "test_local.mp4"),
    device=DEVICE,
)

# Stage 2: Temporal consistency
flow_model = torchvision.models.optical_flow.raft_small(pretrained=True).to(DEVICE).eval()
apply_temporal_style_transfer(
    generator, yolo_model, flow_model, TEST_PATH,
    output_path=os.path.join(SAVE_DIR, "test_temporal.mp4"),
    device=DEVICE,
)
```

### Output

```
output/
    test_local.mp4              # Stage 1: local-only style transfer
    test_temporal.mp4           # Stage 2: + temporal blending
    test_keyframes/             # 10+ comparison frames for the report
```

---

## Dependencies

### Already installed

| Package | Version in pyproject.toml | Used for |
|---|---|---|
| `timm` | `>=1.0.25` | DINO feature extraction (1.3) |
| `umap-learn` | `>=0.5.11` | UMAP visualisation (1.3) |
| `clean-fid` | `>=0.1.35` | FID evaluation (2.1) |
| `scikit-learn` | `>=1.7.2` | k-means clustering (1.3) |
| `torchvision` | `>=0.25.0` | RAFT optical flow (2.2) |

### Still needed

The junyanz CycleGAN repo has a small set of extra deps (`dominate`, `visdom`) not yet in `pyproject.toml`. Check `external/cyclegan/requirements.txt` after cloning and add any missing ones with:

```bash
uv add dominate visdom  # example — verify against actual requirements.txt
```

RAFT: `torchvision.models.optical_flow` (already have torchvision — no install needed). DINO weights: auto-downloaded by `timm` on first use, cached in `~/.cache/huggingface/hub`.

### Rule

Always use `uv add <package>` to add new dependencies. Never `pip install` ad-hoc or edit `pyproject.toml` directly. Use `uv sync` after pulling to keep the venv in sync with `pyproject.toml`.

---

## New Files Summary

| File | Purpose |
|---|---|
| `src/data_selection.py` | 1.3 — DINO features, k-means clustering, balanced sampling |
| `src/style_transfer.py` | 2.1 — CycleGAN data prep, training wrapper, FID/SSIM eval |
| `src/video_pipeline.py` | 2.2 — Per-frame local application, temporal blending |
| `scripts/train_cyclegan.sh` | Slurm job for CycleGAN training |
| `scripts/run_test_video.sh` | Slurm job for test video processing |

---

## Execution Order

```
1. Re-extract MafiaVideogame with n2save=1500; raise movie sources to n2save=600 each
2. Run 1.3 data selection → output/selected_for_training/{game,movie}/  # target 450 per domain
4. Clone junyanz/pytorch-CycleGAN-and-pix2pix → external/cyclegan/
5. Prepare CycleGAN data (symlinks trainA/trainB)
6. Train CycleGAN (Slurm, ~3-6 hours)
7. Evaluate CycleGAN: FID + SSIM + success/failure pairs
8. Build local style transfer pipeline (Stage 1)
9. Generate test_local.mp4 (~70s, ~1700–2100 frames, budget 30–70 min on 2080 Ti)
10. Add RAFT temporal blending (Stage 2)
11. Generate test_temporal.mp4
12. Extract keyframes for report comparison
```

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| CycleGAN doesn't converge in 100 epochs | No style transfer at all | Monitor training loss; fall back to 200 epochs or reduce image pool size |
| Too few game patches (~240) | Imbalanced training | Re-extract MafiaVideogame with n2save=1500 — 2h21m of footage makes this trivial |
| RAFT + CycleGAN exceeds 11 GB VRAM | OOM on 2080 Ti | Use `raft_small`; keep all models loaded once (see VRAM budget table — ~5–7 GB working total); if OOM, run RAFT and CycleGAN sequentially with `torch.cuda.empty_cache()` between them |
| Poisson blending artifacts at bbox edges | Visible seams in output video | Fall back to Gaussian feathering; expand bbox by 10% to include context |
| Temporal blending causes ghosting on fast motion | Visual quality degradation | Increase α or disable blending when flow magnitude exceeds threshold |
| Test video processing too slow for Slurm wall time | Incomplete output | Process at lower fps (skip every other frame, interpolate) or use a shorter clip |

---

## Report Figures (planned)

1. **1.3**: UMAP scatter plot (domain-coloured + cluster-coloured), histogram of per-cluster sample counts
2. **2.1**: 4×5 grid of input/output pairs per direction; FID progression over training epochs; 2×10 success/failure grids
3. **2.2**: Side-by-side keyframes (original → local-only → temporal); temporal consistency plot (frame-to-frame pixel diff over time for both methods); zoom-in on seam boundaries and failure cases
