# Implementation Plan — Stages 1.3, 2.1, 2.2

## Current State

Stages 1.1 and 1.2 are complete. The `20260225-104100` run produced:

- **1,000 patches** (250 per source: MafiaVideogame, TheGodfather, TheIrishman, TheSopranos)
- **Score range**: 0.57–0.94 (mean 0.72)
- **Classification**: 632 full\_body\_front (63.2%), 228 head\_shoulder\_front (22.8%), 65 full\_body\_back (6.5%), 38 head\_shoulder\_back (3.8%), 37 others (3.7%)
- **MafiaVideogame is massively undersampled**: 250 patches from 2h21m = ~1.8 patches/min vs ~14 patches/min for movie sources

Dependencies already in `pyproject.toml`: torch, torchvision, ultralytics, opencv-python, scikit-learn, numpy, tqdm, timm, umap-learn, clean-fid.

---

## 1.3 — Training Data Selection (15%)

### Goal

Select high-quality, diverse human patches from 1.1 for CycleGAN training. The brief requires **original observations and insights**, so the selection method must be motivated by empirical analysis of the extracted data, not just applied off-the-shelf.

### Method

**Two-stage pipeline**: quality filter → diversity selection.

1. **Quality filter** — Laplacian variance ≥ 0.6 score threshold to remove blurry/low-quality patches
2. **DINO diversity selection** — Extract features with `vit_small_patch8_224.dino` (via `timm`), k-means cluster per domain, stratified sample across clusters

Split game/movie for k-means separately to preserve domain balance. Target **~450 patches per domain** (total ~900).

### Empirical Insights for the Report

The brief explicitly asks you to *"present and utilise your own observations and insights"*. These visual analyses motivate and justify the selection method — include them as figures in the report:

> [!IMPORTANT]
> Pick 2–3 of these. Each one should lead to a concrete design choice in your pipeline, not just be a pretty picture.

#### Option A: Per-Film Color Distribution (recommended)

Plot per-channel HSV histograms (or violin plots) for patches from each of the three movies vs the game. This will reveal:
- **The Godfather** has warm, amber tones (high saturation, low value)
- **The Irishman** is cooler and more desaturated (digital grading)
- **The Sopranos** is closest to naturalistic TV lighting

**Insight → design choice**: The three films have inconsistent palettes, so naive random sampling is biased toward whichever film dominates. DINO diversity sampling corrects for this by ensuring cluster coverage across all visual styles.

#### Option B: Blur Quality Spectrum (recommended)

Show a grid of 12–15 patches arranged by Laplacian variance (lowest → highest), with the variance value overlaid. Mark the cutoff threshold visually.

**Insight → design choice**: There's a clear visual cliff where patches below threshold N are too blurry for texture transfer. This motivates the quality floor and makes the threshold choice empirically grounded rather than arbitrary.

#### Option C: UMAP Embedding Visualisation (recommended)

Two side-by-side UMAP scatter plots of the DINO features:
1. Coloured by **domain** (game vs movie) — shows domain separation
2. Coloured by **film source** (MafiaVideogame / Godfather / Irishman / Sopranos) — shows sub-clusters within movie domain

**Insight → design choice**: The movie domain fragments into sub-clusters per film, confirming that random sampling would over-represent the densest sub-cluster. k-means + per-cluster sampling ensures representation across all visual styles.

#### Option D: Patch Scale Distribution

Histogram of bbox areas (in pixels) per source. Game patches tend to be smaller/more distant (wide shots) vs movie patches which include many close-ups.

**Insight → design choice**: Extreme size outliers may harm CycleGAN training (very small patches upscaled to 256×256 become blurry). Could motivate a minimum bbox area filter.

#### Option E: Pairwise Distance Before/After Selection

Box plot comparing average pairwise cosine distance in DINO space for: (a) random sample of 450 patches, (b) DINO-diversity selected 450 patches. The diversity-selected set should have higher mean pairwise distance.

**Insight → design choice**: Quantitative proof that the selection method increases diversity vs random sampling.

### Implementation

**New file**: `src/data_selection.py`

```python
select_training_data(
    classifications_dir,       # output/classifications/<timestamp>
    output_dir,                # output/selected_for_training/
    min_score=0.6,
    n_clusters=50,             # per domain
    target_per_domain=450,
    device='cuda',
) -> dict                     # stats: per-domain counts, cluster info, paths to plots
```

Steps:
1. Glob all `.jpg` in each class subdir; parse score from filename; drop below `min_score`
2. Split by domain: game (`MafiaVideogame` in filename) vs movie
3. Per domain: DINO ViT-S/8 → 384-dim CLS token → k-means → sample from clusters by score
4. Copy selected patches to `output/selected_for_training/{game,movie}/`
5. Generate UMAP plot + any other insight figures

**DINO loading** (`timm`, already installed):
```python
dino = timm.create_model('vit_small_patch8_224.dino', pretrained=True, num_classes=0)
```

### Prerequisite: Re-extract with More Game Patches

MafiaVideogame is undersampled. Re-run extraction with `n2save=1500` for game, `n2save=600` per movie source. This requires updating `nb_main.py` to use per-video targets rather than an even split.

### Output

```
output/selected_for_training/
    game/       # ~450 patches
    movie/      # ~450 patches
    _selection_summary.txt
    umap_plot.png
    color_distributions.png  # or other insight figures
```

---

## 2.1 — Image Model Deployment (20%)

### Goal

Train/download a CycleGAN for game↔movie translation. Apply to **full frames** (not human patches). Evaluate with FID + SSIM. Show 10 success + 10 failure cases per direction.

> [!IMPORTANT]
> The brief says *"apply it to your dataset of frames"*. 2.1 is whole-frame translation. The distribution mismatch between training on human patches and applying to full frames is a **known limitation** — discuss it in the report as motivation for 2.2's local approach.

### Model Choice

Use the **junyanz/pytorch-CycleGAN-and-pix2pix** repo. This is the canonical CycleGAN implementation and also supports CUT (`--model cut`) in the same codebase.

**CycleGAN vs CUT decision**:

| | CycleGAN | CUT |
|---|---|---|
| Training speed | ~3–6 hrs | ~2–3 hrs |
| VRAM | ~2 GB | ~1.5 GB |
| Small dataset performance | Decent, mode-collapse risk | Better on limited data |
| Brief alignment | Explicitly mentioned | Still unpaired — satisfies brief |

**Recommendation**: Train CycleGAN (as the brief explicitly mentions it). Mention CUT as a known alternative in the report discussion. If CycleGAN shows mode collapse, switch to CUT as a fallback — it's a one-flag change (`--model cut`).

### Training Config

```bash
python train.py --dataroot datasets/game2movie \
    --name game2movie --model cycle_gan \
    --n_epochs 50 --n_epochs_decay 50 \
    --load_size 286 --crop_size 256 \
    --batch_size 1 --pool_size 50 --no_dropout \
    --save_epoch_freq 10 --gpu_ids 0
```

Train **from scratch** (no horse2zebra pretrained weights) — avoids pretrained-bias artifacts and keeps the report narrative clean.

### CycleGAN Normalisation Pipeline

The junyanz repo normalises all inputs to `[-1, 1]`. Every code path that touches the generator needs explicit preprocess/postprocess:

```python
def preprocess_for_cyclegan(bgr_uint8: np.ndarray) -> torch.Tensor:
    """BGR uint8 numpy → [-1, 1] RGB float32 tensor, NCHW."""
    rgb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0)

def postprocess_from_cyclegan(tensor: torch.Tensor) -> np.ndarray:
    """[-1, 1] RGB float32 tensor → BGR uint8 numpy."""
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    rgb = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
```

Put these in `src/style_transfer.py` — reused in 2.1 eval and 2.2 video pipeline.

### Evaluation Metrics

- **FID** (`clean-fid`, already installed): `FID(generated_game→movie, real_movie)` and `FID(generated_movie→game, real_game)`
- **SSID**: `skimage.metrics.structural_similarity` input vs translated output (measures structure preservation)
- **10 success + 10 failure pairs per direction** — select failures by lowest SSID or visual inspection

### 2.1 Application to Full Frames

For evaluation, apply the trained CycleGAN to a set of **full frames** sampled from the training videos. These are the same frames selected in 1.1 (before cropping to human patches). This means:
- Read the original frame from the video at the stored `frame_num`
- Resize to CycleGAN input size (286→256 crop)
- Translate → save

The patch-trained model applied to full frames will produce artifacts (background contamination, inconsistent style on non-human regions). **Document these failures explicitly** — they are the motivation for 2.2.

### Implementation

**New file**: `src/style_transfer.py`

```python
prepare_cyclegan_data(selected_dir, cyclegan_data_dir) -> None  # symlinks trainA/trainB
load_cyclegan_generator(checkpoint_path, device) -> torch.nn.Module
preprocess_for_cyclegan(bgr_uint8) -> torch.Tensor
postprocess_from_cyclegan(tensor) -> np.ndarray
evaluate_cyclegan(checkpoint_dir, test_dir, direction) -> dict  # FID, per-image SSIM
apply_to_full_frames(generator, frame_paths, output_dir, device) -> None
```

**New file**: `scripts/train_cyclegan.sh` (Slurm wrapper)

### Output

```
external/cyclegan/checkpoints/game2movie/    # model weights
output/cyclegan_results/
    game_to_movie/    # translated full frames
    movie_to_game/
    fid_scores.txt
    ssim_scores.txt
    success_pairs/    # 10 best per direction
    failure_pairs/    # 10 worst per direction
```

---

## 2.2 — Local Temporal Enhancement (30%)

### Goal

Improve on 2.1 by: (1) applying CycleGAN to human patches only (local method using 1.1–1.3), and (2) adding temporal consistency across frames. Compare quantitatively to 2.1.

This section carries the most marks. The brief offers two improvement paths:
1. *"Using methods in 1.1 to 1.3, improve by using local information"*
2. *"Developing an advanced model that considers temporal information for extra credit"*

**We do both**: local patch application + temporal blending.

### Stage 1 — Local Patch Application

Re-run YOLOv8-pose on `Test.mp4` frames → crop → CycleGAN → composite back.

Key details:
- **Resize**: Crop to bbox → resize to 256×256 → CycleGAN → resize back to original bbox dims (bilinear)
- **Multiple humans**: Process all detections per frame, largest-area first (most important subject gets clean translation)
- **Background unchanged**: Only human patches get style-transferred — this is the explicit improvement over 2.1

**Compositing** — two modes, selected per-detection by border proximity:
1. **`cv2.seamlessClone`** (`NORMAL_CLONE`) — best quality, but requires bbox ROI to be ≥5px from frame border
2. **Gaussian feathering** — alpha blend with a soft mask tapering at bbox edges (~10px border). Fallback for edge-case bboxes

### Stage 2 — Temporal Consistency

Two approaches, implemented incrementally:

#### Primary: EMA Patch Blending (simpler, always works)

For each tracked detection across consecutive frames, blend the current translated patch with the previous frame's translated patch at the same location, weighted by bbox IoU overlap:

```
translated_patch[t] = β * CycleGAN(crop[t]) + (1-β) * translated_patch[t-1]
```

where β = IoU(bbox[t], bbox[t-1]). Low overlap (new person / fast motion) → β≈1 (no blending). High overlap (stationary) → smooth.

This is cheap, doesn't require an extra model, and still smooths flicker effectively.

#### Advanced: RAFT Optical Flow (extra credit)

Full-frame temporal blending using RAFT flow estimation:

```
output[t] = α * stylized[t] + (1-α) * warp(output[t-1], flow[t-1→t])
```

- Use `torchvision.models.optical_flow.raft_small` (pretrained, lower VRAM than `raft_large`)
- α=0.7 as starting point. Scene cuts (reuse `frame_difference` from 1.1) → reset buffer
- Warp via `torch.nn.functional.grid_sample`

**Implementation order**: Build EMA first → produce working video → add RAFT on top. This way you always have a submittable output.

### VRAM Budget (RTX 2080 Ti — 11 GB)

| Model | Precision | Approx. VRAM |
|---|---|---|
| YOLOv8m | FP16 | ~0.8 GB |
| CycleGAN ResNet-9 generator | FP32 | ~0.5 GB |
| RAFT small | FP32 | ~1.0 GB |
| PyTorch CUDA overhead | — | ~0.8 GB |
| Per-frame activations | | ~2–4 GB |
| **Working total** | | **~5–7 GB** |

Comfortable with 4–6 GB headroom. Precautions:
- `torch.no_grad()` for all inference
- YOLO with `half=True`
- If OOM: run RAFT and CycleGAN sequentially per frame with `torch.cuda.empty_cache()` between them

### Temporal Consistency Metric

**Warping error** = average pixel difference between a warped stylised frame and the next stylised frame. Compute for both 2.1 (full-frame) and 2.2 (local + temporal) to quantify improvement.

### Implementation

**New file**: `src/video_pipeline.py`

```python
apply_local_style_transfer(
    cyclegan_model, yolo_model, video_path, output_path,
    device, blend_mode='poisson',
) -> None

apply_temporal_style_transfer(
    cyclegan_model, yolo_model, flow_model,  # RAFT (optional, None for EMA-only)
    video_path, output_path, device,
    alpha=0.7, scene_change_threshold=30.0,
    blend_mode='poisson',
) -> None
```

**Video I/O**: `cv2.VideoWriter` with same fps/resolution as input. Compress final with ffmpeg.

### Output

```
output/
    test_baseline_full_frame.mp4   # 2.1: whole-frame CycleGAN (for comparison)
    test_local.mp4                 # 2.2 Stage 1: local-only
    test_temporal.mp4              # 2.2 Stage 2: local + temporal
    test_keyframes/                # 10+ comparison frames for report
```

---

## New Files Summary

| File | Purpose |
|---|---|
| `src/data_selection.py` | 1.3 — DINO features, k-means clustering, balanced sampling, insight plots |
| `src/style_transfer.py` | 2.1 — CycleGAN data prep, training wrapper, normalisation, FID/SSIM eval |
| `src/video_pipeline.py` | 2.2 — Per-frame local application, EMA + RAFT temporal blending |
| `scripts/train_cyclegan.sh` | Slurm job for CycleGAN training |
| `scripts/run_test_video.sh` | Slurm job for test video processing |

---

## Execution Order

```
1. Re-extract MafiaVideogame with n2save=1500; raise movie sources to n2save=600
2. Run 1.3 data selection → output/selected_for_training/{game,movie}/
3. Generate insight plots for report (color distributions, UMAP, blur spectrum)
4. Clone junyanz/pytorch-CycleGAN-and-pix2pix → external/cyclegan/
5. Prepare CycleGAN data (symlinks trainA/trainB)
6. Train CycleGAN (Slurm, ~3–6 hours)
7. Evaluate CycleGAN: FID + SSIM + success/failure pairs on full frames
8. Build local style transfer pipeline (2.2 Stage 1)
9. Generate test_local.mp4
10. Add temporal blending — EMA first, then RAFT
11. Generate test_temporal.mp4
12. Extract keyframes for report comparison + compute warping error
```

---

## Dependencies

### Already installed

| Package | Used for |
|---|---|
| `timm` | DINO feature extraction (1.3) |
| `umap-learn` | UMAP visualisation (1.3) |
| `clean-fid` | FID evaluation (2.1) |
| `scikit-learn` | k-means clustering (1.3) |
| `torchvision` | RAFT optical flow (2.2) |

### Still needed

```bash
uv add dominate visdom  # junyanz CycleGAN repo deps — verify against requirements.txt after cloning
```

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| CycleGAN doesn't converge | Monitor loss; extend to 200 epochs or switch to `--model cut` (same repo, one flag) |
| Too few game patches | Re-extract MafiaVideogame with n2save=1500 — trivial given 2h21m of footage |
| RAFT + CycleGAN exceeds 11 GB | Use `raft_small`; if OOM, run sequentially with `torch.cuda.empty_cache()` |
| `seamlessClone` fails at frame edges | Fall back to Gaussian feathering per-detection based on border proximity check |
| Temporal blending causes ghosting | Increase α; disable blending when flow magnitude exceeds threshold |
| RAFT too complex / slow | EMA patch blending is the primary method; RAFT is additive extra credit |

---

## Report Figures (planned)

1. **1.3**: Color distribution per film (HSV violins), blur quality spectrum grid, UMAP scatter (domain + film-coloured)
2. **2.1**: Input/output grid per direction, FID over epochs, 10 success + 10 failure pairs, discussion of full-frame limitations
3. **2.2**: Side-by-side keyframes (original → full-frame 2.1 → local 2.2 → temporal 2.2), warping error plot, zoom on seam boundaries

---

## Verification Plan

### Automated

- **1.3**: Assert output dir contains ~450 patches per domain; assert no patch below `min_score`; assert UMAP plot file exists
- **2.1**: Run `clean-fid` on generated vs real images; assert FID scores are finite and within expected range; assert 10 success + 10 failure pairs saved
- **2.2**: Assert output videos exist, are correct duration (~70s), and correct resolution (1280×720); compute warping error and assert it's lower for temporal vs local-only

### Manual (user)

- **1.3**: Visually inspect UMAP plot and color distribution figures — do they tell a coherent story?
- **2.1**: Watch translated frames — is the style transfer visually apparent? Are failure cases genuine?
- **2.2**: Watch both output videos — is temporal consistency visibly improved? Are seams visible?
