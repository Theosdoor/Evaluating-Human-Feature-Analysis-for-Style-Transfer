# Implementation Plan — Full Pipeline

## Current State

| Stage | Status | Key files |
|---|---|---|
| 1.1 — Patch Extraction | ✅ Done | `src/feat_extract.py`, `nb_main.py` |
| 1.2 — Classification | ✅ Done (testing MediaPipe redundancy) | `src/classification.py` |
| 1.3 — Training Data Selection | 🔲 Not started | `src/data_selection.py` (new) |
| 2.1 — Image Model Deployment | 🔲 Not started | `src/style_transfer.py` (new) |
| 2.2 — Local Temporal Enhancement | 🔲 Not started | `src/video_pipeline.py` (new) |

**Existing run** (`20260225-104100`): 1,000 patches (250/source), scores 0.57–0.94, classification: 63.2% full\_body\_front, 22.8% head\_shoulder\_front, 6.5% full\_body\_back, 3.8% head\_shoulder\_back, 3.7% others.

**MafiaVideogame undersampled**: 250 patches from 2h21m = ~1.8 patches/min vs ~14 patches/min for movie sources.

---

## 1.1 — Human Patch Extraction (10%) ✅

### What's Implemented

[feat_extract.py](file:///Users/Subspace_Explorer/Projects/ACV_cswk/src/feat_extract.py) — reference-based design (detections store `{video_path, frame_num, bbox}`, patches cached in memory).

**Frame filtering before YOLO** (in `extract_humans_from_video`):
1. Inter-frame diff < `scene_change_threshold` (8.0) → skip duplicate frames
2. Enforce `yolo_interval=10` minimum gap
3. `cap.grab()` (no decode) for frames deep inside the interval gap
4. Buffer `yolo_batch_size=8` eligible frames → single GPU call

**Scoring** (`score_detection`): weighted composite of confidence (0.30), relative size (0.25), sharpness (0.20), centering (0.15), aspect ratio (0.10).

**Diverse sampling**: greedy, score-descending, with `temporal_gap=10` per-source. O(n).

**Saving**: `ThreadPoolExecutor` for async disk I/O. Fast path uses cached patches; fallback re-reads videos.

### Implemented Improvements

- [x] **Equal per-domain budgets**: `n2save=4000` split 2000 game / 2000 movie. Movie budget split proportionally by film duration (Godfather 338, Irishman 581, Sopranos 1081). Ensures both domains have comparable raw pools for DINO diversity selection.
- [x] **Lower `temporal_gap`** from 30 → 10: retains brief back-views and head-shoulder poses; DINO diversity in 1.3 handles deduplication

### Report Notes (100 words max)

Mention: YOLOv8m for person detection, frame differencing to skip duplicates, batched GPU inference, composite quality scoring, diverse sampling with temporal gap. Justify the lightweight frame selection as necessary given 2h21m + 53min of source footage.

---

## 1.2 — Classification (15%) ✅

### What's Implemented

[classification.py](file:///Users/Subspace_Explorer/Projects/ACV_cswk/src/classification.py) — evidence-weighted rule-based classifier over YOLOv8-pose keypoints.

**Body extent** (`classify_body_extent`):
- Full body: ankle visibility (strong), hips + knee (moderate), ≥2 lower-body points (weak)
- Head-shoulder: ≥1 shoulder or ≥2 upper-body points
- Aspect-ratio veto: h/w < 1.5 downgrades full\_body → head\_shoulder

**Orientation** (`classify_orientation`): accumulates `front_score` / `back_score`:
- MediaPipe face detection: +3.0 front (strongest signal)
- YOLO face keypoints (nose, eyes): +3.0 for ≥2, +1.2 for 1
- Back evidence: +2.5 when no face at any confidence but ≥6 keypoints overall
- Nose between shoulders: +1.0 front
- Ears without face: +1.2 per ear back
- Shoulder/hip width ratio: +0.3 front or +0.2 back
- Margin < 0.12 → `others` (ambiguous)

**Adaptive threshold** (`adaptive_conf_high`): per-patch confidence floor from median keypoint confidence, clipped to [0.15, 0.30]. Handles film footage having systematically lower pose confidence.

### Open Question: MediaPipe Redundancy

Currently testing whether MediaPipe face detection adds value over YOLO keypoints alone:
- MediaPipe contributes +3.0 to `front_score` (strongest single signal)
- But YOLO face keypoints (≥2 visible) also contribute +3.0
- If both fire on the same patches (high correlation), MediaPipe is redundant overhead

**Test**: Run classification with `face_detector=None` and compare results. If the `others` class grows significantly (many patches lose the front evidence floor), MediaPipe is earning its keep. If distribution stays similar, remove it.

**Impact of removal**: Simplifies pipeline (no external .tflite download), removes `mediapipe` dependency, slightly faster per-image. No effect on body extent classification.

### Potential Improvements

- [ ] **Remove MediaPipe if redundant** — pending test results
- [ ] **GCN classifier** — possible but likely circular (training on pseudo-labels from these same rules). Not recommended unless you have hand-labelled ground truth. Discuss in report as potential improvement only.

### Report Notes (100 words max)

Mention: YOLOv8-pose 17 COCO keypoints, evidence-weighted scoring, adaptive confidence threshold for film/game consistency, geometric rules for body extent (ankle/knee visibility) and orientation (face keypoint patterns). Justify the rule-based approach: no labelled training data, geometric features are directly interpretable.

---

## 1.3 — Training Data Selection (15%)

### Goal

Select high-quality, diverse patches for CUT training. The brief requires **original observations and insights** — the method must be motivated by empirical analysis, not just applied off-the-shelf.

### Method: Quality Filter → Diversity Selection

1. **Quality floor**: Drop patches with extraction score < 0.6 (~trims worst 10%)
2. **DINO diversity**: `vit_small_patch8_224.dino` (via `timm`) → 384-dim CLS token → k-means per domain → stratified sample from clusters by score
3. **Domain balance**: k-means run separately on game and movie pools. Target **~450 patches per domain**

### Empirical Insights for the Report

> [!IMPORTANT]
> Pick 2–3. Each must lead to a concrete design choice — not just a visualisation.

#### A. Per-Film Color Distribution ⭐

HSV violin plots for patches from each movie vs game. Reveals:
- The Godfather: warm amber tones
- The Irishman: cool desaturated digital grading
- The Sopranos: naturalistic TV lighting

**Insight → choice**: Films have inconsistent palettes → random sampling biases toward the dominant film → DINO diversity corrects by ensuring coverage across all visual styles.

#### B. Blur Quality Spectrum ⭐

Grid of 12–15 patches ordered by Laplacian variance (lowest → highest) with value overlaid. Mark the cutoff.

**Insight → choice**: Clear visual cliff at threshold N → empirically grounded quality floor, not arbitrary.

#### C. UMAP Embedding Visualisation ⭐

Two side-by-side scatters (DINO features):
1. Coloured by domain (game/movie) — shows separation
2. Coloured by source film — shows sub-clusters within movie domain

**Insight → choice**: Movie fragments into per-film clusters → naive random sampling over-represents the densest cluster → k-means per-cluster sampling ensures representation.

#### D. Patch Scale Distribution

Bbox area histograms per source. Game = more distant wide shots; movie = more close-ups.

**Insight → choice**: Extreme size outliers (very small patches upscaled to 256×256) degrade CUT training → motivate minimum bbox area filter.

#### E. Pairwise Distance Before/After Selection

Box plot of mean pairwise cosine distance: random 450 vs DINO-selected 450.

**Insight → choice**: Quantitative proof that selection increases diversity.

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
) -> dict                     # stats + paths to generated plots
```

### Prerequisite

Re-extract MafiaVideogame with higher `n2save` (see 1.1 improvements).

### Output

```
output/selected_for_training/
    game/       # ~450 patches
    movie/      # ~450 patches
    _selection_summary.txt
    umap_plot.png
    color_distributions.png
```

---

## 2.1 — Image Model Deployment (20%)

### Goal

Train an unpaired image-to-image model on patches from 1.3. **Apply to full frames** (not human patches). Evaluate with FID + SSIM. Show 10 success + 10 failure cases per direction.

> [!IMPORTANT]
> 2.1 applies the model to full frames. The distribution mismatch (trained on cropped human patches, applied to full 1280×720 frames) is a **known limitation** — document it as motivation for 2.2's local approach.

### Model: CUT (Contrastive Unpaired Translation)

The brief says *"any unpaired image-to-image network (e.g., CycleGAN)"*. We use **CUT** — same junyanz repo, better performance on limited data (~450 patches/domain), faster training.

| | CycleGAN | CUT |
|---|---|---|
| Training | Two generators + two discriminators | One generator + PatchNCE loss |
| Speed | ~3–6 hrs | ~2–3 hrs |
| VRAM | ~2 GB | ~1.5 GB |
| Small datasets | Mode collapse risk | Designed for limited data |
| Codebase | `junyanz/pytorch-CycleGAN-and-pix2pix` | **Same repo** — `--model cut` |

Mention CycleGAN as the canonical alternative in the report; justify CUT as better suited to the dataset size.

### Training Config

```bash
cd external/cyclegan
python train.py --dataroot datasets/game2movie \
    --name game2movie --model cut \
    --CUT_mode CUT \
    --n_epochs 50 --n_epochs_decay 50 \
    --load_size 286 --crop_size 256 \
    --batch_size 1 --no_dropout \
    --save_epoch_freq 10 --gpu_ids 0
```

### Normalisation Pipeline

The junyanz repo normalises inputs to `[-1, 1]`. These helpers go in `src/style_transfer.py` and are reused everywhere (2.1 eval, 2.2 video pipeline, notebook visualisation):

```python
def preprocess_for_model(bgr_uint8) -> torch.Tensor:     # BGR uint8 → [-1,1] RGB NCHW
def postprocess_from_model(tensor) -> np.ndarray:         # [-1,1] RGB → BGR uint8
```

### Loading the Generator for Inference

```python
def load_generator(checkpoint_path, device) -> torch.nn.Module:
    # Uses junyanz repo's define_G to build architecture matching training config
    # Loads state dict from checkpoint_path
```

CUT saves `latest_net_G.pth` (single generator, game→movie direction). For the reverse direction (movie→game), train a second CUT model with `--direction BtoA`, or accept that 2.1 only evaluates one direction with full quantitative metrics.

### Evaluation

- **FID** (`clean-fid`): `FID(generated_game→movie, real_movie)`. With ~450 images, note high variance in report.
- **SSIM** (`skimage`): input vs translated (structure preservation, not style quality)
- **10 success + 10 failure pairs per direction** — select failures by lowest SSIM or visual inspection (face artifacts, background bleed, mode collapse)

### 2.1 Application to Full Frames

Sample full frames from the training videos (the same frames selected in 1.1, before cropping):
- Read original frame at stored `frame_num`
- Resize to CUT input size → translate → save
- Document artifacts: background contamination, inconsistent style on non-human regions

### Implementation

**New file**: `src/style_transfer.py`

```python
prepare_data(selected_dir, data_dir) -> None          # symlinks trainA/trainB
load_generator(checkpoint_path, device) -> nn.Module
preprocess_for_model(bgr_uint8) -> torch.Tensor
postprocess_from_model(tensor) -> np.ndarray
evaluate_model(checkpoint_dir, test_dir) -> dict       # FID, SSIM
apply_to_full_frames(generator, frame_paths, output_dir, device) -> None
```

**New file**: `scripts/train_cut.sh` (Slurm wrapper)

### Output

```
external/cyclegan/checkpoints/game2movie/
output/cut_results/
    game_to_movie/          # translated full frames
    fid_scores.txt
    ssim_scores.txt
    success_pairs/          # 10 best
    failure_pairs/          # 10 worst
```

---

## 2.2 — Local Temporal Enhancement (30%)

### Goal

Improve on 2.1 by: (1) applying CUT to human patches only (local, using 1.1–1.3), and (2) adding temporal consistency. Compare quantitatively to 2.1. This section carries the most marks.

The brief offers two paths:
1. *"Local information from 1.1–1.3"*
2. *"Advanced temporal approach — extra credit"*

**We do both.**

### Stage 1 — Local Patch Application

Re-run YOLOv8-pose on `Test.mp4` → crop human patches → CUT → composite back onto original frame.

- **Resize**: crop → 256×256 → CUT → resize back to original bbox dims (bilinear)
- **Multiple humans**: process all detections per frame, largest-area first
- **Background unchanged**: only human patches get style-transferred (the explicit improvement over 2.1)

**Compositing** — two modes, selected per-detection:
1. **`cv2.seamlessClone`** (`NORMAL_CLONE`): best quality, but bbox must be ≥5px from frame border
2. **Gaussian feathering**: alpha blend with soft mask (~10px border taper). Fallback for edge-case bboxes

### Stage 2 — Temporal Consistency

**Primary method: EMA patch blending** (simple, always works):

```
translated[t] = β * CUT(crop[t]) + (1-β) * translated[t-1]
```

β derived from IoU(bbox[t], bbox[t-1]). Low overlap (new person / fast motion) → β≈1 (no blending). High overlap (stationary) → temporal smoothing.

**Advanced method: RAFT optical flow** (extra credit):

```
output[t] = α * stylized[t] + (1-α) * warp(output[t-1], flow[t-1→t])
```

- `torchvision.models.optical_flow.raft_small` (pretrained, ~1 GB VRAM)
- α=0.7, scene cuts (reuse `frame_difference` from 1.1) → reset buffer
- Warp via `torch.nn.functional.grid_sample`

**Build order**: EMA first → working video → add RAFT on top. Always have a submittable output.

### VRAM Budget (RTX 2080 Ti — 11 GB)

| Model | VRAM |
|---|---|
| YOLOv8m (FP16) | ~0.8 GB |
| CUT generator | ~0.5 GB |
| RAFT small | ~1.0 GB |
| CUDA overhead + activations | ~3–5 GB |
| **Total** | **~5–7 GB** ✅ |

### Quantitative Comparison: 2.1 vs 2.2

- **Warping error**: avg pixel diff between warped stylised frame and next stylised frame. Lower = more temporally consistent.
- **FID on patches**: compare FID of CUT-translated patches (2.2) vs full-frame crops (2.1) against real movie patches.
- Visual comparison: 10+ keyframe triplets (original → 2.1 full-frame → 2.2 local+temporal).

### Implementation

**New file**: `src/video_pipeline.py`

```python
apply_local_style_transfer(
    model, yolo_model, video_path, output_path,
    device, blend_mode='poisson',
) -> None

apply_temporal_style_transfer(
    model, yolo_model, flow_model,   # RAFT (None for EMA-only)
    video_path, output_path, device,
    alpha=0.7, scene_change_threshold=30.0,
) -> None
```

**New file**: `scripts/run_test_video.sh` (Slurm wrapper)

### Output

```
output/
    test_baseline_full_frame.mp4   # 2.1 applied to full frames (for comparison)
    test_local.mp4                 # 2.2 Stage 1: local-only
    test_temporal.mp4              # 2.2 Stage 2: local + temporal
    test_keyframes/                # 10+ comparison frames
```

---

## New Files Summary

| File | Purpose |
|---|---|
| `src/data_selection.py` | 1.3 — DINO features, k-means, balanced sampling, insight plots |
| `src/style_transfer.py` | 2.1 — CUT data prep, normalisation, generator loading, evaluation |
| `src/video_pipeline.py` | 2.2 — Local patch application, EMA + RAFT temporal blending |
| `scripts/train_cut.sh` | Slurm job for CUT training |
| `scripts/run_test_video.sh` | Slurm job for test video processing |

---

## Execution Order

```
 1. Re-extract MafiaVideogame with higher n2save; adjust per-video targets
 2. (Optional) Test classification with face_detector=None → decide MediaPipe
 3. Run 1.3 data selection → output/selected_for_training/{game,movie}/
 4. Generate insight plots for report (color distributions, UMAP, blur spectrum)
 5. Clone junyanz/pytorch-CycleGAN-and-pix2pix → external/cyclegan/
 6. Prepare CUT data (symlinks trainA/trainB)
 7. Train CUT (Slurm, ~2–3 hours)
 8. Evaluate CUT: FID + SSIM + success/failure pairs on full frames
 9. Build local style transfer pipeline (2.2 Stage 1)
10. Generate test_local.mp4
11. Add temporal blending — EMA first, then RAFT if time permits
12. Generate test_temporal.mp4
13. Extract keyframes + compute warping error for report
```

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| CUT doesn't converge | Monitor loss; extend to 200 epochs or try `--CUT_mode FastCUT` |
| Too few game patches | Re-extract MafiaVideogame with n2save=1500 |
| RAFT + CUT exceeds 11 GB | Use `raft_small`; if OOM, run sequentially with `torch.cuda.empty_cache()` |
| `seamlessClone` fails at frame edges | Fall back to Gaussian feathering per-detection |
| Temporal blending ghosting | Increase α; disable blending when flow magnitude exceeds threshold |
| RAFT too complex / slow | EMA is primary method; RAFT is additive extra credit |

---

## Report Figures

1. **1.3**: Per-film HSV violins, blur spectrum grid, UMAP scatter (domain + film-coloured)
2. **2.1**: Input/output grid, FID over epochs, 10 success + 10 failure pairs, full-frame limitation discussion
3. **2.2**: Keyframe triplets (original → 2.1 → 2.2), warping error plot, seam boundary zoom
