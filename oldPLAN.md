# Implementation Plan — Full Pipeline


## 1.3 — Training Data Selection (15%)

### Goal

Select high-quality, diverse patches for CUT training. The brief requires **original observations and insights** — the method must be motivated by empirical analysis, not just applied off-the-shelf.

### Method: Quality Filter → Joint Diversity Selection

1. **Quality floor**: Drop patches with extraction score < 0.6 (~trims worst 10%)
2. **Dual-space diversity**: Concatenate two complementary feature vectors per patch:
   - **DINO** (`vit_small_patch8_224.dino` via `timm`) → 384-dim CLS token — **appearance** diversity (colour, texture, framing)
   - **GCN graph embedding** (from 1.2, penultimate layer) → 64-dim — **pose** diversity (body configuration, orientation)
   - L2-normalise each, concatenate → 448-dim joint vector
3. **k-means** on joint vectors per domain → stratified sample from clusters by score
4. **Domain balance**: target **~450 patches per domain**

### Key Insight: Appearance ≠ Pose Diversity

> [!IMPORTANT]
> Two patches can look visually similar (same lighting, same film) but have very different poses, and vice versa. Selecting on DINO alone misses pose diversity; selecting on pose alone misses visual diversity. Joint selection ensures coverage in both spaces.

Visualize with UMAP: colour by pose class vs. DINO cluster → they won't align, proving complementarity.

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

#### C. UMAP Embedding Visualisation — Dual Space ⭐⭐

Three side-by-side scatters:
1. DINO features, coloured by **pose class** (from GCN) — pose classes scatter across DINO space
2. GCN embeddings, coloured by **source film** — films mix in pose space
3. Joint (DINO+GCN) features, coloured by **k-means cluster** — clusters capture both dimensions

**Insight → choice**: Neither feature space alone captures full diversity → joint selection is necessary.

#### D. Pose-Class Balance Before/After Selection

Bar chart: pose-class distribution in raw pool vs. DINO-only selection vs. joint selection.

**Insight → choice**: DINO-only under-represents rare poses (e.g., full-body back); joint selection re-balances.

### Implementation

**New file**: `src/data_selection.py`

```python
select_training_data(
    classifications_dir,       # output/classifications/<timestamp>
    gcn_model,                 # trained PoseGCN from 1.2
    output_dir,                # output/selected_for_training/
    min_score=0.6,
    n_clusters=50,             # per domain
    target_per_domain=450,
    device='cuda',
) -> dict                     # stats + paths to generated plots
```

### Prerequisite

Re-extract MafiaVideogame with higher `n2save` (see 1.1 improvements). Train GCN (1.2) first.

### Output

```
output/selected_for_training/
    game/       # ~450 patches
    movie/      # ~450 patches
    _selection_summary.txt
    umap_dual_space.png
    color_distributions.png
    pose_balance.png
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

**We do both.** The skeleton graph from 1.2 threads through: classification → data selection → temporal consistency.

### Stage 1 — Local Patch Application

Re-run YOLOv8-pose on `Test.mp4` → crop human patches → CUT → composite back onto original frame.

- **Resize**: crop → 256×256 → CUT → resize back to original bbox dims (bilinear)
- **Multiple humans**: process all detections per frame, largest-area first
- **Background unchanged**: only human patches get style-transferred (the explicit improvement over 2.1)

**Compositing** — two modes, selected per-detection:
1. **`cv2.seamlessClone`** (`NORMAL_CLONE`): best quality, but bbox must be ≥5px from frame border
2. **Gaussian feathering**: alpha blend with soft mask (~10px border taper). Fallback for edge-case bboxes

### Stage 2 — Skeleton-Graph Temporal Consistency

Use the skeleton graph representation from 1.2 to guide temporal blending — a **semantically-aware** alternative to pixel-level optical flow.

#### Core Idea

YOLOv8-pose already runs per-frame (for patch extraction). The 17 keypoints form a skeleton graph that captures **how the person is moving**. Compare skeletons across consecutive frames to decide blending strength:

```
Frame t-1:  skeleton G_{t-1}  ──┐
                                ├── pose_similarity → blending weight β
Frame t:    skeleton G_t      ──┘

output[t] = β · CUT(crop[t]) + (1-β) · output[t-1]
```

#### Two Similarity Metrics (compare in ablation)

**A. Joint displacement** (simple, fast):
```python
# Mean normalised Euclidean distance between matched keypoints
# Only count keypoints visible in both frames (conf > threshold)
dist = mean(||kp_t[i] - kp_{t-1}[i]||_2 for i in visible_both)
# Normalise by bbox diagonal for scale invariance
similarity = 1 - clamp(dist / bbox_diag, 0, 1)
```

**B. Graph embedding distance** (richer, uses trained GCN):
```python
# Pass both skeletons through GCN from 1.2 (penultimate layer → 64-dim)
emb_t   = gcn.embed(graph_t)      # (64,)
emb_t1  = gcn.embed(graph_{t-1})   # (64,)
similarity = cosine_similarity(emb_t, emb_t1)
```

Method B captures **structural** similarity — two poses can have similar joint positions but different body configurations (e.g., arms crossed vs. arms at sides). The GCN embedding encodes topology-aware features.

#### Blending Logic

```python
if similarity > 0.9:    # near-identical pose → heavy smoothing
    β = 0.3
elif similarity < 0.3:  # new person / scene cut → fresh output
    β = 1.0
else:                   # proportional
    β = lerp(0.3, 1.0, 1 - similarity)
```

Scene cuts (reuse `frame_difference` from 1.1) → reset buffer entirely (β=1.0, flush history).

#### Comparison with RAFT Optical Flow

| | Skeleton-graph temporal | RAFT optical flow |
|---|---|---|
| **Signal** | 17 joint positions (sparse, semantic) | Dense pixel-level flow field |
| **Captures** | Body pose changes, gait | All motion (clothing, hair, background) |
| **Cost** | ~Free (reuses YOLO keypoints) | ~1 GB VRAM, ~0.1s/frame |
| **Warping** | No — guides blending weight only | Yes — warps previous frame pixel-by-pixel |
| **Failure mode** | Misses fine-grained motion (hair, fabric) | Expensive; flow errors cause ghosting |
| **Report value** | Novel, ties whole pipeline together | Well-known off-the-shelf method |

RAFT is more powerful for **pixel-precise** temporal consistency (it can warp frames), but the skeleton approach is:
- Cheaper (no extra model)
- Semantically meaningful ("the person moved their arm" vs. "pixels shifted")
- A **unifying narrative** across 1.2→1.3→2.2

**Build order**: Skeleton-graph temporal first (primary method) → optionally add RAFT for pixel-level warping comparison if time permits.

### VRAM Budget (RTX 2080 Ti — 11 GB)

| Model | VRAM |
|---|---|
| YOLOv8m (FP16) | ~0.8 GB |
| CUT generator | ~0.5 GB |
| PoseGCN (tiny) | ~0.01 GB |
| CUDA overhead + activations | ~3–5 GB |
| **Total** | **~4–6 GB** ✅ |

(RAFT adds ~1 GB if used as optional comparison.)

### Quantitative Comparison: 2.1 vs 2.2

- **Temporal flicker metric**: mean absolute pixel diff between consecutive stylised frames (lower = more temporally consistent)
- **FID on patches**: CUT-translated patches (2.2) vs full-frame crops (2.1) against real movie patches
- **Ablation**: no temporal → EMA+IoU → skeleton-joint → skeleton-GCN (→ RAFT if time)
- Visual comparison: 10+ keyframe triplets (original → 2.1 full-frame → 2.2 local+temporal)

### Implementation

**New file**: `src/video_pipeline.py`

```python
def skeleton_similarity(
    kps_prev, kps_curr, gcn_model=None, method='joint_displacement'
) -> float:
    """Compute pose similarity between two frames' skeletons."""

def apply_local_style_transfer(
    model, yolo_model, video_path, output_path,
    device, blend_mode='poisson',
) -> None:
    """Stage 1: local patch CUT, no temporal blending."""

def apply_temporal_style_transfer(
    model, yolo_model, video_path, output_path, device,
    gcn_model=None,                  # PoseGCN for embedding similarity
    temporal_method='skeleton_gcn',  # 'ema_iou' | 'skeleton_joint' | 'skeleton_gcn' | 'raft'
    scene_change_threshold=30.0,
) -> None:
    """Stage 2: local patch CUT + temporal blending."""
```

**New file**: `scripts/run_test_video.sh` (Slurm wrapper)

### Output

```
output/
    test_baseline_full_frame.mp4   # 2.1 applied to full frames (for comparison)
    test_local.mp4                 # 2.2 Stage 1: local-only
    test_temporal.mp4              # 2.2 Stage 2: local + skeleton temporal
    test_keyframes/                # 10+ comparison frames
    temporal_ablation/             # flicker metrics per method
```

---

## New Files Summary

| File | Purpose |
|---|---|
| `src/pose_gcn.py` | 1.2 — Graph construction, GCN model, training, embedding extraction |
| `src/data_selection.py` | 1.3 — DINO + GCN features, k-means, balanced sampling, insight plots |
| `src/style_transfer.py` | 2.1 — CUT data prep, normalisation, generator loading, evaluation |
| `src/video_pipeline.py` | 2.2 — Local patch application, skeleton-graph temporal blending |
| `scripts/train_cut.sh` | Slurm job for CUT training |
| `scripts/run_test_video.sh` | Slurm job for test video processing |

---

## Execution Order

```
 1. Re-extract MafiaVideogame with higher n2save; adjust per-video targets
 2. Generate pseudo-labels with rule-based classifier → bootstrap labels
 3. Train PoseGCN on pseudo-labels (~30s CPU) → src/pose_gcn.py
 4. Hand-label ~150 patches → evaluate GCN vs rules
 5. Run 1.3 data selection (DINO + GCN joint features) → output/selected_for_training/
 6. Generate insight plots (color distrib, dual-space UMAP, pose balance)
 7. Clone junyanz/pytorch-CycleGAN-and-pix2pix → external/cyclegan/
 8. Prepare CUT data (symlinks trainA/trainB)
 9. Train CUT (Slurm, ~2–3 hours)
10. Evaluate CUT: FID + SSIM + success/failure pairs on full frames
11. Build local style transfer pipeline (2.2 Stage 1)
12. Generate test_local.mp4
13. Add skeleton-graph temporal blending (2.2 Stage 2)
14. Generate test_temporal.mp4
15. Temporal ablation: no-temporal → EMA+IoU → skeleton-joint → skeleton-GCN
16. Extract keyframes + compute flicker metrics for report
```

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| CUT doesn't converge | Monitor loss; extend to 200 epochs or try `--CUT_mode FastCUT` |
| Too few game patches | Re-extract MafiaVideogame with n2save=1500 |
| Pseudo-labels too noisy for GCN | Hand-label 150 patches for validation; clean obvious misclassifications |
| GCN overfits on tiny dataset | Heavy dropout (0.3), weight decay; model is only ~10K params |
| `seamlessClone` fails at frame edges | Fall back to Gaussian feathering per-detection |
| Skeleton temporal misses fine motion | Acknowledge as limitation; RAFT is optional pixel-level comparison |
| Keypoint not detected in frame | Fall back to IoU-based EMA when <5 keypoints visible |

---

## Report Figures

1. **1.2**: Skeleton graph visualisation, GCN architecture diagram, confusion matrix (GCN vs rules vs hand-labels)
2. **1.3**: Dual-space UMAP (DINO vs GCN vs joint), per-film HSV violins, pose-class balance bar chart
3. **2.1**: Input/output grid, FID over epochs, 10 success + 10 failure pairs, full-frame limitation discussion
4. **2.2**: Keyframe triplets (original → 2.1 → 2.2), temporal ablation flicker plot, seam boundary zoom
