# Implementation Plan — Full Pipeline

## Current State

| Stage | Status | Key files |
|---|---|---|
| 1.1 — Patch Extraction | ✅ Done | `src/feat_extract.py`, `nb_main.py` |
| 1.2 — Classification | 🔄 Reworking → GCN | `src/classification.py` (pseudo-labels), `src/pose_gcn.py` (new) |
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

## 1.2 — Classification (15%) 🔄

### Method: GCN on Skeleton Graph

Represent each person's pose as a **graph** and classify with a small **Graph Convolutional Network**. This replaces the previous hand-crafted rule-based classifier with a learned approach that exploits skeletal topology.

**Transfer learning framing**: YOLOv8-pose (pretrained CNN, frozen) extracts 17 keypoints → GCN (trained from scratch) classifies pose type. The pretrained detector is the feature extractor; the GCN is the task-specific head.

#### Graph Construction

- **17 nodes** — one per COCO keypoint
- **~16 edges** — following the anatomical skeleton (nose↔L_eye, L_hip↔L_knee, etc.)
- **Node features**: `(x/w, y/h, confidence)` — normalised to bounding box for translation/scale invariance
- **Adjacency matrix `A`** — fixed, binary, symmetric; same for every sample

```
        nose
       / | \
    L_eye  R_eye
      |      |
   L_ear   R_ear
       \   /
    L_shoulder─R_shoulder
      |           |
    L_elbow    R_elbow
      |           |
    L_wrist    R_wrist
      |           |
    L_hip───── R_hip
      |           |
    L_knee     R_knee
      |           |
    L_ankle    R_ankle
```

#### GCN Architecture

```
Input: (17, 3)  →  GCNConv(3→64) + ReLU + Dropout
                →  GCNConv(64→128) + ReLU + Dropout
                →  GCNConv(128→64) + ReLU
                →  Global Mean Pool  →  (64,)
                →  Linear(64→5)      →  softmax
```

~10K parameters. Trains in seconds on CPU. Each GCN layer aggregates neighbour features: after 2–3 hops, every joint "sees" the full skeleton.

#### Training Labels: Pseudo-Label Strategy

No hand-labelled ground truth exists. Strategy:

1. **Pseudo-labels**: Run existing rule-based classifier (`classification.py`) on all ~4000 patches → noisy labels
2. **Train GCN** on pseudo-labels (80/20 split, ~30s on CPU)
3. **Hand-label ~150 patches** (30/class) as a held-out validation set
4. **Evaluate**: Compare GCN vs rules on the hand-labelled set

The GCN is not circular — it _generalises_ from the heuristic patterns and learns to handle the edge cases where rigid rules fail. Report discusses patches where GCN and rules disagree.

#### Existing Rule-Based Classifier (Retained for Pseudo-Labels)

[classification.py](file:///Users/Subspace_Explorer/Projects/ACV_cswk/src/classification.py) — evidence-weighted rules over YOLOv8-pose keypoints.

- **Body extent**: ankle/knee visibility → full body; shoulder/upper-body points → head-shoulder; aspect-ratio veto
- **Orientation**: face keypoint patterns, nose-between-shoulders, ear evidence, shoulder/hip width ratio
- **Adaptive threshold**: per-patch confidence floor from median keypoint confidence

These rules become the pseudo-label generator. MediaPipe dependency is removed (YOLO keypoints sufficient for pseudo-labels; GCN learns finer distinctions).

### Report Notes (100 words max)

Mention: YOLOv8-pose as pretrained feature extractor (transfer learning), skeleton represented as graph (17 nodes, COCO topology), GCN learns pose classification from graph structure. Pseudo-labels from geometric rules bootstrap training; hand-labelled validation set evaluates. Show confusion matrix (GCN vs rules vs hand-labels), discuss disagreement cases. Justify GCN over rules: learns implicit spatial relationships (joint angles, relative positions) not captured by threshold-based heuristics.

---

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

---

## Unifying Narrative: Skeleton Graph Throughout

The skeleton graph representation threads through the entire pipeline:

```
1.1  YOLOv8-pose → 17 keypoints per detection
        ↓
1.2  Skeleton graph → GCN → pose classification (5 classes)
        ↓
1.3  GCN embeddings + DINO features → joint diversity selection
        ↓
2.2  Skeleton similarity across frames → temporal blending weight
```

This is the **core technical contribution** to emphasise in the report: a single geometric representation (skeleton graph) unifies feature extraction, classification, data curation, and temporal consistency.
