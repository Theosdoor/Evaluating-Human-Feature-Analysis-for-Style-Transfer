---
description: "ACV pipeline API reference — function signatures, architecture details, and nb_main.py cell structure. Load when editing src/, scripts/, or nb_main.py."
applyTo: "src/**,scripts/**,nb_main.py"
---

# Pipeline API Reference

## Stage 1.1 — Extraction (`src/feat_extract.py`)

**Frame filtering before YOLO**:
1. Inter-frame diff < `scene_change_threshold=8.0` → skip
2. Enforce `yolo_interval=10` minimum gap
3. Buffer `yolo_batch_size=8` eligible frames → single GPU call via `_process_batch`
4. `cap.grab()` (no decode) during gap skip

**Film vs game blur threshold** — inferred from filename keywords (`movie`, `godfather`, `irishman`, `sopranos`):
- Film: `blur_threshold_film=40.0` | Game: `blur_threshold_game=100.0`

**Detection dict keys**: `video_path`, `frame_num`, `confidence`, `bbox` (x1,y1,x2,y2), `area`, `relative_area`, `blur_score`, `blur_thresh`, `centering`, `patch` (cached crop), `score`.

**Patch filename**: `human_{idx:04d}_{source_tag}_f{frame_num:06d}_conf{conf:.2f}_score{score:.2f}.jpg` (JPEG quality 92)

```python
frame_difference(prev_gray, curr_gray) -> float
compute_blur_score(patch) -> float  # Laplacian variance; higher = sharper
_process_batch(model, frame_batch, frame_meta, img_area, blur_thresh, video_path) -> list[dict]

extract_humans_from_video(
    model, video_path,
    yolo_interval=10, scene_change_threshold=8.0,
    blur_threshold_film=40.0, blur_threshold_game=100.0,
    yolo_batch_size=8, on_batch=None,
) -> list[dict]

score_detection(det) -> float
    # conf×0.30 + relative_area×0.25 + blur×0.20 + centering×0.15 + aspect×0.10

diverse_sampling(detections, target_count=1000, temporal_gap=10) -> list[dict]
    # Greedy O(n); detections must be pre-sorted by score descending

save_patches(detections, output_dir, io_workers=4) -> int
    # Fast path: cached 'patch' key; fallback: sequential video re-read

save_extraction_summary(extract_save_path, detections, selected_detections) -> str
reload_extracted_patches(extract_save_path, train_paths) -> list[dict]
```

---

## Stage 1.2 — Classification (`src/classification.py`)

**5 classes**: `full_body_front`, `full_body_back`, `head_shoulder_front`, `head_shoulder_back`, `others`

**Two-step**: `classify_body_extent` → `classify_orientation`. Either returns None → `others`.

**`classify_orientation` scoring** (`front_score` / `back_score`):
- MediaPipe fired + YOLO face kps: +3.0 front | MediaPipe alone: +1.0
- YOLO n_face ≥ 2: +3.0 | n_face == 1: +1.2 | n_face_low ≥ 1: +0.5
- No face (n_kp_any ≥ 6): +2.5 back
- Nose between shoulders: +1.0 front | ears w/o frontal features: +1.2×count back
- Shoulder/hip width ratio > 1.15: +0.3 front
- Ambiguity margin < 0.12 → `others`; front override if `front_score > 2.5`

**Adaptive conf**: `adaptive_conf_high` = median(kp_conf) − 0.1, clipped [0.15, 0.30].  
**Constants**: `CONF_HIGH = 0.35`, `CONF_LOW = 0.15`

```python
adaptive_conf_high(keypoints) -> float
build_face_detector(min_confidence=0.5, model_path=None) -> FaceDetector | None
_run_face_detection(face_detector, image_path, bbox=None) -> bool

kp(keypoints, idx, threshold=CONF_HIGH) -> (x, y, conf)  # (0,0,0) if below threshold
is_visible(keypoints, idx, threshold=CONF_HIGH) -> bool
count_visible(keypoints, indices, threshold=CONF_HIGH) -> int

classify_body_extent(keypoints, conf_high=None, bbox=None) -> 'full_body' | 'head_shoulder' | None
    # Aspect-ratio veto: h/w < 1.5 downgrades full_body → head_shoulder
classify_orientation(keypoints, conf_high=None, face_detected=False) -> 'front' | 'back' | None
classify_orientation_debug(keypoints, conf_high=None, face_detected=False) -> (orientation, trace_dict)
classify_keypoints(keypoints, face_detected=False, bbox=None) -> str  # one of CLASSES

classify_patch(pose_model, image_path, face_detector=None) -> str
classify_directory(
    pose_model, input_dir, output_dir,
    batch_size=32, copy_files=True, save_debug_viz=False, face_detector=None,
) -> (results: dict[str,str], summary: dict[str,int])
reload_classification_results(cls_save_path) -> (results: dict[str,str], summary: dict[str,int])
```

### COCO Keypoint Indices (17 points)
```
 0: nose          1: left_eye      2: right_eye
 3: left_ear      4: right_ear     5: left_shoulder
 6: right_shoulder  7: left_elbow  8: right_elbow
 9: left_wrist   10: right_wrist  11: left_hip
12: right_hip    13: left_knee    14: right_knee
15: left_ankle   16: right_ankle
```

---

## Stage 1.3 — Data Selection (`src/data_selection.py`, planned)

```python
select_training_data(
    classifications_dir, output_dir,
    min_score=0.6, n_clusters=50, target_per_domain=450, device='cuda',
) -> dict
```

DINO model: `timm.create_model('vit_small_patch8_224.dino', pretrained=True, num_classes=0)` → 384-dim CLS token.

---

## Stage 2.1 — Style Transfer (`src/style_transfer.py`, planned)

```python
preprocess_for_cyclegan(bgr_uint8: np.ndarray) -> torch.Tensor   # BGR uint8 → [-1,1] NCHW float32
postprocess_from_cyclegan(tensor: torch.Tensor) -> np.ndarray    # [-1,1] NCHW → BGR uint8
prepare_cyclegan_data(selected_dir, cyclegan_data_dir) -> None   # symlinks trainA/trainB
load_cyclegan_generator(checkpoint_path, device) -> torch.nn.Module
evaluate_cyclegan(checkpoint_dir, test_dir, direction) -> dict   # FID, per-image SSIM
apply_to_full_frames(generator, frame_paths, output_dir, device) -> None
```

CycleGAN training flags: `--model cycle_gan --n_epochs 50 --n_epochs_decay 50 --load_size 286 --crop_size 256 --batch_size 1`

---

## Stage 2.2 — Video Pipeline (`src/video_pipeline.py`, planned)

```python
apply_local_style_transfer(
    cyclegan_model, yolo_model, video_path, output_path, device,
    blend_mode='poisson',
) -> None

apply_temporal_style_transfer(
    cyclegan_model, yolo_model,
    flow_model,             # RAFT (torchvision raft_small); None for EMA-only
    video_path, output_path, device,
    alpha=0.7, scene_change_threshold=30.0, blend_mode='poisson',
) -> None
```

**Compositing**: `cv2.seamlessClone` (NORMAL_CLONE) when bbox ≥5px from frame edge; Gaussian feathering fallback.  
**Temporal**: EMA (`β = IoU(bbox[t], bbox[t-1])`) is primary; RAFT optical flow + `grid_sample` warp is extra credit.  
**VRAM budget** (2080 Ti 11 GB): YOLO FP16 ~0.8 GB, CycleGAN ~0.5 GB, RAFT small ~1.0 GB — use `torch.no_grad()` throughout.

---

## `nb_main.py` Cell Structure

| Cell | Content |
|---|---|
| 1 | Imports, Colab detection, `sys.path` |
| 2 | Constants: `DATA_DIR`, `TRAIN_PATHS`, `TEST_PATH`, `SAVE_DIR`, `SAVE_NAME`, `RELOAD_RUN`, `RECLASSIFY`, `DEVICE` |
| 3 | **Stage 1.1**: extract or reload. Per-video targets via `_video_targets`. Calls `extract_humans_from_video` → `score_detection` → sort → `diverse_sampling` → `save_patches` |
| 4 | `save_extraction_summary(...)` |
| 5 | **Stage 1.2**: classify or reload. `classify_directory(..., batch_size=32, copy_files=True, save_debug_viz=True)` |
| 6 | Patch viewer: `show_patch_debug(rel_path)` — side-by-side patch + debug_viz |
| 7 | Deep diagnostic: 4-panel figure via `classify_orientation_debug`. Uses `DIAG_RUN`, `DIAG_CLASS`, `DIAG_IMAGE` |

```python
RELOAD_RUN = None    # timestamp string to skip extraction/classification
RECLASSIFY = True    # re-run classification even when RELOAD_RUN is set
n2save = 4000
detection_b_size = 32
classify_b_size = 32
```

---

## `scripts/ablate_mediapipe.py`

CLI ablation: classify same patches with vs without MediaPipe.

```python
parse_args()  # --input-dir, --output-dir, --pose-model, --face-model-path, --batch-size=32, --device=auto
run_ablation(model, image_paths, batch_size, face_detector) -> list[dict]
    # fields: filename, label_no_mediapipe, label_with_mediapipe, changed, face_detected
```

Outputs: `per_image_results.csv`, `summary.json`, `summary.txt`

## `scripts/experiments.py`

Diagnostic viewer only — `show_patch_debug(rel_path, root=PROJECT_ROOT)` and 4-panel classification figure. Not for production.
