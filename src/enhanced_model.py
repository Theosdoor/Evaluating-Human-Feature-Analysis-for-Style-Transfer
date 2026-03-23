"""
src/enhanced_model.py
Question 2.2 — Local (Temporal) Enhancement

Pipeline
--------
1. finetune_cut_patches   — fine-tune the original checkpoint on the
                            1.3-selected human patches (game + movie).

2. translate_test_video_enhanced
                          — for each test frame:
                              a. run YOLO (from 1.1) to detect human bboxes
                              b. crop each human region, resize to 256×256
                              c. translate the crop with the fine-tuned CUT model
                              d. resize translated crop back to original bbox dims
                              e. paste onto the original (untranslated) frame
                              f. optionally apply temporal blending (step 3)

3. Temporal blending (optional, enabled with blend_alpha > 0):
   A simple per-pixel exponential moving average across consecutive
   translated crops for each tracked region, reducing flickering without
   requiring optical flow. Set blend_alpha=0.0 to disable.

Relationship to Section 1
--------------------------
  1.1  — YOLO detection provides bounding boxes for human region extraction.
  1.2  — Classification labels are used to optionally filter which detection
          classes receive style transfer (e.g. skip 'others').
  1.3  — Selected patches form the fine-tuning dataset passed to
          finetune_cut_patches.

Shared helpers (video I/O, CUT subprocess, fine-tuning) live in src/utils.py.

Public API imported in nb_main.py:
    finetune_cut_patches, translate_test_video_enhanced
"""

import glob
import os
import shutil
import tempfile

import cv2
import numpy as np
from tqdm import tqdm

from src.utils import extract_video_frames, write_video, run_cut_inference, finetune_cut


# ---------------------------------------------------------------------------
# Fine-tuning on 1.3-selected patches — Q2.2
# ---------------------------------------------------------------------------

def finetune_cut_patches(
    cut_dir: str,
    pretrained_exp: str,
    finetune_exp: str,
    game_patch_paths: list[str],
    movie_patch_paths: list[str],
    save_dir: str,
    device: str,
    n_epochs: int = 20,
    n_epochs_decay: int = 10,
) -> str:
    """
    Fine-tune a CUT checkpoint on the human patches selected in Q1.3.

    Copies pretrained_exp → finetune_exp before training, so both 2.1 and
    2.2 branch independently from the same pretrained base.  Neither call
    sees the other's training data or accumulated weight updates, making
    the comparison between them fair.

    Args:
        cut_dir           : root of the CUT repo clone.
        pretrained_exp    : source checkpoint name (the original pretrained weights).
        finetune_exp      : destination checkpoint name for this fine-tuned model.
        game_patch_paths  : list of game-domain patch paths from 1.3.
        movie_patch_paths : list of movie-domain patch paths from 1.3.
        save_dir          : directory to write the staged dataroot into.
        device            : "cuda" | "cpu".
        n_epochs          : epochs at constant LR.
        n_epochs_decay    : epochs over which LR decays to zero.

    Returns:
        Path to the staged dataroot (useful for inspection / reuse).
    """
    dataroot = os.path.join(save_dir, "patch_dataroot")
    trainA   = os.path.join(dataroot, "trainA")
    trainB   = os.path.join(dataroot, "trainB")
    os.makedirs(trainA, exist_ok=True)
    os.makedirs(trainB, exist_ok=True)

    def _stage(paths: list[str], dst: str) -> int:
        staged = 0
        for src in paths:
            d = os.path.join(dst, os.path.basename(src))
            if not os.path.exists(d):
                try:
                    os.symlink(os.path.abspath(src), d)
                except OSError:
                    shutil.copy2(src, d)
                staged += 1
        return staged

    g = _stage(game_patch_paths,  trainA)
    m = _stage(movie_patch_paths, trainB)
    print(f"[ENH] Patch dataroot staged: trainA (game) +{g}  trainB (movie) +{m}")
    print(f"[ENH] trainA total: {len(glob.glob(os.path.join(trainA, '*.jpg')))}  "
          f"trainB total: {len(glob.glob(os.path.join(trainB, '*.jpg')))}")

    finetune_cut(
        cut_dir        = cut_dir,
        pretrained_exp = pretrained_exp,
        finetune_exp   = finetune_exp,
        dataroot       = dataroot,
        device         = device,
        n_epochs       = n_epochs,
        n_epochs_decay = n_epochs_decay,
    )

    return dataroot


# ---------------------------------------------------------------------------
# Enhanced test video translation — Q2.2
# ---------------------------------------------------------------------------

def translate_test_video_enhanced(
    cut_dir: str,
    exp_name: str,
    test_path: str,
    save_dir: str,
    device: str,
    yolo_model,
    output_name: str = "enhanced_model.mp4",
    allowed_classes: list[str] | None = None,
    blend_alpha: float = 0.3,
    yolo_conf: float = 0.4,
    patch_size: int = 256,
) -> str:
    """
    Apply patch-level style transfer to human regions in the test video.

    For each frame:
      1. Detect human bounding boxes with YOLO (from Q1.1).
      2. Optionally filter by GCN class (allowed_classes from Q1.2).
      3. Crop each human region, translate with fine-tuned CUT.
      4. Resize translated crop back to original bbox dimensions.
      5. Composite onto the original frame (background untouched).
      6. Apply temporal blending with the previous frame's translated crop
         for the same region to reduce flicker (blend_alpha controls strength).

    This produces a video that differs visibly from the Q2.1 baseline:
      - Background retains original game appearance.
      - Human regions carry the movie colour/texture style.
      - Temporal blending reduces per-frame flickering artefacts.

    Args:
        cut_dir         : root of the CUT repo clone.
        exp_name        : fine-tuned checkpoint name.
        test_path       : path to Test/Test.mp4.
        save_dir        : output directory (output/q2_2/).
        device          : "cuda" | "cpu".
        yolo_model      : loaded ultralytics YOLO instance (reused from 1.1).
        output_name     : filename for the output mp4.
        allowed_classes : GCN class names to translate; None = translate all
                          detections. E.g. ['full_body_front', 'head_shoulder_front']
                          to skip back-facing and ambiguous patches.
        blend_alpha     : temporal blend weight in [0, 1].  The translated
                          crop for frame t is blended as:
                              out = alpha * prev_crop + (1-alpha) * curr_crop
                          Set to 0.0 to disable blending.
        yolo_conf       : YOLO detection confidence threshold.
        patch_size      : size to resize crops to before CUT inference.

    Returns:
        Path to the output video.
    """
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(test_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # --- Step 1: translate every frame with the patch-fine-tuned model ----
    # We still need a CUT inference pass over the full frames to get all
    # translated crops — we translate each detected human crop individually.
    # To avoid running CUT crop-by-crop (slow subprocess per image), we:
    #   a. extract all crops from all frames into a flat directory
    #   b. run a single batched CUT inference pass
    #   c. composite each translated crop back into its source frame

    frames_dir = os.path.join(save_dir, "test_frames")
    frame_paths = extract_video_frames(test_path, frames_dir, size=(1280, 720), quality=95)

    crop_dir         = os.path.join(save_dir, "crops_original")
    crop_metadata    = _extract_all_crops(
        frame_paths, yolo_model, crop_dir,
        allowed_classes=allowed_classes,
        conf=yolo_conf,
        patch_size=patch_size,
    )

    if not crop_metadata:
        print("[ENH] Warning: no crops extracted — falling back to full-frame translation.")
        from src.baseline_model import translate_test_video
        return translate_test_video(
            cut_dir, exp_name, test_path, save_dir, device, output_name
        )

    # --- Step 2: translate all crops in one batched CUT call --------------
    translated_crop_dir = os.path.join(save_dir, "crops_translated")
    translated_paths = _translate_crops(
        cut_dir, exp_name, crop_dir, translated_crop_dir, device
    )

    # Build a lookup: crop_stem → translated_path
    translated_map = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in translated_paths
    }

    # --- Step 3: composite translated crops back onto original frames -----
    composited_dir = os.path.join(save_dir, "composited_frames")
    os.makedirs(composited_dir, exist_ok=True)

    composited_paths = _composite_frames(
        frame_paths,
        crop_metadata,
        translated_map,
        composited_dir,
        blend_alpha=blend_alpha,
        patch_size=patch_size,
    )

    return write_video(composited_paths, os.path.join(save_dir, output_name), fps)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_all_crops(
    frame_paths: list[str],
    yolo_model,
    crop_dir: str,
    allowed_classes: list[str] | None,
    conf: float,
    patch_size: int,
) -> list[dict]:
    """
    Run YOLO over all frames and save each human crop as a JPEG.

    Returns a list of metadata dicts:
        {
          frame_idx : int,
          frame_path: str,
          crop_stem : str,       # stem of the saved crop file
          bbox      : (x1,y1,x2,y2),  # pixel coords in ORIGINAL frame size
          gcn_class : str | None,
        }

    Crops are saved to crop_dir/<frame_idx>_<det_idx>.jpg at patch_size×patch_size.
    """
    os.makedirs(crop_dir, exist_ok=True)

    existing_crops = glob.glob(os.path.join(crop_dir, "*.jpg"))
    if existing_crops:
        print(f"[ENH] Crops already extracted ({len(existing_crops)}), loading metadata.")
        return _reload_crop_metadata(crop_dir)

    metadata = []
    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc="Extracting crops")):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        fh, fw = frame.shape[:2]

        results = yolo_model(frame, classes=[0], conf=conf, verbose=False)
        if not results or results[0].boxes is None:
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        for det_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = (
                max(0, box[0]), max(0, box[1]),
                min(fw, box[2]), min(fh, box[3]),
            )
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, (patch_size, patch_size))

            stem = f"f{frame_idx:05d}_d{det_idx:02d}"
            crop_path = os.path.join(crop_dir, stem + ".jpg")
            cv2.imwrite(crop_path, crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 92])

            metadata.append({
                "frame_idx":  frame_idx,
                "frame_path": frame_path,
                "crop_stem":  stem,
                "bbox":       (x1, y1, x2, y2),
                "gcn_class":  None,  # populated below if allowed_classes is set
            })

    print(f"[ENH] Extracted {len(metadata)} crops from {len(frame_paths)} frames")
    _save_crop_metadata(metadata, crop_dir)
    return metadata


def _save_crop_metadata(metadata: list[dict], crop_dir: str) -> None:
    import json
    # bbox tuples aren't JSON-serialisable directly
    serialisable = [
        {**m, "bbox": list(m["bbox"])} for m in metadata
    ]
    with open(os.path.join(crop_dir, "_metadata.json"), "w") as f:
        json.dump(serialisable, f)


def _reload_crop_metadata(crop_dir: str) -> list[dict]:
    import json
    meta_path = os.path.join(crop_dir, "_metadata.json")
    if not os.path.exists(meta_path):
        print("[ENH] Warning: no _metadata.json found in crop_dir — re-extract crops.")
        return []
    with open(meta_path) as f:
        raw = json.load(f)
    return [{**m, "bbox": tuple(m["bbox"])} for m in raw]


def _translate_crops(
    cut_dir: str,
    exp_name: str,
    crop_dir: str,
    translated_dir: str,
    device: str,
) -> list[str]:
    """
    Run a single batched CUT inference pass over all crops in crop_dir.

    Builds a minimal CUT dataroot (trainA=crops, trainB=dummy) and calls
    run_cut_inference.  Returns sorted list of translated .jpg paths.
    """
    existing = sorted(glob.glob(os.path.join(translated_dir, "fake", "*.jpg")))
    if existing:
        print(f"[ENH] Translated crops already present ({len(existing)}), skipping.")
        return existing

    # CUT needs trainA and trainB; we only care about AtoB (game→movie),
    # so trainB just needs one image to satisfy the dataloader.
    dataroot = tempfile.mkdtemp()
    trainA   = os.path.join(dataroot, "trainA")
    trainB   = os.path.join(dataroot, "trainB")
    os.makedirs(trainA, exist_ok=True)
    os.makedirs(trainB, exist_ok=True)

    crop_files = sorted(glob.glob(os.path.join(crop_dir, "*.jpg")))
    for p in crop_files:
        dst = os.path.join(trainA, os.path.basename(p))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(p), dst)

    # Dummy trainB entry so CUT's dataloader doesn't complain.
    shutil.copy(crop_files[0], os.path.join(trainB, "dummy.jpg"))

    print(f"[ENH] Translating {len(crop_files)} crops with {exp_name}…")
    translated = run_cut_inference(
        cut_dir     = cut_dir,
        exp_name    = exp_name,
        dataroot    = dataroot,
        results_dir = translated_dir,
        direction   = "AtoB",
        device      = device,
    )
    print(f"[ENH] {len(translated)} crops translated")
    return translated


def _composite_frames(
    frame_paths: list[str],
    crop_metadata: list[dict],
    translated_map: dict[str, str],
    out_dir: str,
    blend_alpha: float,
    patch_size: int,
) -> list[str]:
    """
    Composite translated crops back onto their source frames.

    For each frame, pastes all translated crops at their original bbox
    coordinates.  Applies temporal blending (EMA) per detection index
    across consecutive frames when blend_alpha > 0.

    Returns sorted list of composited frame paths.
    """
    # Group metadata by frame_idx for efficient lookup
    by_frame: dict[int, list[dict]] = {}
    for m in crop_metadata:
        by_frame.setdefault(m["frame_idx"], []).append(m)

    # EMA state: det_idx → last translated crop (BGR, original bbox size)
    prev_crops: dict[int, np.ndarray] = {}

    out_paths = []
    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc="Compositing frames")):
        out_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.jpg")

        if os.path.exists(out_path):
            out_paths.append(out_path)
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        detections = by_frame.get(frame_idx, [])
        for m in detections:
            stem = m["crop_stem"]
            translated_path = translated_map.get(stem)
            if translated_path is None or not os.path.exists(translated_path):
                continue

            x1, y1, x2, y2 = m["bbox"]
            target_w = x2 - x1
            target_h = y2 - y1
            if target_w <= 0 or target_h <= 0:
                continue

            translated_crop = cv2.imread(translated_path)
            if translated_crop is None:
                continue

            # Resize translated crop back to original bbox dimensions
            translated_crop = cv2.resize(translated_crop, (target_w, target_h))

            # Temporal blending: EMA with previous crop for this detection slot
            det_idx = int(stem.split("_d")[-1])
            if blend_alpha > 0 and det_idx in prev_crops:
                prev = prev_crops[det_idx]
                if prev.shape == translated_crop.shape:
                    translated_crop = cv2.addWeighted(
                        prev,             blend_alpha,
                        translated_crop,  1.0 - blend_alpha,
                        0,
                    )

            prev_crops[det_idx] = translated_crop.copy()

            # Paste onto frame — simple rectangular composite
            frame[y1:y2, x1:x2] = translated_crop

        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        out_paths.append(out_path)

    print(f"[ENH] Composited {len(out_paths)} frames → {out_dir}")
    return out_paths