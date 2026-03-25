"""
src/enhanced_model.py
Question 2.2 — Local (Temporal) Enhancement

Pipeline
--------
1. finetune_cut_patches   — fine-tune the original checkpoint on the
                            1.3-selected human patches (game + movie).

2. translate_test_video_enhanced
                          — for each test frame:
                              a. detect humans using feat_extract (every frame,
                                 lenient blur, no scene-change skip)
                              b. classify patches with the trained 1.2 GCN;
                                 filter out unwanted classes (e.g. 'others')
                              c. translate kept crops in one batched CUT call
                              d. resize translated crop back to original bbox dims
                              e. paste onto the original (untranslated) frame
                              f. optionally apply temporal blending (step 3)

3. Temporal blending (optional, enabled with blend_alpha > 0):
   A simple per-pixel exponential moving average across consecutive
   translated crops for each tracked region, reducing flickering without
   requiring optical flow. Set blend_alpha=0.0 to disable.

Relationship to Section 1
--------------------------
  1.1  — extract_humans_from_video (feat_extract) provides detections with
          bboxes; reused here with yolo_interval=1, scene_change_threshold=0
          to cover every frame of the test video.
  1.2  — Trained GCN (load_gcn_model + run_inference) filters patches to the
          four human-pose classes, skipping 'others' and any other exclusions.
  1.3  — Selected patches form the fine-tuning dataset passed to
          finetune_cut_patches.

Shared helpers (video I/O, CUT subprocess, fine-tuning) live in src/utils.py.

Public API imported in nb_main.py:
    finetune_cut_patches, translate_test_video_enhanced
"""

import glob
import json
import os
import shutil
import tempfile

import cv2
import numpy as np
from tqdm import tqdm

from cleanfid import fid as cleanfid

from src.utils import extract_video_frames, write_video, run_cut_inference, finetune_cut, JPEG_QUALITY


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
) -> dict:
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
        Training info dict from finetune_cut (pretrained_exp, n_epochs,
        n_epochs_decay, load_size, crop_size, batch_size, training_time_s,
        training_time_human).
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

    training_info = finetune_cut(
        cut_dir        = cut_dir,
        pretrained_exp = pretrained_exp,
        finetune_exp   = finetune_exp,
        dataroot       = dataroot,
        device         = device,
        n_epochs       = n_epochs,
        n_epochs_decay = n_epochs_decay,
    )

    return training_info


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
    pose_model_path: str,
    gcn_save_path: str,
    output_name: str = "enhanced_model.mp4",
    exclude_classes: list[str] | None = None,
    direction: str = "AtoB",
    blend_alpha: float = 0.3,
    blur_threshold: float = 10.0,
    patch_size: int = 256,
    use_stgcn: bool = True,
    stgcn_window: int = 5,
    feather_px: int = 10,
) -> str:
    """
    Apply patch-level style transfer to human regions in the test video,
    using the 1.2 GCN (or ST-GCN when use_stgcn=True) to filter which
    patches receive translation.

    For each test frame:
      1. Detect human bounding boxes with YOLO (every frame, no scene-change
         skip, lenient blur threshold).
      2. Classify patches with the trained GCN; drop excluded classes.
         When use_stgcn=True, a T-frame spatio-temporal window (Yan et al.
         2018) is used instead of per-frame classification — the same PoseGCN
         weights operate on 17*T-node graphs, adding temporal consistency.
      3. Translate all kept crops in one batched CUT call.
      4. Resize translated crop back to original bbox dimensions.
      5. Composite onto the original frame (background untouched).
      6. Apply temporal blending (EMA) between consecutive frames to reduce
         flicker (blend_alpha controls strength).

    Args:
        cut_dir         : root of the CUT repo clone.
        exp_name        : fine-tuned checkpoint name.
        test_path       : path to Test/Test.mp4.
        save_dir        : output directory (output/q2_2/).
        device          : "cuda" | "cpu".
        yolo_model      : loaded ultralytics YOLO instance (reused from 1.1).
        pose_model_path : path to YOLO pose model weights (for GCN keypoints).
        gcn_save_path   : path to output/gcn_results/<run>/ containing
                          gcn_model_<run>.pt (hidden dim inferred from checkpoint).
        output_name     : filename for the output mp4.
        exclude_classes : class names to skip during compositing.
                          Defaults to ['others'] if None.
        blend_alpha     : temporal blend weight in [0, 1].
                              out = alpha * prev_crop + (1-alpha) * curr_crop
                          Set to 0.0 to disable.
        blur_threshold  : Laplacian variance floor for patch acceptance.
                          Lower = more lenient. Applies to both film and game.
        patch_size      : size to resize crops to before CUT inference.
        use_stgcn       : if True (default), use spatio-temporal GCN classification
                          (ST-GCN, Yan et al. 2018) over stgcn_window consecutive
                          frames per tracked person instead of per-frame GCN.
        stgcn_window    : temporal window size T when use_stgcn=True (default 5).
        feather_px      : width in pixels of the Gaussian-feathered border blended
                          between the translated crop and the original frame.
                          Set to 0 to disable (hard paste).

    Returns:
        Path to the output video.
    """
    from ultralytics import YOLO as _YOLO
    from src.gcn import load_gcn_model, extract_and_save_keypoints, load_keypoints
    from src.gcn import run_inference as gcn_run_inference

    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(test_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # --- Step 1: Extract all frames to disk (needed for compositing) ------
    # Save at original resolution so YOLO bbox coordinates stay valid.
    frames_dir = os.path.join(save_dir, "test_frames")
    frame_paths = extract_video_frames(test_path, frames_dir, size=(orig_w, orig_h), quality=95)

    # --- Step 2: Detect + save human patches from every frame -------------
    patch_dir = os.path.join(save_dir, "test_patches")
    crop_metadata = _extract_test_patches(
        test_path, yolo_model, frame_paths, patch_dir,
        blur_threshold=blur_threshold,
        patch_size=patch_size,
    )

    if not crop_metadata:
        print("[ENH] Warning: no crops extracted — falling back to full-frame translation.")
        from src.baseline_model import translate_test_video
        return translate_test_video(cut_dir, exp_name, test_path, save_dir, device, output_name)

    # --- Step 3: GCN inference → filter patches by class ------------------
    npz_path = os.path.join(patch_dir, "_keypoints.npz")
    if os.path.exists(npz_path):
        keypoints_dict = load_keypoints(npz_path)
        print(f"[ENH] Loaded cached keypoints ({len(keypoints_dict)} patches)")
    else:
        pose_model = _YOLO(pose_model_path)
        pose_model.to(device)
        keypoints_dict = extract_and_save_keypoints(pose_model, patch_dir, npz_path)

    gcn_model = load_gcn_model(gcn_save_path, device)
    if use_stgcn:
        from src.stgcn import run_stgcn_inference
        gcn_results = run_stgcn_inference(
            gcn_model, crop_metadata, keypoints_dict, device, T=stgcn_window,
        )
        print(f"[ENH] ST-GCN inference (T={stgcn_window}) on {len(gcn_results)} patches")
    else:
        all_fnames = sorted(os.path.basename(p) for p in glob.glob(os.path.join(patch_dir, "*.jpg")))
        gcn_results = gcn_run_inference(gcn_model, all_fnames, keypoints_dict, device)

    exclude_set = set(exclude_classes if exclude_classes is not None else ["others"])
    kept_stems = {
        os.path.splitext(fn)[0]
        for fn, cls in gcn_results.items()
        if cls not in exclude_set
    }
    print(
        f"[ENH] GCN kept {len(kept_stems)}/{len(gcn_results)} patches "
        f"(excluded: {sorted(exclude_set)})"
    )

    filtered_metadata = [m for m in crop_metadata if m["crop_stem"] in kept_stems]
    if not filtered_metadata:
        print(f"[ENH] Warning: GCN filtered all {len(crop_metadata)} patches to excluded classes "
              f"({sorted(exclude_set)}). Compositing all detected patches without GCN filter.")
    else:
        crop_metadata = filtered_metadata

    # --- Step 4: Translate all kept crops in one batched CUT call ---------
    # _translate_crops runs CUT over all .jpg in patch_dir; _composite_frames
    # only composites stems present in crop_metadata, so filtered-out patches
    # are translated but never pasted (acceptable overhead).
    translated_crop_dir = os.path.join(save_dir, "crops_translated")
    translated_paths = _translate_crops(cut_dir, exp_name, patch_dir, translated_crop_dir, device, direction)
    translated_map = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in translated_paths
    }

    meta_stems    = {m["crop_stem"] for m in crop_metadata}
    covered       = meta_stems & translated_map.keys()
    print(f"[ENH] translated_map: {len(translated_map)} entries, "
          f"{len(covered)}/{len(meta_stems)} crop stems matched")
    if not covered:
        print("[ENH] WARNING: no crop stems matched translated_map — "
              "stale crops_translated/ cache or CUT naming mismatch. "
              "Delete output/q2_2/ and rerun.")

    # --- Step 5: Composite translated crops back onto original frames -----
    composited_dir = os.path.join(save_dir, "composited_frames")
    os.makedirs(composited_dir, exist_ok=True)
    composited_paths = _composite_frames(
        frame_paths, crop_metadata, translated_map,
        composited_dir, blend_alpha=blend_alpha, patch_size=patch_size,
        feather_px=feather_px,
    )

    return write_video(composited_paths, os.path.join(save_dir, output_name), fps)


# ---------------------------------------------------------------------------
# Bounding-box FID — Q2.2
# ---------------------------------------------------------------------------

def compute_bbox_fid(
    real_patch_dir: str,
    translated_crop_dir: str,
    device: str,
    num_workers: int = 2,
) -> float:
    """
    Compute FID over human patches only (not full frames).

    real_patch_dir      : directory of real movie-domain patches
                          (use patch_dataroot/trainB staged during fine-tuning).
    translated_crop_dir : directory containing a 'fake/' sub-folder of
                          translated game patches (output of _translate_crops).
    """
    fake_dir = os.path.join(translated_crop_dir, "fake")
    if not os.path.isdir(fake_dir):
        fake_dir = translated_crop_dir
    print("[ENH] Computing bbox FID…")
    score = cleanfid.compute_fid(
        real_patch_dir, fake_dir,
        device=device, num_workers=num_workers, verbose=False,
    )
    print(f"[ENH] Bbox FID: {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_test_patches(
    test_path: str,
    yolo_model,
    frame_paths: list[str],
    patch_dir: str,
    blur_threshold: float,
    patch_size: int,
) -> list[dict]:
    """
    Run YOLO on every frame of test_path via feat_extract and save crops.

    Uses extract_humans_from_video with yolo_interval=1 and
    scene_change_threshold=0 so no frames are skipped. blur_threshold is
    applied uniformly (overriding the film/game split used in 1.1).

    Returns crop_metadata: list of dicts compatible with _composite_frames:
        {frame_idx, frame_path, crop_stem, bbox, gcn_class}
    """
    from src.feat_extract import extract_humans_from_video

    os.makedirs(patch_dir, exist_ok=True)

    meta_path = os.path.join(patch_dir, "_metadata.json")
    if os.path.exists(meta_path):
        print(f"[ENH] Test patches already cached, reloading metadata.")
        return _reload_crop_metadata(patch_dir)

    detections = extract_humans_from_video(
        yolo_model, test_path,
        yolo_interval=1,
        scene_change_threshold=0,
        blur_threshold_film=blur_threshold,
        blur_threshold_game=blur_threshold,
    )

    # Group by frame_num to assign a stable det_idx within each frame
    by_frame: dict[int, list] = {}
    for det in detections:
        by_frame.setdefault(det["frame_num"], []).append(det)

    metadata = []
    for frame_num in sorted(by_frame):
        fp = frame_paths[frame_num] if frame_num < len(frame_paths) else None
        for det_idx, det in enumerate(by_frame[frame_num]):
            patch = det.get("patch")
            if patch is None or patch.size == 0:
                continue
            stem = f"f{frame_num:06d}_d{det_idx:04d}"
            cv2.imwrite(
                os.path.join(patch_dir, stem + ".jpg"),
                cv2.resize(patch, (patch_size, patch_size)),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            metadata.append({
                "frame_idx":  frame_num,
                "frame_path": fp,
                "crop_stem":  stem,
                "bbox":       det["bbox"],
                "gcn_class":  None,
            })

    print(f"[ENH] Extracted {len(metadata)} test patches → {patch_dir}")
    _save_crop_metadata(metadata, patch_dir)
    return metadata


def _save_crop_metadata(metadata: list[dict], patch_dir: str) -> None:
    serialisable = [{**m, "frame_idx": int(m["frame_idx"]), "bbox": [int(x) for x in m["bbox"]]} for m in metadata]
    with open(os.path.join(patch_dir, "_metadata.json"), "w") as f:
        json.dump(serialisable, f)


def _reload_crop_metadata(patch_dir: str) -> list[dict]:
    meta_path = os.path.join(patch_dir, "_metadata.json")
    if not os.path.exists(meta_path):
        print("[ENH] Warning: no _metadata.json found — re-extract patches.")
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
    direction: str = "AtoB",
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

    print(f"[ENH] Translating {len(crop_files)} crops with {exp_name} ({direction})…")
    translated = run_cut_inference(
        cut_dir     = cut_dir,
        exp_name    = exp_name,
        dataroot    = dataroot,
        results_dir = translated_dir,
        direction   = direction,
        device      = device,
    )
    print(f"[ENH] {len(translated)} crops translated")
    return translated


def _feather_mask(h: int, w: int, feather: int) -> np.ndarray:
    """
    Return a float32 HxWx1 mask that is 1 in the centre and smoothly fades
    to 0 at the border over `feather` pixels using a Gaussian blur.
    """
    if feather <= 0:
        return np.ones((h, w, 1), dtype=np.float32)
    # Start with 1s, zero out the border band, then blur to get a gradient.
    mask = np.zeros((h, w), dtype=np.float32)
    f = min(feather, h // 2, w // 2)
    if f > 0:
        mask[f:h - f, f:w - f] = 1.0
    ksize = feather * 2 + 1  # always odd
    mask = cv2.GaussianBlur(mask, (ksize, ksize), feather / 2.0)
    return mask[:, :, np.newaxis]


def _composite_frames(
    frame_paths: list[str],
    crop_metadata: list[dict],
    translated_map: dict[str, str],
    out_dir: str,
    blend_alpha: float,
    patch_size: int,
    feather_px: int = 10,
) -> list[str]:
    """
    Composite translated crops back onto their source frames.

    For each frame, pastes all translated crops at their original bbox
    coordinates.  Applies:
      - spatial feathering: Gaussian-blended border (feather_px wide) between
        the translated crop and the original frame, avoiding hard edges.
      - temporal blending: EMA per detection index across consecutive frames
        when blend_alpha > 0, reducing flicker.

    Returns sorted list of composited frame paths.
    """
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
            print(f"[ENH] Warning: could not read {frame_path}, skipping frame.")
            continue

        for m in by_frame.get(frame_idx, []):
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

            translated_crop = cv2.resize(translated_crop, (target_w, target_h))

            # Temporal blending: EMA keyed by det_idx within the frame stem
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

            # Spatial feathering: blend translated crop into original frame
            # with a Gaussian-feathered mask to avoid hard bbox borders.
            if feather_px > 0:
                mask = _feather_mask(target_h, target_w, feather_px)
                orig = frame[y1:y2, x1:x2].astype(np.float32)
                blended = translated_crop.astype(np.float32) * mask + orig * (1.0 - mask)
                frame[y1:y2, x1:x2] = blended.astype(np.uint8)
            else:
                frame[y1:y2, x1:x2] = translated_crop

        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        out_paths.append(out_path)

    print(f"[ENH] Composited {len(out_paths)} frames → {out_dir}")
    return out_paths
