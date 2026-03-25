"""
src/utils.py

Shared utilities for both notebook and src/ modules.

Video I/O
---------
    extract_video_frames(video_path, out_dir)  → list[str]
    write_video(frame_paths, out_path, fps)

CUT subprocess wrapper
----------------------
    run_cut_inference(cut_dir, exp_name, dataroot, results_dir, direction, device)
        → list[str]   # sorted fake paths

Classification output helpers
------------------------------
    get_next_reclassify_dir(base_dir, run_name) → str
"""

import glob
import os
import shutil
import subprocess
import sys
import time

import cv2
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def rm(*paths: str) -> None:
    """Delete directories/files, silently skip if absent."""
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

JPEG_QUALITY = 92

MOVIE_KEYWORDS = ('movie', 'film', 'godfather', 'irishman', 'sopranos')
GAME_KEYWORDS  = ('game', 'mafia')


def is_movie_source(path: str) -> bool:
    """Return True if path looks like movie-domain footage."""
    p = path.lower()
    return any(kw in p for kw in MOVIE_KEYWORDS)


def is_game_source(path: str) -> bool:
    """Return True if path looks like game-domain footage."""
    p = path.lower()
    return any(kw in p for kw in GAME_KEYWORDS)


# ---------------------------------------------------------------------------
# Video seek helper
# ---------------------------------------------------------------------------

def seek_and_read(cap: cv2.VideoCapture, target: int, cur: int):
    """
    Advance cap to frame `target` starting from `cur` using grab() for
    skipped frames, then read() to decode the target frame.

    Returns (frame, cur+1) where frame is the decoded BGR array (or None
    if cap.read() fails).  Seeks backwards with set() if target < cur.
    """
    if target < cur:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        cur = target
    while cur < target:
        cap.grab()
        cur += 1
    ret, frame = cap.read()
    return (frame if ret else None), cur + 1


# ---------------------------------------------------------------------------
# Classification directory helpers (unchanged from original)
# ---------------------------------------------------------------------------

def get_next_reclassify_dir(base_dir: str, run_name: str) -> str:
    """
    Return a new classification directory path for reclassification runs.

    Example:
    If run_name is "20260314-195748" and existing directories include
    "20260314-195748-1" and "20260314-195748-3", this returns
    "20260314-195748-4".
    """
    prefix = f"{run_name}-"
    max_idx = 0

    if os.path.isdir(base_dir):
        for name in os.listdir(base_dir):
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix):]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))

    return os.path.join(base_dir, f"{run_name}-{max_idx + 1}")


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def extract_video_frames(
    video_path: str,
    out_dir: str,
    size: tuple[int, int] = (256, 256),
    quality: int = 95,
) -> list[str]:
    """
    Extract every frame of video_path to out_dir as zero-padded JPEGs.

    Skips extraction if the expected frames already exist (cache-friendly).
    Uses a while-True read loop rather than CAP_PROP_FRAME_COUNT, which is
    unreliable for many mp4 files and would produce wrong frame totals.

    Args:
        video_path : source video file.
        out_dir    : directory to write frame_NNNNN.jpg files.
        size       : (w, h) to resize each frame before saving.
        quality    : JPEG quality [0-100].

    Returns:
        Sorted list of absolute paths to the written (or cached) JPEG files.
    """
    os.makedirs(out_dir, exist_ok=True)

    existing = sorted(glob.glob(os.path.join(out_dir, "frame_*.jpg")))
    if existing:
        print(f"[VIDEO] Frames already extracted ({len(existing)}), skipping: {out_dir}")
        return existing

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_paths = []
    i = 0
    with tqdm(desc=f"Extracting {os.path.basename(video_path)}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            p = os.path.join(out_dir, f"frame_{i:05d}.jpg")
            cv2.imwrite(p, cv2.resize(frame, size), [cv2.IMWRITE_JPEG_QUALITY, quality])
            frame_paths.append(p)
            i += 1
            pbar.update(1)

    cap.release()
    print(f"[VIDEO] {len(frame_paths)} frames extracted → {out_dir}")
    return frame_paths


def write_video(
    frame_paths: list[str],
    out_path: str,
    fps: float,
) -> str:
    """
    Write a list of JPEG frame paths to an mp4 video.

    Frames are sorted numerically by the integer embedded after the last
    underscore in the stem (e.g. frame_00042 → 42), so lexicographic sort
    artefacts from CUT's output filenames don't cause timelapse effects.

    Args:
        frame_paths : list of JPEG paths (need not be pre-sorted).
        out_path    : destination .mp4 path.
        fps         : output frame rate.

    Returns:
        out_path
    """
    def _idx(p: str) -> int:
        stem = os.path.splitext(os.path.basename(p))[0]
        digits = "".join(filter(str.isdigit, stem.split("_")[-1]))
        return int(digits) if digits else 0

    sorted_paths = sorted(frame_paths, key=_idx)

    if not sorted_paths:
        print(f"[VIDEO] Warning: no frames to write → {out_path}")
        return out_path

    sample = cv2.imread(sorted_paths[0])
    if sample is None:
        raise RuntimeError(f"Cannot read first frame: {sorted_paths[0]}")
    h, w = sample.shape[:2]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for fp in tqdm(sorted_paths, desc=f"Writing {os.path.basename(out_path)}"):
        frame = cv2.imread(fp)
        if frame is not None:
            writer.write(frame)
    writer.release()
    print(f"[VIDEO] Video → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CUT subprocess wrapper
# ---------------------------------------------------------------------------

def run_cut_inference(
    cut_dir: str,
    exp_name: str,
    dataroot: str,
    results_dir: str,
    direction: str,
    device: str,
) -> list[str]:
    """
    Run CUT's test.py and return sorted paths to the translated fake images.

    Each call gets its own isolated results directory (passed as
    --results_dir), so multiple calls with the same exp_name never overwrite
    each other's raw CUT outputs.

    Args:
        cut_dir     : root of the CUT repo clone.
        exp_name    : checkpoint name (e.g. "horse2zebra_cut_pretrained").
        dataroot    : directory with trainA/ and trainB/ (or testA/testB/).
        results_dir : where to write copied fake images (call-specific).
        direction   : "AtoB" or "BtoA".
        device      : "cuda" | "cpu".

    Returns:
        Sorted list of .jpg paths in results_dir/fake_M/ (AtoB) or results_dir/fake_G/ (BtoA).
    """
    # Semantic output label: AtoB = game→movie → fake_M, BtoA = movie→game → fake_G.
    # Note: CUT's model always internally names outputs "fake_B" because it
    # hardcodes visual_names = ['real_A', 'fake_B', 'real_B']. When direction=BtoA,
    # the dataloader swaps domains so the translated output (fake_G semantically)
    # is still written to a fake_B subdir. We copy it to the correct semantic name.
    out_label = "fake_M" if direction == "AtoB" else "fake_G"
    fake_dir = os.path.join(results_dir, out_label)
    os.makedirs(fake_dir, exist_ok=True)

    existing = sorted(glob.glob(os.path.join(fake_dir, "*.jpg")))
    if existing:
        print(f"[CUT] Cache hit: {fake_dir} ({len(existing)} images)")
        return existing

    # CUT writes raw outputs to --results_dir/<exp_name>/<phase>_latest/images/
    # Using a call-specific results_dir keeps each invocation isolated.
    cut_results_root = os.path.join(results_dir, "cut_raw")
    phase   = "test" if os.path.isdir(os.path.join(dataroot, "testA")) else "train"
    gpu_ids = "0" if device == "cuda" else "-1"

    cmd = [
        sys.executable, os.path.join(cut_dir, "test.py"),
        "--dataroot",        dataroot,
        "--name",            exp_name,
        "--model",           "cut",
        "--direction",       direction,
        "--results_dir",     cut_results_root,
        "--checkpoints_dir", os.path.join(cut_dir, "checkpoints"),
        "--gpu_ids",         gpu_ids,
        "--load_size",       "256",
        "--crop_size",       "256",
        "--no_flip",
        "--num_test",        "9999",
        "--phase",           phase,
        "--eval",
    ]
    subprocess.run(cmd, check=True, cwd=cut_dir)

    raw_dir = os.path.join(cut_results_root, exp_name, f"{phase}_latest", "images")
    # CUT always writes translated images to fake_B regardless of --direction.
    cut_fake_tag = "fake_B"
    subdir = os.path.join(raw_dir, cut_fake_tag)

    if os.path.isdir(subdir):
        sources = sorted({
            p for ext in ("*.jpg", "*.png")
            for p in glob.glob(os.path.join(subdir, ext))
        })
    else:
        sources = sorted({
            p for ext in ("jpg", "png")
            for p in glob.glob(os.path.join(raw_dir, f"*{cut_fake_tag}*.{ext}"))
        })

    if not sources:
        print(f"[CUT] WARNING: no fake outputs in {raw_dir} (direction={direction})")

    for p in sources:
        stem = os.path.splitext(os.path.basename(p))[0].replace(f"_{cut_fake_tag}", "")
        dst  = os.path.join(fake_dir, stem + ".jpg")
        if not os.path.exists(dst):
            img = cv2.imread(p)
            if img is not None:
                cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            else:
                shutil.copy(p, dst)

    result = sorted(glob.glob(os.path.join(fake_dir, "*.jpg")))
    if not result:
        print(f"[CUT] WARNING: {fake_dir} empty after inference")
    return result


# ---------------------------------------------------------------------------
# CUT fine-tuning
# ---------------------------------------------------------------------------

def finetune_cut(
    cut_dir: str,
    pretrained_exp: str,
    finetune_exp: str,
    dataroot: str,
    device: str,
    n_epochs: int = 20,
    n_epochs_decay: int = 10,
    batch_size: int = 1,
    load_size: int = 286,
    crop_size: int = 256,
) -> dict:
    """
    Fine-tune a pretrained CUT checkpoint on dataroot and save to a new
    independent experiment directory.

    The pretrained weights are copied to a new checkpoint directory
    (finetune_exp) before training begins, so the original pretrained
    checkpoint is never modified.  Both 2.1 and 2.2 call this function
    with the same pretrained_exp but different finetune_exp values,
    giving each an independent checkpoint that branches from the same
    pretrained base — making the comparison between them fair.

    Checkpoint layout after this call:
        cut_dir/checkpoints/<pretrained_exp>/   ← untouched
        cut_dir/checkpoints/<finetune_exp>/     ← new fine-tuned weights

    Args:
        cut_dir        : root of the CUT repo clone.
        pretrained_exp : source checkpoint name to copy from.
        finetune_exp   : destination checkpoint name to train into.
        dataroot       : directory with trainA/ and trainB/ subdirectories.
        device         : "cuda" | "cpu".
        n_epochs       : epochs at constant LR.
        n_epochs_decay : epochs over which LR decays to zero.
        batch_size     : images per step (1 is fine for 2080 Ti at 256px).
        load_size      : resize before random crop.
        crop_size      : training crop size.
    """
    ckpt_dir       = os.path.join(cut_dir, "checkpoints")
    src_ckpt_dir   = os.path.join(ckpt_dir, pretrained_exp)
    dst_ckpt_dir   = os.path.join(ckpt_dir, finetune_exp)

    if not os.path.isdir(src_ckpt_dir):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {src_ckpt_dir}\n"
            f"Run ensure_pretrained_models() first."
        )

    # Skip re-training if we already completed the full schedule.
    done_flag = os.path.join(dst_ckpt_dir, "training_done.flag")
    if os.path.isfile(done_flag):
        print(f"[CUT] {finetune_exp} already fully trained (found training_done.flag), skipping.")
        return {
            "pretrained_exp":    pretrained_exp,
            "n_epochs":          n_epochs,
            "n_epochs_decay":    n_epochs_decay,
            "load_size":         load_size,
            "crop_size":         crop_size,
            "batch_size":        batch_size,
            "training_time_s":   0,
            "training_time_human": "0m 0s (cached)",
        }

    # Copy pretrained weights to the new experiment directory so training
    # resumes from that base without touching the originals.
    if not os.path.isdir(dst_ckpt_dir):
        print(f"[CUT] Copying {pretrained_exp} → {finetune_exp}")
        shutil.copytree(src_ckpt_dir, dst_ckpt_dir)
    else:
        print(f"[CUT] Checkpoint {finetune_exp} already exists, resuming from it.")

    gpu_ids = "0" if device == "cuda" else "-1"

    # Only resume if ALL required CUT networks are present in dst_ckpt_dir.
    # The pretrained checkpoint may not include all networks (e.g. net_F), so
    # blindly passing --continue_train causes a FileNotFoundError on first run.
    # CUT requires: net_G, net_D, and net_F
    required_nets = ["latest_net_G.pth", "latest_net_D.pth", "latest_net_F.pth"]
    has_checkpoint = all(
        os.path.isfile(os.path.join(dst_ckpt_dir, net)) for net in required_nets
    )

    cmd = [
        sys.executable, os.path.join(cut_dir, "train.py"),
        "--dataroot",        dataroot,
        "--name",            finetune_exp,
        "--model",           "cut",
        "--checkpoints_dir", ckpt_dir,
        "--gpu_ids",         gpu_ids,
        "--load_size",       str(load_size),
        "--crop_size",       str(crop_size),
        "--batch_size",      str(batch_size),
        "--n_epochs",        str(n_epochs),
        "--n_epochs_decay",  str(n_epochs_decay),
        "--display_id",      "-1",   # no visdom
        "--no_html",
    ]
    if has_checkpoint:
        cmd.append("--continue_train")

    print(f"[CUT] Fine-tuning {finetune_exp} for {n_epochs}+{n_epochs_decay} epochs…")
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=cut_dir)
    training_time_s = time.time() - t0

    training_info = {
        "pretrained_exp": pretrained_exp,
        "n_epochs": n_epochs,
        "n_epochs_decay": n_epochs_decay,
        "load_size": load_size,
        "crop_size": crop_size,
        "batch_size": batch_size,
        "training_time_s": round(training_time_s),
        "training_time_human": f"{int(training_time_s // 60)}m {int(training_time_s % 60)}s",
    }
    # Write completion marker so future runs skip re-training.
    with open(done_flag, "w") as _f:
        _f.write(f"n_epochs={n_epochs} n_epochs_decay={n_epochs_decay}\n")
    print(f"[CUT] Fine-tuning complete in {training_info['training_time_human']} → checkpoints/{finetune_exp}/")
    return training_info