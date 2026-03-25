"""
scripts/run_m2g_inference.py

Re-run CUT 2.1 and 2.2 in the BtoA (movie→game) direction using already-saved
finetuned checkpoints on HPC. Produces outputs suitable for use in
figure_select.py (2.1 tab: movie orig + fake-game pairs; 2.2 tab: triples).

Does NOT retrain anything. Requires the finetuned checkpoints to be present in
    external/contrastive-unpaired-translation/checkpoints/
    ├── cut_finetuned_fullframe/   ← 2.1
    └── cut_finetuned_patches/     ← 2.2

Usage
-----
    source .venv/bin/activate
    python3 scripts/run_m2g_inference.py

Outputs
-------
    output/q2_1/<run>/results/test_m2g/fake_G/    ← 2.1 fake game frames
    output/q2_1/<run>/test_frames_movie/           ← movie originals (shared input)
    output/q2_2/<run>/enh_frames/                  ← 2.2 enhanced composited frames

Prints the figure_select.py invocation at the end.
"""

import glob
import os
import random
import shutil
import sys
import time

import cv2
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feat_extract import reload_extracted_patches
from src.utils import run_cut_inference, extract_video_frames, write_video, JPEG_QUALITY
from src.baseline_model import make_inference_dataroot

# ---------------------------------------------------------------------------
# Configuration — edit these to match your HPC run's timestamps
# ---------------------------------------------------------------------------

RELOAD_EXTRACT  = "20260324-185427"          # 1.1 extraction timestamp
RELOAD_GCN      = "20260325-095859_ablation_manual"  # 1.2 GCN timestamp
FULLFRAME_MODEL = "cut_finetuned_fullframe"  # 2.1 checkpoint name
PATCH_MODEL     = "cut_finetuned_patches"    # 2.2 checkpoint name

SAVE_NAME = time.strftime('%Y%m%d-%H%M%S')

DATA_DIR = os.path.join(PROJECT_ROOT, "downloaded_data")
SAVE_DIR = os.path.join(PROJECT_ROOT, "output")

MOVIE_CLIPS = [
    os.path.join(DATA_DIR, "Train/movie/TheGodfather.mp4"),
    os.path.join(DATA_DIR, "Train/movie/TheIrishman.mp4"),
    os.path.join(DATA_DIR, "Train/movie/TheSopranos.mp4"),
]

# Frames per clip when building the mixed movie test set
FRAMES_PER_CLIP = 80   # 3 clips × 80 = 240 movie test frames total
FRAME_SIZE      = (256, 256)

CUT_DIR = os.path.join(PROJECT_ROOT, "external/contrastive-unpaired-translation")

DEVICE = "cuda"   # HPC always has CUDA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mixed_movie_frames(out_dir: str) -> list[str]:
    """
    Extract FRAMES_PER_CLIP evenly-spaced frames from each movie clip and
    copy them all into out_dir. Returns sorted list of frame paths.
    Skips extraction if out_dir is already populated.
    """
    existing = sorted(glob.glob(os.path.join(out_dir, "*.jpg")))
    if existing:
        print(f"[M2G] Movie frames already present ({len(existing)}), skipping.")
        return existing

    os.makedirs(out_dir, exist_ok=True)
    all_paths: list[str] = []

    for clip in MOVIE_CLIPS:
        clip_name = os.path.splitext(os.path.basename(clip))[0]
        tmp_dir   = os.path.join(out_dir, f"_tmp_{clip_name}")
        raw_paths = extract_video_frames(clip, tmp_dir, size=FRAME_SIZE)

        # Evenly space FRAMES_PER_CLIP frames across the full clip
        step     = max(1, len(raw_paths) // FRAMES_PER_CLIP)
        selected = raw_paths[::step][:FRAMES_PER_CLIP]

        for p in selected:
            dst = os.path.join(out_dir, f"{clip_name}_{os.path.basename(p)}")
            shutil.copy(p, dst)
            all_paths.append(dst)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[M2G] {clip_name}: kept {len(selected)} frames")

    print(f"[M2G] Mixed movie test set: {len(all_paths)} frames → {out_dir}")
    return sorted(glob.glob(os.path.join(out_dir, "*.jpg")))


def _run_q21_m2g(movie_frames_dir: str, q2_1_out_dir: str) -> str:
    """
    Run 2.1 checkpoint in BtoA on movie frames → fake_G.
    Returns path to fake_G directory.
    """
    # CUT needs trainA (input domain) + trainB (target domain, at least one file).
    # BtoA: B=movie → A=game, so trainA=movie frames, trainB=dummy.
    dummy_dir = os.path.join(q2_1_out_dir, "_dummy_game_frame")
    os.makedirs(dummy_dir, exist_ok=True)
    movie_files = sorted(glob.glob(os.path.join(movie_frames_dir, "*.jpg")))
    if not movie_files:
        raise RuntimeError(f"No movie frames in {movie_frames_dir}")
    shutil.copy(movie_files[0], os.path.join(dummy_dir, "dummy.jpg"))

    dataroot  = make_inference_dataroot(movie_frames_dir, dummy_dir)
    fake_g_dir = os.path.join(q2_1_out_dir, "results", "test_m2g")

    print("[M2G] Running 2.1 (cut_finetuned_fullframe) BtoA…")
    fake_paths = run_cut_inference(
        cut_dir     = CUT_DIR,
        exp_name    = FULLFRAME_MODEL,
        dataroot    = dataroot,
        results_dir = fake_g_dir,
        direction   = "BtoA",
        device      = DEVICE,
    )
    print(f"[M2G] 2.1 BtoA: {len(fake_paths)} fake-game frames")
    shutil.rmtree(dummy_dir, ignore_errors=True)
    return os.path.join(fake_g_dir, "fake_G")


def _run_q22_m2g(
    movie_frames_dir: str,
    q2_2_out_dir:     str,
    gcn_save_path:    str,
) -> str:
    """
    Run 2.2 enhanced model in BtoA on movie frames, composite translated human
    patches back onto movie source frames. Returns path to enh_frames/ dir.

    For the 2.2 tab in figure_select.py the triple is:
        [movie original | 2.1 fake-game | 2.2 enhanced-game]
    """
    from ultralytics import YOLO
    from src.enhanced_model import translate_test_video_enhanced

    yolo_model_path = os.path.join(PROJECT_ROOT, "models/yolo26m.pt")
    pose_model_path = os.path.join(PROJECT_ROOT, "models/yolo26m-pose.pt")

    # translate_test_video_enhanced expects a video file, not frames.
    # Build a temporary video from the mixed movie frames so we can reuse it.
    tmp_video_path = os.path.join(q2_2_out_dir, "_movie_mix.mp4")
    if not os.path.exists(tmp_video_path):
        os.makedirs(q2_2_out_dir, exist_ok=True)
        movie_files = sorted(glob.glob(os.path.join(movie_frames_dir, "*.jpg")))
        sample = cv2.imread(movie_files[0])
        h, w = sample.shape[:2]
        writer = cv2.VideoWriter(
            tmp_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            25.0,
            (w, h),
        )
        for p in tqdm(movie_files, desc="Building tmp movie video"):
            frame = cv2.imread(p)
            if frame is not None:
                writer.write(frame)
        writer.release()
        print(f"[M2G] Wrote temp movie video: {tmp_video_path}")

    yolo_model = YOLO(yolo_model_path)
    yolo_model.to(DEVICE)

    print("[M2G] Running 2.2 (cut_finetuned_patches) BtoA enhanced…")
    translate_test_video_enhanced(
        cut_dir         = CUT_DIR,
        exp_name        = PATCH_MODEL,
        test_path       = tmp_video_path,
        save_dir        = q2_2_out_dir,
        device          = DEVICE,
        yolo_model      = yolo_model,
        pose_model_path = pose_model_path,
        gcn_save_path   = gcn_save_path,
        output_name     = "m2g_enhanced.mp4",
        exclude_classes = ["others"],
        direction       = "BtoA",
    )

    # Copy composited frames to enh_frames/ so figure_select auto-discovers them.
    composited_dir = os.path.join(q2_2_out_dir, "composited_frames")
    enh_dir        = os.path.join(q2_2_out_dir, "enh_frames")
    if os.path.isdir(composited_dir) and not os.path.isdir(enh_dir):
        shutil.copytree(composited_dir, enh_dir)
        print(f"[M2G] Copied composited_frames → enh_frames")

    os.remove(tmp_video_path)
    return enh_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Resolve output dirs
    extract_save_path = os.path.join(SAVE_DIR, "extracted_humans", RELOAD_EXTRACT)
    gcn_save_path     = os.path.join(SAVE_DIR, "gcn_results",       RELOAD_GCN)
    q2_1_out_dir      = os.path.join(SAVE_DIR, "q2_1", SAVE_NAME)
    q2_2_out_dir      = os.path.join(SAVE_DIR, "q2_2", SAVE_NAME)
    movie_frames_dir  = os.path.join(q2_1_out_dir, "test_frames_movie")

    os.makedirs(q2_1_out_dir, exist_ok=True)

    # Check checkpoints exist before doing anything expensive
    for ckpt_name in (FULLFRAME_MODEL, PATCH_MODEL):
        ckpt_dir = os.path.join(CUT_DIR, "checkpoints", ckpt_name)
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(
                f"CUT checkpoint not found: {ckpt_dir}\n"
                "Make sure the finetuned weights are still present on HPC."
            )
    print(f"[M2G] Checkpoints verified: {FULLFRAME_MODEL}, {PATCH_MODEL}")

    # 1. Build mixed movie test frames
    _build_mixed_movie_frames(movie_frames_dir)

    # 2. Run 2.1 in BtoA direction
    fake_g_21_dir = _run_q21_m2g(movie_frames_dir, q2_1_out_dir)

    # 3. Run 2.2 in BtoA direction (patch-level compositing on movie frames)
    enh_dir = _run_q22_m2g(movie_frames_dir, q2_2_out_dir, gcn_save_path)

    # 4. Print figure_select invocations
    print("\n" + "="*70)
    print("[M2G] Done. Run figure_select.py with:")
    print()
    print("  # 2.1 tab (movie original | fake-game):")
    print(f"  python3 scripts/figure_select.py \\")
    print(f"      --q21-orig-dir {movie_frames_dir} \\")
    print(f"      --q21-fake-dir {fake_g_21_dir}")
    print()
    print("  # 2.2 tab (movie original | 2.1 fake-game | 2.2 enhanced-game):")
    print(f"  python3 scripts/figure_select.py \\")
    print(f"      --q22-orig-dir {movie_frames_dir} \\")
    print(f"      --q22-21-dir   {fake_g_21_dir} \\")
    print(f"      --q22-22-dir   {enh_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
