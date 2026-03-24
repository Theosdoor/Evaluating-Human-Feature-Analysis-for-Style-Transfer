"""
scripts/export_false_others.py

Find test-video patches that the GCN predicts as "others" but which have
visible upper-body keypoints — likely mis-classifications that would benefit
from manual labelling and GCN retraining.

Exports the candidate patches to an output directory so you can inspect
them and then feed them through add_annotation.py or annotate.py.

Usage
-----
    # Use the most recent q2_2 run:
    python3 scripts/export_false_others.py

    # Specify a run explicitly:
    python3 scripts/export_false_others.py --q2-run 20260324-092209

    # Raise the upper-body keypoint threshold (default 4):
    python3 scripts/export_false_others.py --min-upper-kps 6

    # Also write a shell command to add them all as head_shoulder_front:
    python3 scripts/export_false_others.py --suggest-label head_shoulder_front

Output
------
    output/false_others/<q2-run>/          — symlinks to the original patches
    output/false_others/<q2-run>/report.txt — per-patch summary
"""

import argparse
import glob
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR     = os.path.join(PROJECT_ROOT, "output")

# COCO upper-body keypoint indices (nose through wrists):
# 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
# 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
# 9=left_wrist, 10=right_wrist
UPPER_BODY_KPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
KPS_CONF_THRESHOLD = 0.5


# ---------------------------------------------------------------------------

def latest_q2_run(save_dir: str) -> str:
    q2_root = os.path.join(save_dir, "q2_2")
    runs = sorted(
        d for d in os.listdir(q2_root)
        if os.path.isdir(os.path.join(q2_root, d))
        and os.path.exists(os.path.join(q2_root, d, "test_patches", "_keypoints.npz"))
    )
    if not runs:
        raise FileNotFoundError(
            f"No completed q2_2 runs (with _keypoints.npz) found in {q2_root}"
        )
    return runs[-1]


def frame_idx_from_fname(fname: str) -> int:
    """Extract frame index from filenames like f000094_d0000.jpg → 94."""
    base = os.path.splitext(fname)[0]  # strip .jpg
    # strip leading "game_" prefix added by this script on re-export
    if base.startswith("game_"):
        base = base[5:]
    part = base.split("_")[0]          # e.g. "f000094"
    if part.startswith("f") and part[1:].isdigit():
        return int(part[1:])
    return -1


def count_upper_body_kps(kps: np.ndarray, indices: list[int], threshold: float) -> int:
    """Count keypoints in `indices` with confidence above threshold."""
    return int(sum(kps[i, 2] > threshold for i in indices if i < len(kps)))


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export GCN false-'others' patches for manual labelling."
    )
    parser.add_argument(
        "--q2-run", default=None,
        help="q2_2 run name (default: most recent).",
    )
    parser.add_argument(
        "--gcn-run", default=None,
        help="GCN results run name under output/gcn_results/ (default: most recent).",
    )
    parser.add_argument(
        "--min-upper-kps", type=int, default=4,
        help="Minimum upper-body keypoints with conf>0.5 to flag a patch (default: 4).",
    )
    parser.add_argument(
        "--suggest-label", default=None, choices=[
            "full_body_front", "full_body_back",
            "head_shoulder_front", "head_shoulder_back",
        ],
        help="If set, print an add_annotation.py command for each exported patch.",
    )
    parser.add_argument(
        "--temporal-gap", type=int, default=30,
        help="Minimum frame gap between exported patches (default: 30 = 1 s at 30 fps). "
             "Filenames must follow the f<NNNNNN>_d<NNNN>.jpg convention.",
    )
    parser.add_argument(
        "--max-patches", type=int, default=None,
        help="Maximum number of patches to export (default: all candidates).",
    )
    parser.add_argument(
        "--anno-run", default=None,
        help="Annotation run dir to merge new labels into (default: most recent in "
             "output/manual_annotated/).  Passed as --out-dir to annotate.py so "
             "new annotations are merged with the existing set for GCN retraining.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, PROJECT_ROOT)
    from src.gcn import load_gcn_model, run_inference, load_keypoints

    # --- Locate run directories -------------------------------------------
    q2_run   = args.q2_run   or latest_q2_run(SAVE_DIR)
    q2_dir   = os.path.join(SAVE_DIR, "q2_2", q2_run)
    patch_dir = os.path.join(q2_dir, "test_patches")

    gcn_root = os.path.join(SAVE_DIR, "gcn_results")
    gcn_run  = args.gcn_run or sorted(os.listdir(gcn_root))[-1]
    gcn_path = os.path.join(gcn_root, gcn_run)

    anno_root = os.path.join(SAVE_DIR, "manual_annotated")
    if args.anno_run:
        anno_dir = os.path.join(anno_root, args.anno_run)
    else:
        anno_runs = sorted(
            d for d in os.listdir(anno_root)
            if os.path.isdir(os.path.join(anno_root, d))
        )
        anno_dir = os.path.join(anno_root, anno_runs[-1]) if anno_runs else None

    npz_path = os.path.join(patch_dir, "_keypoints.npz")

    print(f"[export_false_others] q2_2 run  : {q2_run}")
    print(f"[export_false_others] GCN run   : {gcn_run}")
    print(f"[export_false_others] patch_dir : {patch_dir}")

    if not os.path.exists(npz_path):
        print(f"ERROR: keypoints file not found: {npz_path}", file=sys.stderr)
        sys.exit(1)

    # --- Run GCN inference ------------------------------------------------
    all_fnames = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(patch_dir, "*.jpg"))
    )
    print(f"[export_false_others] {len(all_fnames)} patches found")

    kps_dict = load_keypoints(npz_path)
    model    = load_gcn_model(gcn_path, device)
    results  = run_inference(model, all_fnames, kps_dict, device)

    # --- Filter: predicted "others" with sufficient upper-body keypoints --
    candidates = []
    for fname, label in results.items():
        if label != "others":
            continue
        entry = kps_dict.get(fname)
        if entry is None:
            continue   # no_pose — handled separately; not a GCN error
        n_upper = count_upper_body_kps(
            entry["kps"], UPPER_BODY_KPS, KPS_CONF_THRESHOLD
        )
        if n_upper >= args.min_upper_kps:
            candidates.append((fname, n_upper, entry["kps"]))

    candidates.sort(key=lambda x: -x[1])   # most upper-body kps first
    print(f"[export_false_others] {len(candidates)} false-'others' candidates "
          f"(≥{args.min_upper_kps} upper-body kps)")

    # --- Enforce temporal gap to avoid exporting near-duplicate frames -----
    if args.temporal_gap > 0:
        seen_frames: list[int] = []
        filtered = []
        for item in candidates:
            fidx = frame_idx_from_fname(item[0])
            if fidx >= 0 and any(abs(fidx - f) < args.temporal_gap for f in seen_frames):
                continue
            filtered.append(item)
            if fidx >= 0:
                seen_frames.append(fidx)
        print(f"[export_false_others] {len(filtered)} after temporal-gap={args.temporal_gap} filter")
        candidates = filtered

    if args.max_patches:
        candidates = candidates[: args.max_patches]
        print(f"[export_false_others] {len(candidates)} after --max-patches cap")

    if not candidates:
        print("[export_false_others] Nothing to export.")
        return

    # --- Export via symlinks into others/ subdir --------------------------
    # Structure matches what annotate.py expects: <cls_dir>/<class>/<patches>
    # Patches land in others/ so the annotation tool shows them as needing
    # reclassification — the user assigns the real label interactively.
    out_dir     = os.path.join(SAVE_DIR, "false_others", q2_run)
    others_dir  = os.path.join(out_dir, "others")
    os.makedirs(others_dir, exist_ok=True)
    # Create stubs for the remaining classes so annotate.py's reclassify
    # panel can move patches into them.
    for cls in ["full_body_front", "full_body_back",
                "head_shoulder_front", "head_shoulder_back", "bad_extraction"]:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    report_lines = []
    exported_paths = []
    for fname, n_upper, kps in candidates:
        src = os.path.abspath(os.path.join(patch_dir, fname))
        # Prefix with "game_" so annotate.py's domain filter recognises these
        # test patches (all from the game test video) rather than returning
        # "unknown" and silently excluding them from the annotation queue.
        dst_fname = "game_" + fname
        dst = os.path.join(others_dir, dst_fname)
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)
        exported_paths.append(src)
        report_lines.append(f"{dst_fname}  upper_kps={n_upper}")

    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(f"q2_run: {q2_run}\n")
        f.write(f"gcn_run: {gcn_run}\n")
        f.write(f"min_upper_kps: {args.min_upper_kps}\n")
        f.write(f"total: {len(candidates)}\n\n")
        f.write("\n".join(report_lines))

    print(f"[export_false_others] Exported → {others_dir}  ({len(candidates)} patches)")
    print(f"[export_false_others] Report   → {report_path}")
    out_dir_flag = f" --out-dir {anno_dir}" if anno_dir else ""
    print(f"[export_false_others] Annotate → python3 scripts/annotate.py --cls-dir {out_dir}{out_dir_flag} --kp-cache {npz_path}")

    if args.suggest_label:
        paths_str = " ".join(f'"{p}"' for p in exported_paths)
        print(f"\n# Add all as '{args.suggest_label}':")
        print(
            f"python3 scripts/add_annotation.py {paths_str} "
            f"--label {args.suggest_label} --interactive"
        )


if __name__ == "__main__":
    main()
