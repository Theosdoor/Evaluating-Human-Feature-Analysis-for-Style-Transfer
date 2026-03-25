"""
scripts/train_gcn.py

Offline GCN training script.  Run this once to produce (or refresh) the
committed checkpoint at checkpoints/gcn_model.pt.  nb_main.py then uses
that checkpoint for fast inference on any newly-extracted patches.

Usage (from repo root)
----------------------
Manual-annotation labels (recommended) — uses all available manual annotations:

    python scripts/train_gcn.py \
        --extract-dir  output/extracted_humans/20260324-104659 \
        --label-source manual \
        --pose-model   models/yolo26m-pose.pt

  Pass --annotations <dir> to use a specific session instead of all available.

Rule-based labels with manual val set:

    python scripts/train_gcn.py \
        --extract-dir   output/extracted_humans/20260324-104659 \
        --label-source  rule \
        --init-cls-dir  output/init_classifications/20260314-195748-6 \
        --val-annotations output/manual_annotated/all \
        --pose-model    models/yolo26m-pose.pt

  Omit --val-annotations to train rule-based without any validation set.
  Omit --init-cls-dir to run rule-based classification from scratch.

Merge all manual annotations into output/manual_annotated/all/:

    python scripts/train_gcn.py --merge-annotations

Ablation (train both label sources, plot comparison):

    python scripts/train_gcn.py --ablation \
        --extract-dir  output/extracted_humans/20260324-104659 \
        --init-cls-dir output/init_classifications/20260314-195748-6 \
        --pose-model   models/yolo26m-pose.pt
"""

import argparse
import json
import os
import shutil
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train GCN pose classifier and save checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--extract-dir",  default=None,
                   help="Path to output/extracted_humans/<run>/ (patches + _keypoints.npz). "
                        "Required for training; not needed for --merge-annotations.")
    p.add_argument("--label-source", choices=["manual", "rule"], default="manual",
                   help="Label source for training.")
    p.add_argument("--annotations",  default=None,
                   help="(manual) Path to manual_annotated/<run>/ dir containing annotations.json. "
                        "If omitted, all available manual annotations are merged and used.")
    p.add_argument("--val-annotations", default=None,
                   help="(rule) Path to manual_annotated/<run>/ dir to use as validation set. "
                        "If omitted, rule-based training runs without any validation set.")
    p.add_argument("--merge-annotations", action="store_true",
                   help="Merge all output/manual_annotated/*/ into output/manual_annotated/all/ "
                        "and exit (no training).")
    p.add_argument("--init-cls-dir", default=None,
                   help="(rule) Path to existing init_classifications/<run>/ dir. "
                        "If omitted, classification is run from scratch.")
    p.add_argument("--pose-model",   default=os.path.join(PROJECT_ROOT, "models/yolo26m-pose.pt"),
                   help="Path to YOLO pose model weights.")
    p.add_argument("--ckpt-dir",  default=os.path.join(PROJECT_ROOT, "checkpoints"),
                   help="Directory to copy the trained checkpoint into (saved as gcn_model_<save-name>.pt).")
    p.add_argument("--save-name",    default=None,
                   help="Run name for output/gcn_results/<save-name>/. Defaults to timestamp.")
    p.add_argument("--device",       default=None,
                   help="'cuda' | 'mps' | 'cpu'. Auto-detected if not set.")
    # GCN hyper-parameters
    p.add_argument("--hidden",      type=int,   default=128)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--batch-size",  type=int,   default=256)
    # Ablation
    p.add_argument("--ablation", action="store_true",
                   help="Train both label sources and plot per-class val-accuracy comparison. "
                        "Uses all manual annotations as val for rule-based side.")
    p.add_argument("--figures-dir", default=os.path.join(PROJECT_ROOT, "figures"),
                   help="Where to save the ablation plot.")
    return p.parse_args()


def _resolve_device(requested: str | None) -> str:
    import torch
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def merge_all_manual_annotations(manual_annotated_base: str) -> str:
    """
    Merge all output/manual_annotated/*/annotations.json (excluding 'all/')
    into output/manual_annotated/all/annotations.json.

    Deduplicates by filename — later annotation sessions overwrite earlier ones
    for the same image.  No images are copied; the JSON is the only output.

    Returns the path to the merged annotations.json.
    """
    import glob as glob_mod

    all_dir = os.path.join(manual_annotated_base, "all")
    os.makedirs(all_dir, exist_ok=True)

    ann_files = sorted(
        glob_mod.glob(os.path.join(manual_annotated_base, "*/annotations.json"))
    )
    # Exclude the 'all/' aggregation dir itself to avoid self-inclusion.
    ann_files = [f for f in ann_files if os.path.dirname(f) != all_dir]

    if not ann_files:
        print("[GCN-TRAIN] No annotations.json files found under "
              f"{manual_annotated_base} — nothing to merge.")
        out_path = os.path.join(all_dir, "annotations.json")
        with open(out_path, "w") as fh:
            json.dump({}, fh)
        return out_path

    merged: dict = {}
    for ann_path in ann_files:
        with open(ann_path) as fh:
            data = json.load(fh)
        merged.update(data)
        print(f"[GCN-TRAIN] Loaded {len(data)} annotations from {ann_path}")

    out_path = os.path.join(all_dir, "annotations.json")
    with open(out_path, "w") as fh:
        json.dump(merged, fh, indent=2)
    print(f"[GCN-TRAIN] Merged {len(merged)} unique annotations → {out_path}")
    return out_path


def _run_init_cls(extract_dir: str, pose_model_path: str, device: str) -> str:
    """Run rule-based classification and return the output directory."""
    from ultralytics import YOLO
    from src.classification import classify_directory, save_classification_summary
    from src.utils import get_next_reclassify_dir

    init_cls_base = os.path.join(PROJECT_ROOT, "output", "init_classifications")
    init_cls_dir = get_next_reclassify_dir(init_cls_base, os.path.basename(extract_dir))

    pose_model = YOLO(pose_model_path)
    pose_model.to(device)

    _, summary = classify_directory(
        pose_model,
        input_dir=extract_dir,
        output_dir=init_cls_dir,
        batch_size=32,
        copy_files=True,
        save_debug_viz=False,
        save_keypoints=True,
    )
    save_classification_summary(init_cls_dir, summary, extract_dir)
    print(f"[GCN-TRAIN] Rule-based classification saved to {init_cls_dir}")
    return init_cls_dir


def _train(
    labelled_dir: str,
    label_source: str,
    extract_dir: str,
    pose_model_path: str,
    device: str,
    save_name: str,
    hidden: int,
    lr: float,
    epochs: int,
    dropout: float,
    batch_size: int,
    ckpt_dir: str,
    val_labelled_dir: str | None = None,
) -> dict:
    """Run run_gcn_pipeline and copy checkpoint to ckpt_dir. Returns per_class_val_acc."""
    from src.gcn import run_gcn_pipeline

    gcn_save_dir = os.path.join(PROJECT_ROOT, "output", "gcn_results", save_name)
    _, _, per_class_val_acc = run_gcn_pipeline(
        labelled_dir     = labelled_dir,
        cls_source       = label_source,
        all_patches_dir  = extract_dir,
        save_dir         = gcn_save_dir,
        pose_model_path  = pose_model_path,
        device           = device,
        hidden           = hidden,
        lr               = lr,
        epochs           = epochs,
        dropout          = dropout,
        batch_size       = batch_size,
        val_labelled_dir = val_labelled_dir,
        save_plots       = True,
    )

    trained_ckpt = os.path.join(gcn_save_dir, f"gcn_model_{save_name}.pt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_dest = os.path.join(ckpt_dir, f"gcn_model_{save_name}.pt")
    shutil.copy(trained_ckpt, ckpt_dest)
    print(f"[GCN-TRAIN] Checkpoint saved → {ckpt_dest}")
    return per_class_val_acc


def main() -> None:
    args = _parse_args()

    manual_annotated_base = os.path.join(PROJECT_ROOT, "output", "manual_annotated")

    # --- Merge-only mode ---
    if args.merge_annotations:
        merge_all_manual_annotations(manual_annotated_base)
        return

    if args.extract_dir is None:
        raise ValueError("--extract-dir is required for training.")

    device    = _resolve_device(args.device)
    save_name = args.save_name or time.strftime("%Y%m%d-%H%M%S")

    print(f"[GCN-TRAIN] device={device}  label_source={args.label_source}  save_name={save_name}")

    if args.ablation:
        # --- Ablation: train both sources, plot comparison ---
        # Resolve manual annotations dir (auto-merge if not specified).
        ann_dir = args.annotations
        if ann_dir is None:
            print("[GCN-TRAIN] --annotations not set; merging all manual annotations.")
            merge_all_manual_annotations(manual_annotated_base)
            ann_dir = os.path.join(manual_annotated_base, "all")

        # Rule side — validate on all manual annotations for a trusted val set.
        rule_dir = args.init_cls_dir
        if rule_dir is None:
            rule_dir = _run_init_cls(args.extract_dir, args.pose_model, device)

        rule_acc = _train(
            labelled_dir     = rule_dir,
            label_source     = "rule",
            extract_dir      = args.extract_dir,
            pose_model_path  = args.pose_model,
            device           = device,
            save_name        = save_name + "_ablation_rule",
            hidden           = args.hidden,
            lr               = args.lr,
            epochs           = args.epochs,
            dropout          = args.dropout,
            batch_size       = args.batch_size,
            ckpt_dir         = args.ckpt_dir,
            val_labelled_dir = ann_dir,
        )

        # Manual side — this is the "real" checkpoint.
        manual_acc = _train(
            labelled_dir    = ann_dir,
            label_source    = "manual",
            extract_dir     = args.extract_dir,
            pose_model_path = args.pose_model,
            device          = device,
            save_name       = save_name + "_ablation_manual",
            hidden          = args.hidden,
            lr              = args.lr,
            epochs          = args.epochs,
            dropout         = args.dropout,
            batch_size      = args.batch_size,
            ckpt_dir        = args.ckpt_dir,
        )

        from src.gcn import plot_annotation_ablation
        os.makedirs(args.figures_dir, exist_ok=True)
        plot_annotation_ablation(
            rule_per_class_val_acc   = rule_acc,
            manual_per_class_val_acc = manual_acc,
            save_path = os.path.join(args.figures_dir, "gcn_annotation_ablation.png"),
        )
        return

    # --- Single training run ---
    if args.label_source == "manual":
        # Auto-merge all available manual annotations if no specific dir given.
        if args.annotations is None:
            print("[GCN-TRAIN] --annotations not set; merging all manual annotations.")
            merge_all_manual_annotations(manual_annotated_base)
            labelled_dir = os.path.join(manual_annotated_base, "all")
        else:
            labelled_dir = args.annotations
        val_labelled_dir = None  # manual source uses internal 80/20 split

    else:  # rule
        labelled_dir = args.init_cls_dir
        if labelled_dir is None:
            labelled_dir = _run_init_cls(args.extract_dir, args.pose_model, device)
        val_labelled_dir = args.val_annotations

    _train(
        labelled_dir     = labelled_dir,
        label_source     = args.label_source,
        extract_dir      = args.extract_dir,
        pose_model_path  = args.pose_model,
        device           = device,
        save_name        = f"{save_name}_{args.label_source}",
        hidden           = args.hidden,
        lr               = args.lr,
        epochs           = args.epochs,
        dropout          = args.dropout,
        batch_size       = args.batch_size,
        ckpt_dir         = args.ckpt_dir,
        val_labelled_dir = val_labelled_dir,
    )


if __name__ == "__main__":
    main()
