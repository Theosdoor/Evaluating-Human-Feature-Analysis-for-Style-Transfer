"""
scripts/train_gcn.py

Offline GCN training script.  Run this once to produce (or refresh) the
committed checkpoint at checkpoints/gcn_model.pt.  nb_main.py then uses
that checkpoint for fast inference on any newly-extracted patches.

Usage (from repo root)
----------------------
Manual-annotation labels (recommended):

    python scripts/train_gcn.py \
        --extract-dir  output/extracted_humans/20260324-104659 \
        --label-source manual \
        --annotations  output/manual_annotated/20260314-195748-6 \
        --pose-model   models/yolo26m-pose.pt \
        --output-ckpt  checkpoints/gcn_model.pt

Rule-based labels (no manual effort; lower accuracy):

    python scripts/train_gcn.py \
        --extract-dir  output/extracted_humans/20260324-104659 \
        --label-source rule \
        --init-cls-dir output/init_classifications/20260314-195748-6 \
        --pose-model   models/yolo26m-pose.pt \
        --output-ckpt  checkpoints/gcn_model.pt

  Omit --init-cls-dir to run rule-based classification from scratch
  (requires --pose-model).

Ablation (train both label sources, plot comparison):

    python scripts/train_gcn.py --ablation \
        --extract-dir  output/extracted_humans/20260324-104659 \
        --annotations  output/manual_annotated/20260314-195748-6 \
        --init-cls-dir output/init_classifications/20260314-195748-6 \
        --pose-model   models/yolo26m-pose.pt
"""

import argparse
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
    p.add_argument("--extract-dir",  required=True,
                   help="Path to output/extracted_humans/<run>/ (patches + _keypoints.npz).")
    p.add_argument("--label-source", choices=["manual", "rule"], default="manual",
                   help="Label source for training.")
    p.add_argument("--annotations",  default=None,
                   help="(manual) Path to manual_annotated/<run>/ dir containing annotations.json.")
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
                   help="Train both label sources and plot per-class val-accuracy comparison.")
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
) -> dict:
    """Run run_gcn_pipeline and copy checkpoint to ckpt_dir. Returns per_class_val_acc."""
    from src.gcn import run_gcn_pipeline

    gcn_save_dir = os.path.join(PROJECT_ROOT, "output", "gcn_results", save_name)
    _, _, per_class_val_acc = run_gcn_pipeline(
        labelled_dir    = labelled_dir,
        cls_source      = label_source,
        all_patches_dir = extract_dir,
        save_dir        = gcn_save_dir,
        pose_model_path = pose_model_path,
        device          = device,
        hidden          = hidden,
        lr              = lr,
        epochs          = epochs,
        dropout         = dropout,
        batch_size      = batch_size,
        save_plots      = True,
    )

    trained_ckpt = os.path.join(gcn_save_dir, f"gcn_model_{save_name}.pt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_dest = os.path.join(ckpt_dir, f"gcn_model_{save_name}.pt")
    shutil.copy(trained_ckpt, ckpt_dest)
    print(f"[GCN-TRAIN] Checkpoint saved → {ckpt_dest}")
    return per_class_val_acc


def main() -> None:
    args = _parse_args()
    device    = _resolve_device(args.device)
    save_name = args.save_name or time.strftime("%Y%m%d-%H%M%S")

    print(f"[GCN-TRAIN] device={device}  label_source={args.label_source}  save_name={save_name}")

    if args.ablation:
        # --- Ablation: train both sources, plot comparison ---
        if not args.annotations:
            raise ValueError("--ablation requires --annotations for the manual-label run.")

        # Rule side
        rule_dir = args.init_cls_dir
        if rule_dir is None:
            rule_dir = _run_init_cls(args.extract_dir, args.pose_model, device)

        rule_acc = _train(
            labelled_dir    = rule_dir,
            label_source    = "rule",
            extract_dir     = args.extract_dir,
            pose_model_path = args.pose_model,
            device          = device,
            save_name       = save_name + "_ablation_rule",
            hidden          = args.hidden,
            lr              = args.lr,
            epochs          = args.epochs,
            dropout         = args.dropout,
            batch_size      = args.batch_size,
            ckpt_dir        = args.ckpt_dir,
        )

        # Manual side — this is the "real" checkpoint
        manual_acc = _train(
            labelled_dir    = args.annotations,
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
        if not args.annotations:
            raise ValueError("--label-source manual requires --annotations <dir>.")
        labelled_dir = args.annotations

    else:  # rule
        labelled_dir = args.init_cls_dir
        if labelled_dir is None:
            labelled_dir = _run_init_cls(args.extract_dir, args.pose_model, device)

    _train(
        labelled_dir    = labelled_dir,
        label_source    = args.label_source,
        extract_dir     = args.extract_dir,
        pose_model_path = args.pose_model,
        device          = device,
        save_name       = save_name,
        hidden          = args.hidden,
        lr              = args.lr,
        epochs          = args.epochs,
        dropout         = args.dropout,
        batch_size      = args.batch_size,
        ckpt_dir        = args.ckpt_dir,
    )


if __name__ == "__main__":
    main()
