#!/usr/bin/env python3
"""
Ablation study for MediaPipe face detection in orientation classification.

This script classifies the same patch set twice using identical YOLO pose
outputs:
1) without MediaPipe (face_detected=False)
2) with MediaPipe (face_detected from blaze_face detector)

Outputs:
- per_image_results.csv      : image-level labels and change flags
- summary.json              : machine-readable counts and metrics
- summary.txt               : human-readable report

Example
-------
python3 scripts/ablate_mediapipe.py \
  --input-dir output/extracted_humans/20260225-104100 \
  --output-dir output/ablation_mediapipe/20260225-104100 \
  --pose-model models/yolo26m-pose.pt \
  --batch-size 32
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from ultralytics import YOLO

# Ensure imports work when this script is run from project root.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification import (  # noqa: E402
    CLASSES,
    _run_face_detection,
    build_face_detector,
    classify_keypoints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study: classification with vs without MediaPipe"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing patch images (.jpg/.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where ablation outputs are written",
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "yolo26m-pose.pt",
        help="Path to YOLO pose model (.pt)",
    )
    parser.add_argument(
        "--face-model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "blaze_face_short_range.tflite",
        help="Path for MediaPipe face detector model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for YOLO pose inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with columns filename,label for accuracy comparison"
        ),
    )
    return parser.parse_args()


def select_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect_images(input_dir: Path) -> List[Path]:
    images = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    return images


def load_ground_truth(csv_path: Optional[Path]) -> Dict[str, str]:
    if csv_path is None:
        return {}
    gt: Dict[str, str] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"filename", "label"}
        if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Ground-truth CSV must contain columns: filename,label"
            )
        for row in reader:
            gt[row["filename"]] = row["label"]
    return gt


def orientation_bucket(label: str) -> str:
    if label.endswith("_front"):
        return "front"
    if label.endswith("_back"):
        return "back"
    return "other"


def run_ablation(
    model: YOLO,
    image_paths: List[Path],
    batch_size: int,
    face_detector,
) -> List[dict]:
    records: List[dict] = []

    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    for start in tqdm(
        range(0, len(image_paths), batch_size),
        total=total_batches,
        desc="Ablation inference",
        unit="batch",
    ):
        batch_paths = image_paths[start : start + batch_size]
        batch_str_paths = [str(p) for p in batch_paths]
        batch_results = model(batch_str_paths, verbose=False)

        for img_path, result in zip(batch_paths, batch_results):
            filename = img_path.name

            if result.keypoints is None or result.keypoints.data.shape[0] == 0:
                label_no_mp = "others"
                label_with_mp = "others"
                face_detected = False
            else:
                boxes = result.boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_idx = int(areas.argmax())
                keypoints = result.keypoints.data[best_idx].cpu().numpy()
                bbox = boxes[best_idx]

                label_no_mp = classify_keypoints(
                    keypoints,
                    face_detected=False,
                    bbox=bbox,
                )

                face_detected = _run_face_detection(
                    face_detector,
                    str(img_path),
                    bbox=bbox,
                )
                label_with_mp = classify_keypoints(
                    keypoints,
                    face_detected=face_detected,
                    bbox=bbox,
                )

            changed = label_no_mp != label_with_mp
            records.append(
                {
                    "filename": filename,
                    "label_no_mediapipe": label_no_mp,
                    "label_with_mediapipe": label_with_mp,
                    "changed": changed,
                    "orientation_no_mediapipe": orientation_bucket(label_no_mp),
                    "orientation_with_mediapipe": orientation_bucket(label_with_mp),
                    "face_detected": face_detected,
                }
            )

    return records


def accuracy(records: List[dict], gt_map: Dict[str, str], label_key: str) -> Optional[float]:
    if not gt_map:
        return None

    valid = [r for r in records if r["filename"] in gt_map]
    if not valid:
        return None

    correct = sum(1 for r in valid if r[label_key] == gt_map[r["filename"]])
    return correct / len(valid)


def summarise(records: List[dict], gt_map: Dict[str, str]) -> dict:
    count_no = Counter(r["label_no_mediapipe"] for r in records)
    count_yes = Counter(r["label_with_mediapipe"] for r in records)
    transitions = Counter(
        (r["label_no_mediapipe"], r["label_with_mediapipe"])
        for r in records
        if r["changed"]
    )
    orient_transitions = Counter(
        (r["orientation_no_mediapipe"], r["orientation_with_mediapipe"])
        for r in records
        if r["orientation_no_mediapipe"] != r["orientation_with_mediapipe"]
    )

    n_total = len(records)
    n_changed = sum(r["changed"] for r in records)

    summary = {
        "n_images": n_total,
        "n_changed": n_changed,
        "change_rate": (n_changed / n_total) if n_total else 0.0,
        "class_counts_no_mediapipe": {c: int(count_no.get(c, 0)) for c in CLASSES},
        "class_counts_with_mediapipe": {c: int(count_yes.get(c, 0)) for c in CLASSES},
        "top_transitions": [
            {"from": a, "to": b, "count": n}
            for (a, b), n in transitions.most_common(15)
        ],
        "orientation_transitions": [
            {"from": a, "to": b, "count": n}
            for (a, b), n in orient_transitions.most_common(10)
        ],
        "n_face_detected": int(sum(1 for r in records if r["face_detected"])),
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
    }

    if gt_map:
        summary["ground_truth_rows"] = len(gt_map)
        summary["matched_ground_truth_rows"] = int(
            sum(1 for r in records if r["filename"] in gt_map)
        )
        summary["accuracy_no_mediapipe"] = accuracy(
            records,
            gt_map,
            "label_no_mediapipe",
        )
        summary["accuracy_with_mediapipe"] = accuracy(
            records,
            gt_map,
            "label_with_mediapipe",
        )

    return summary


def write_csv(records: List[dict], out_csv: Path) -> None:
    fields = [
        "filename",
        "label_no_mediapipe",
        "label_with_mediapipe",
        "changed",
        "orientation_no_mediapipe",
        "orientation_with_mediapipe",
        "face_detected",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_summary_txt(summary: dict, output_path: Path) -> None:
    lines = []
    lines.append("MediaPipe Ablation Study")
    lines.append("=" * 40)
    lines.append(f"Timestamp: {summary['timestamp']}")
    lines.append(f"Images: {summary['n_images']}")
    lines.append(f"Changed labels: {summary['n_changed']} ({100.0 * summary['change_rate']:.2f}%)")
    lines.append(f"Face detected count: {summary['n_face_detected']}")
    lines.append("")

    lines.append("Class counts (NO MediaPipe):")
    for cls in CLASSES:
        lines.append(f"  {cls:22s} {summary['class_counts_no_mediapipe'][cls]:6d}")
    lines.append("")

    lines.append("Class counts (WITH MediaPipe):")
    for cls in CLASSES:
        lines.append(f"  {cls:22s} {summary['class_counts_with_mediapipe'][cls]:6d}")
    lines.append("")

    lines.append("Top label transitions (no_mp -> with_mp):")
    if summary["top_transitions"]:
        for t in summary["top_transitions"]:
            lines.append(f"  {t['from']} -> {t['to']}: {t['count']}")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("Orientation transitions:")
    if summary["orientation_transitions"]:
        for t in summary["orientation_transitions"]:
            lines.append(f"  {t['from']} -> {t['to']}: {t['count']}")
    else:
        lines.append("  (none)")
    lines.append("")

    if "accuracy_no_mediapipe" in summary:
        lines.append("Ground-truth comparison:")
        lines.append(f"  Ground-truth rows: {summary['ground_truth_rows']}")
        lines.append(
            f"  Matched rows: {summary['matched_ground_truth_rows']}"
        )
        no_acc = summary["accuracy_no_mediapipe"]
        yes_acc = summary["accuracy_with_mediapipe"]
        if no_acc is None or yes_acc is None:
            lines.append("  Accuracy: unavailable (no matched rows)")
        else:
            lines.append(f"  Accuracy no MediaPipe:   {100.0 * no_acc:.2f}%")
            lines.append(f"  Accuracy with MediaPipe: {100.0 * yes_acc:.2f}%")
            lines.append(
                f"  Delta (with - without):  {100.0 * (yes_acc - no_acc):+.2f}%"
            )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No .jpg/.png images found in: {input_dir}")

    device = select_device(args.device)
    print(f"Using device: {device}")
    print(f"Input images: {len(image_paths)}")

    pose_model = YOLO(str(args.pose_model.resolve()))
    pose_model.to(device)

    face_detector = build_face_detector(model_path=str(args.face_model_path.resolve()))

    try:
        records = run_ablation(
            model=pose_model,
            image_paths=image_paths,
            batch_size=args.batch_size,
            face_detector=face_detector,
        )
    finally:
        if face_detector is not None:
            face_detector.close()

    gt_map = load_ground_truth(args.ground_truth_csv.resolve()) if args.ground_truth_csv else {}
    summary = summarise(records, gt_map)

    csv_path = output_dir / "per_image_results.csv"
    summary_json_path = output_dir / "summary.json"
    summary_txt_path = output_dir / "summary.txt"

    write_csv(records, csv_path)
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_summary_txt(summary, summary_txt_path)

    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_json_path}")
    print(f"Saved: {summary_txt_path}")


if __name__ == "__main__":
    main()
