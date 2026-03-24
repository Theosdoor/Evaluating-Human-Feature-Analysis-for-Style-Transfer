"""
scripts/sample4submit.py

Populates submit/{1.1,1.2,1.3}/ with randomly sampled images for submission.
Re-running overwrites the previous sample (no timestamps).

Edit the paths / counts below, then:
    python scripts/sample4submit.py
"""

import json
import os
import random
import shutil

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------

N_11 = 50
DIR_11 = "output/extracted_humans/20260324-185427"

N_12 = 20  # per class (5 classes)
DIR_12 = "output/gcn_results/20260324-195802_manual"

N_13 = 50
DIR_13 = "output/train_select/20260324-195936"

SEED = 42

# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def abs_path(rel: str) -> str:
    return os.path.join(PROJECT_ROOT, rel)


def image_files(directory: str) -> list[str]:
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]


def clear_and_make(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def sample(files: list[str], n: int, rng: random.Random) -> list[str]:
    if len(files) <= n:
        return list(files)
    return rng.sample(files, n)


def main() -> None:
    rng = random.Random(SEED)
    submit_root = abs_path("submit")

    # --- 1.1: flat extraction dir ---
    dest_11 = os.path.join(submit_root, "1.1")
    clear_and_make(dest_11)
    imgs_11 = image_files(abs_path(DIR_11))
    for src in sample(imgs_11, N_11, rng):
        shutil.copy(src, dest_11)
    print(f"[SUBMIT] 1.1 → {len(os.listdir(dest_11))} images")

    # --- 1.2: per-class subfolders ---
    dest_12 = os.path.join(submit_root, "1.2")
    clear_and_make(dest_12)
    gcn_dir = abs_path(DIR_12)
    class_dirs = sorted(
        d for d in os.listdir(gcn_dir)
        if os.path.isdir(os.path.join(gcn_dir, d)) and d != "no_pose"
    )
    total_12 = 0
    for cls in class_dirs:
        cls_src = os.path.join(gcn_dir, cls)
        cls_imgs = image_files(cls_src)
        if not cls_imgs:
            continue
        cls_dest = os.path.join(dest_12, cls)
        os.makedirs(cls_dest, exist_ok=True)
        for src in sample(cls_imgs, N_12, rng):
            shutil.copy(src, cls_dest)
        total_12 += len(os.listdir(cls_dest))
    print(f"[SUBMIT] 1.2 → {total_12} images across {len(class_dirs)} classes")

    # --- 1.3: paths from train_split.json ---
    dest_13 = os.path.join(submit_root, "1.3")
    clear_and_make(dest_13)
    split_path = os.path.join(abs_path(DIR_13), "train_split.json")
    with open(split_path) as f:
        split = json.load(f)
    all_paths: list[str] = []
    for paths in split.values():
        all_paths.extend(paths)
    for src in sample(all_paths, N_13, rng):
        shutil.copy(src, dest_13)
    print(f"[SUBMIT] 1.3 → {len(os.listdir(dest_13))} images")

    print(f"[SUBMIT] Done → {submit_root}/")


if __name__ == "__main__":
    main()
