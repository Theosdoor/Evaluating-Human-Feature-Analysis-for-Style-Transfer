"""
scripts/sample4submit.py

Populates submit/{1.1,1.2,1.3}/ with randomly sampled images for submission.
Re-running overwrites the previous sample (no timestamps).

Edit the paths / counts below, then:
    python scripts/sample4submit.py

Or import and call run_sampling() directly with explicit paths.
"""

import json
import os
import random
import shutil

# ---------------------------------------------------------------------------
# Configuration — edit these before running standalone
# ---------------------------------------------------------------------------

N_11 = 50
DIR_11 = "output/extracted_humans/20260324-185427"

N_12 = 20  # per class (5 classes)
DIR_12 = "output/gcn_results/20260325-095859_ablation_manual"

N_13 = 50
DIR_13 = "output/train_select/20260325-105918"

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


def run_sampling(
    dir_11: str,
    dir_12: str,
    dir_13: str,
    n_11: int = 50,
    n_12: int = 20,
    n_13: int = 50,
    seed: int = 42,
    project_root: str = PROJECT_ROOT,
) -> None:
    """Sample submission images from pipeline output directories.

    Parameters
    ----------
    dir_11 : str
        Absolute path to the 1.1 extraction output directory.
    dir_12 : str
        Absolute path to the 1.2 GCN results directory.
    dir_13 : str
        Absolute path to the 1.3 train-select output directory.
    n_11 : int
        Number of patches to sample for 1.1.
    n_12 : int
        Number of patches per class to sample for 1.2.
    n_13 : int
        Number of patches to sample for 1.3.
    seed : int
        Random seed for reproducibility.
    project_root : str
        Project root directory; submit/ is created here.
    """
    rng = random.Random(seed)
    submit_root = os.path.join(project_root, "submit")

    # --- 1.1: flat extraction dir ---
    dest_11 = os.path.join(submit_root, "1.1")
    clear_and_make(dest_11)
    imgs_11 = image_files(dir_11)
    for src in sample(imgs_11, n_11, rng):
        shutil.copy(src, dest_11)
    print(f"[SUBMIT] 1.1 → {len(os.listdir(dest_11))} images")

    # --- 1.2: per-class subfolders ---
    dest_12 = os.path.join(submit_root, "1.2")
    clear_and_make(dest_12)
    class_dirs = sorted(
        d for d in os.listdir(dir_12)
        if os.path.isdir(os.path.join(dir_12, d)) and d != "no_pose"
    )
    total_12 = 0
    for cls in class_dirs:
        cls_src = os.path.join(dir_12, cls)
        cls_imgs = image_files(cls_src)
        if not cls_imgs:
            continue
        cls_dest = os.path.join(dest_12, cls)
        os.makedirs(cls_dest, exist_ok=True)
        for src in sample(cls_imgs, n_12, rng):
            shutil.copy(src, cls_dest)
        total_12 += len(os.listdir(cls_dest))
    print(f"[SUBMIT] 1.2 → {total_12} images across {len(class_dirs)} classes")

    # --- 1.3: paths from train_split.json ---
    dest_13 = os.path.join(submit_root, "1.3")
    clear_and_make(dest_13)
    split_path = os.path.join(dir_13, "train_split.json")
    with open(split_path) as f:
        split = json.load(f)
    all_paths: list[str] = []
    for paths in split.values():
        all_paths.extend(paths)
    for src in sample(all_paths, n_13, rng):
        shutil.copy(src, dest_13)
    print(f"[SUBMIT] 1.3 → {len(os.listdir(dest_13))} images")

    print(f"[SUBMIT] Done → {submit_root}/")


def main() -> None:
    run_sampling(
        dir_11=abs_path(DIR_11),
        dir_12=abs_path(DIR_12),
        dir_13=abs_path(DIR_13),
        n_11=N_11,
        n_12=N_12,
        n_13=N_13,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
