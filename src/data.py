"""
Question 1.3

data.py

Training data selection for human patch style transfer.

get_data_split() is the main entry point — for now it does a
random stratified split by class, keeping domain balance intact.

Designed to be swapped out later for a smarter selection strategy
(e.g. quality-aware, diversity-sampling, or domain-gap-maximising).
"""

import os
import random
from collections import defaultdict

from src.classification import CLASSES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_patches(cls_dir: str) -> dict[str, list[str]]:
    """
    Walk a classification output directory and return a dict:
        { class_name: [absolute_path, ...] }
    Only reads the five CLASSES subdirs.
    """
    patches: dict[str, list[str]] = defaultdict(list)
    for cls in CLASSES:
        subdir = os.path.join(cls_dir, cls)
        if not os.path.isdir(subdir):
            continue
        for fname in sorted(os.listdir(subdir)):
            if fname.lower().endswith(('.jpg', '.png')):
                patches[cls].append(os.path.join(subdir, fname))
    return patches


def _domain_from_path(path: str) -> str:
    """
    Infer domain ('game' or 'movie') from the filename prefix written by
    save_patches() in feat_extract.py.  Filenames are expected to contain
    the source video basename somewhere before the first underscore block,
    or a 'domain=<X>' tag if you add one later.

    Fallback: return 'unknown'.
    """
    fname = os.path.basename(path).lower()
    if 'game' in fname or 'mafia' in fname:
        return 'game'
    if any(kw in fname for kw in ('movie', 'godfather', 'irishman', 'sopranos')):
        return 'movie'
    return 'unknown'


# ---------------------------------------------------------------------------
# Main split function
# ---------------------------------------------------------------------------

def get_data_split(
    path_to_cls_dir: str,
    train_split: float = 0.8,
    seed: int = 42,
    classes: list[str] | None = None,
    exclude_classes: list[str] | None = None,
    min_per_class: int = 1,
) -> dict[str, dict[str, list[str]]]:
    """
    Randomly split classified patches into train and val sets.

    Stratified by class so the class distribution is preserved in both splits.
    Within each class the split is also stratified by domain so that game/movie
    balance is maintained (relevant for CUT which needs matched domain counts).

    Args:
        path_to_cls_dir:  Path to the classification output directory
                          (e.g. output/classifications/<run>).
        train_split:      Fraction of patches assigned to training (default 0.8).
        seed:             RNG seed for reproducibility.
        classes:          Explicit list of classes to include.  Defaults to all
                          five CLASSES.
        exclude_classes:  Classes to drop (e.g. ['others']).  Applied after
                          `classes`.
        min_per_class:    Skip a class/domain group if fewer than this many
                          patches exist (avoids degenerate splits).

    Returns:
        {
            'train': { class_name: [path, ...], ... },
            'val':   { class_name: [path, ...], ... },
        }

    Notes
    -----
    Future strategy ideas for 1.3:
      - Score-aware selection: read scores from extraction summary and prefer
        high-scoring patches (sharpness, detection confidence, pose quality).
      - Domain-gap maximisation: embed patches with a VGG/CLIP backbone,
        keep patches nearest the inter-domain decision boundary.
      - Cluster-then-sample: cluster per-class patches in feature space and
        sample uniformly across clusters to reduce visual redundancy.
      - Hardness filtering: prefer patches where pose classifier confidence
        is high (clean examples) or deliberately low (hard examples for GAN).
    """
    rng = random.Random(seed)

    active_classes = list(classes) if classes else list(CLASSES)
    if exclude_classes:
        active_classes = [c for c in active_classes if c not in exclude_classes]

    all_patches = _collect_patches(path_to_cls_dir)

    train: dict[str, list[str]] = defaultdict(list)
    val:   dict[str, list[str]] = defaultdict(list)

    for cls in active_classes:
        paths = all_patches.get(cls, [])
        if len(paths) < min_per_class:
            continue

        # Stratify within class by domain so game/movie ratio is preserved
        by_domain: dict[str, list[str]] = defaultdict(list)
        for p in paths:
            by_domain[_domain_from_path(p)].append(p)

        for domain, dpaths in by_domain.items():
            if len(dpaths) < min_per_class:
                # too few to split — dump everything into train
                train[cls].extend(dpaths)
                continue

            rng.shuffle(dpaths)
            n_train = max(1, int(len(dpaths) * train_split))
            train[cls].extend(dpaths[:n_train])
            val[cls].extend(dpaths[n_train:])

    # Summary
    total_train = sum(len(v) for v in train.values())
    total_val   = sum(len(v) for v in val.values())
    print(f"Data split — train: {total_train}  val: {total_val}  "
          f"(split={train_split}, seed={seed})")
    for cls in active_classes:
        n_tr = len(train.get(cls, []))
        n_va = len(val.get(cls, []))
        if n_tr + n_va:
            print(f"  {cls:<25}  train={n_tr:>5}  val={n_va:>5}")

    return {'train': dict(train), 'val': dict(val)}


# ---------------------------------------------------------------------------
# Flat accessors (convenience for CUT / style-transfer which wants flat lists)
# ---------------------------------------------------------------------------

def flat_paths(split: dict[str, list[str]]) -> list[str]:
    """Return all paths in a split dict as a single flat list."""
    return [p for paths in split.values() for p in paths]


def flat_paths_by_domain(
    split: dict[str, list[str]]
) -> tuple[list[str], list[str]]:
    """
    Partition a split dict into (game_paths, movie_paths).
    Useful for feeding trainA / trainB into CUT.
    """
    game, movie = [], []
    for paths in split.values():
        for p in paths:
            d = _domain_from_path(p)
            if d == 'game':
                game.append(p)
            elif d == 'movie':
                movie.append(p)
    return game, movie