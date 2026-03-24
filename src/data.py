"""
Question 1.3

data.py

Training data selection for human patch style transfer.

select_with_dino_clustering() is the main entry point for 1.3:
  - Embeds all (class × domain) patches with DINOv2 ViT-B/14
  - For each group, runs K-Means with k proportional to group size
  - Returns the patch nearest (cosine) to each cluster centroid
  - Optionally saves a 2-panel UMAP coloured by class and domain

get_data_split() is retained for val splits used elsewhere.
"""

import json
import os
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

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
    save_patches() in feat_extract.py.

    Fallback: return 'unknown'.
    """
    fname = os.path.basename(path).lower()
    if 'game' in fname or 'mafia' in fname:
        return 'game'
    if any(kw in fname for kw in ('movie', 'godfather', 'irishman', 'sopranos')):
        return 'movie'
    return 'unknown'


# ---------------------------------------------------------------------------
# Random stratified split (retained for val splits used elsewhere)
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
    Randomly split classified patches into train and val sets,
    stratified by class and domain.

    Args:
        path_to_cls_dir:  Classification output directory.
        train_split:      Fraction assigned to training (default 0.8).
        seed:             RNG seed.
        classes:          Classes to include (defaults to all CLASSES).
        exclude_classes:  Classes to drop (e.g. ['others']).
        min_per_class:    Skip a class/domain group below this size.

    Returns:
        { 'train': { class_name: [path, ...] }, 'val': { ... } }
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

        by_domain: dict[str, list[str]] = defaultdict(list)
        for p in paths:
            by_domain[_domain_from_path(p)].append(p)

        for domain, dpaths in by_domain.items():
            if len(dpaths) < min_per_class:
                train[cls].extend(dpaths)
                continue
            rng.shuffle(dpaths)
            n_train = max(1, int(len(dpaths) * train_split))
            train[cls].extend(dpaths[:n_train])
            val[cls].extend(dpaths[n_train:])

    total_train = sum(len(v) for v in train.values())
    total_val   = sum(len(v) for v in val.values())
    print(f"[DATA] Data split — train: {total_train}  val: {total_val}  "
          f"(split={train_split}, seed={seed})")
    for cls in active_classes:
        n_tr = len(train.get(cls, []))
        n_va = len(val.get(cls, []))
        if n_tr + n_va:
            print(f"[DATA] {cls:<25}  train={n_tr:>5}  val={n_va:>5}")

    return {'train': dict(train), 'val': dict(val)}


# ---------------------------------------------------------------------------
# Flat accessors (convenience for CUT / style-transfer)
# ---------------------------------------------------------------------------

def flat_paths(split: dict[str, list[str]]) -> list[str]:
    """Return all paths in a split dict as a single flat list."""
    return [p for paths in split.values() for p in paths]


def flat_paths_by_domain(
    split: dict[str, list[str]]
) -> tuple[list[str], list[str]]:
    """Partition a split dict into (game_paths, movie_paths)."""
    game, movie = [], []
    for paths in split.values():
        for p in paths:
            d = _domain_from_path(p)
            if d == 'game':
                game.append(p)
            elif d == 'movie':
                movie.append(p)
    return game, movie


# ---------------------------------------------------------------------------
# Train-split persistence
# ---------------------------------------------------------------------------

_SPLIT_FILENAME = "train_split.json"


def save_train_split(save_dir: str, game_paths: list[str], movie_paths: list[str]) -> None:
    """Save selected (game_paths, movie_paths) to <save_dir>/train_split.json."""
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, _SPLIT_FILENAME)
    with open(out, "w") as f:
        json.dump({"game": game_paths, "movie": movie_paths}, f)
    print(f"[DATA] Train split saved → {out}  ({len(game_paths)} game, {len(movie_paths)} movie)")


def load_train_split(save_dir: str) -> tuple[list[str], list[str]]:
    """Load (game_paths, movie_paths) from <save_dir>/train_split.json."""
    src = os.path.join(save_dir, _SPLIT_FILENAME)
    with open(src) as f:
        data = json.load(f)
    game, movie = data["game"], data["movie"]
    print(f"[DATA] Train split loaded ← {src}  ({len(game)} game, {len(movie)} movie)")
    return game, movie


# ---------------------------------------------------------------------------
# DINOv2 embedding
# ---------------------------------------------------------------------------

def _load_dino_model(ckpt_path: str, device: str):
    """Load DINOv2 ViT-B/14 (with registers) from a local .pt checkpoint."""
    import timm
    import torch
    model = timm.create_model(
        'vit_base_patch14_reg4_dinov2.lvd142m',
        pretrained=False,
        num_classes=0,
    )
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # Facebook's checkpoint sometimes wraps weights under a 'model' key
    if isinstance(state, dict) and 'model' in state and 'cls_token' not in state:
        state = state['model']
    # Facebook ViT-B/14 checkpoint stores pos_embed with a CLS slot prepended
    # [1, 1+N, D], but timm's reg4 variant expects only spatial patches [1, N, D].
    if 'pos_embed' in state:
        ckpt_pe = state['pos_embed']          # e.g. [1, 1370, 768]
        model_pe_shape = model.pos_embed.shape # e.g. [1, 1369, 768]
        if ckpt_pe.shape[1] == model_pe_shape[1] + 1:
            state['pos_embed'] = ckpt_pe[:, 1:, :]  # drop leading CLS slot
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"[DATA] DINOv2 loaded from {ckpt_path}")
    return model


def _embed_patches(paths: list[str], model, device: str, batch_size: int = 64) -> np.ndarray:
    """Embed image paths with DINOv2. Returns float32 array [N, 768]."""
    import torch
    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((518, 518), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="[DATA] Embedding", unit="batch"):
        batch_paths = paths[i: i + batch_size]
        imgs = []
        for p in batch_paths:
            imgs.append(transform(Image.open(p).convert('RGB')))
        tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            all_embs.append(model(tensor).cpu().numpy())
    return np.concatenate(all_embs, axis=0)


# ---------------------------------------------------------------------------
# Cluster-then-select
# ---------------------------------------------------------------------------

def _cluster_select(paths: list[str], embeddings: np.ndarray, k: int, seed: int) -> list[str]:
    """
    K-Means on L2-normalised embeddings; return the patch nearest to each centroid.

    L2 normalisation before K-Means makes Euclidean distance approximate cosine
    distance, which is appropriate for DINOv2 features.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    k = min(k, len(paths))
    if k >= len(paths):
        return list(paths)

    normed    = normalize(embeddings, norm='l2')
    km        = KMeans(n_clusters=k, random_state=seed, n_init=3, max_iter=300)
    labels    = km.fit_predict(normed)
    centroids = km.cluster_centers_  # in normalised space

    selected = []
    for c_idx in range(k):
        mask = np.where(labels == c_idx)[0]
        if len(mask) == 0:
            continue
        dists = np.linalg.norm(normed[mask] - centroids[c_idx], axis=1)
        selected.append(paths[mask[np.argmin(dists)]])
    return selected


# ---------------------------------------------------------------------------
# UMAP visualisation
# ---------------------------------------------------------------------------

def plot_dino_umap(
    embeddings: np.ndarray,
    labels: list[str],
    domains: list[str],
    save_path: str,
    seed: int = 42,
) -> None:
    """Save a 2-panel UMAP of DINOv2 embeddings: coloured by class and by domain."""
    import umap as umap_lib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    print("[DATA] Computing UMAP projection…")
    xy = umap_lib.UMAP(n_components=2, random_state=seed).fit_transform(embeddings)

    cls_order  = [c for c in CLASSES if c != 'others'] + ['others']
    cls_colors = {c: plt.cm.tab10.colors[i] for i, c in enumerate(cls_order)}
    dom_colors = {'game': '#1f77b4', 'movie': '#ff7f0e', 'unknown': '#aaaaaa'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, values, color_map, cats, title in [
        (axes[0], labels,  cls_colors, cls_order,        'By class'),
        (axes[1], domains, dom_colors, ['game', 'movie'], 'By domain'),
    ]:
        ax.scatter(xy[:, 0], xy[:, 1],
                   c=[color_map.get(v, '#aaaaaa') for v in values],
                   s=3, alpha=0.6, rasterized=True, linewidths=0)
        ax.legend(handles=[mpatches.Patch(color=color_map[c], label=c)
                            for c in cats if c in color_map],
                  markerscale=3, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("DINOv2 patch embeddings (training set)", fontsize=12, fontweight='bold')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[DATA] UMAP → {save_path}")


# ---------------------------------------------------------------------------
# 1.3 — main entry point
# ---------------------------------------------------------------------------

def select_with_dino_clustering(
    gcn_dir: str,
    dino_ckpt: str,
    device: str,
    total_budget: int = 2000,
    seed: int = 42,
    exclude_classes: list[str] | None = None,
    batch_size: int = 64,
    umap_save_path: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Select representative training patches for CUT fine-tuning.

    For each (class × domain) group independently:
      k = round(total_budget × group_size / total_patches)
      K-Means in DINOv2 embedding space (cosine via L2 normalisation).
      Return the patch nearest to each cluster centroid.

    Args:
        gcn_dir        : output/gcn_results/<run>/ with per-class patch subdirs.
        dino_ckpt      : path to models/dinov2_vitb14_reg4_pretrain.pt.
        device         : 'cuda' | 'cpu'.
        total_budget   : total patches to select across all groups.
        seed           : K-Means random seed.
        exclude_classes: classes to skip (default: ['others']).
        batch_size     : DINOv2 inference batch size.
        umap_save_path : if set, save a UMAP figure here.

    Returns:
        (game_paths, movie_paths) of selected representative patches.
    """
    excl = set(exclude_classes or ['others'])
    active_classes = [c for c in CLASSES if c not in excl]

    all_patches = _collect_patches(gcn_dir)
    groups: dict[tuple[str, str], list[str]] = {}
    for cls in active_classes:
        for p in all_patches.get(cls, []):
            dom = _domain_from_path(p)
            groups.setdefault((cls, dom), []).append(p)

    # Sort for reproducible group ordering (also required for UMAP label alignment)
    sorted_groups  = sorted(groups.items())
    all_paths_flat = [p for _, paths in sorted_groups for p in paths]
    total_patches  = len(all_paths_flat)

    if total_patches == 0:
        print("[DATA] No patches found — returning empty selection.")
        return [], []

    print(f"[DATA] Embedding {total_patches} patches with DINOv2…")
    model    = _load_dino_model(dino_ckpt, device)
    all_embs = _embed_patches(all_paths_flat, model, device, batch_size=batch_size)
    path_to_idx = {p: i for i, p in enumerate(all_paths_flat)}

    selected_game:  list[str] = []
    selected_movie: list[str] = []

    print(f"[DATA] Selecting ~{total_budget} representatives via K-Means:")
    for (cls, dom), paths in sorted_groups:
        k      = max(1, round(total_budget * len(paths) / total_patches))
        idxs   = [path_to_idx[p] for p in paths]
        chosen = _cluster_select(paths, all_embs[idxs], k=k, seed=seed)
        print(f"[DATA]   {cls:<25} {dom:<7}  {len(paths):>4} → {len(chosen):>3}")
        if dom == 'game':
            selected_game.extend(chosen)
        else:
            selected_movie.extend(chosen)

    print(f"[DATA] Total selected: {len(selected_game)} game  {len(selected_movie)} movie")

    if umap_save_path:
        labels  = [cls for (cls, dom), paths in sorted_groups for _ in paths]
        domains = [dom for (cls, dom), paths in sorted_groups for _ in paths]
        plot_dino_umap(all_embs, labels, domains, umap_save_path, seed=seed)

    return selected_game, selected_movie
