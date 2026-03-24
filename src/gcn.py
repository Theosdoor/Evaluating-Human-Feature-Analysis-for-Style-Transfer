"""
src/gcn.py

Question 1.2 (extension) — GCN-based patch classifier.

Pipeline
--------
1. Load keypoints from _keypoints.npz in the extracted_humans dir
   (or re-run pose inference and save them if missing).
2. Build one graph per patch:
     nodes  : 17 COCO keypoints
     edges  : COCO skeleton (bidirectional)
     features: (x_norm, y_norm, conf) — coordinates normalised by bbox dims
3. Train a GCN on the manually/rule-labelled subset (80/20 stratified split).
   Confidence-weighted global pooling aggregates node embeddings to a graph vector.
   WeightedRandomSampler rebalances minority classes (fb_back, hs_back) each epoch.
4. Run inference on all patches; patches with no pose detection → 'others'.
5. Save results to output/gcn_results/<run>/ in the same per-class subdir format
   as init_classifications, so downstream steps (1.3, flat_paths_by_domain) are
   unaffected.

Node features
-------------
Option used: (x_norm, y_norm, conf) — bbox-normalised coordinates + confidence.
# TODO: if results are poor, extend to option 3: add relative angles/distances
#       between connected keypoints as additional node features.

Usage (from nb_main)
--------------------
    from src.gcn import run_gcn_pipeline
    results, summary = run_gcn_pipeline(
        labelled_dir   = cls_save_path,       # init_classifications or manual_annotated dir
        cls_source     = "manual",            # "manual" | "rule"
        all_patches_dir= extract_save_path,   # extracted_humans/<ts>/
        save_dir       = gcn_save_dir,
        pose_model_path= os.path.join(PROJECT_ROOT, "models/yolo26m-pose.pt"),
        device         = DEVICE,
        lr             = 3e-4,
        epochs         = 150,
        hidden         = 128,
    )
"""

import json
import os
import shutil
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.classification import CLASSES, ClassifierConfig, DEFAULT_CONFIG, COCO_SKELETON

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SKELETON_EDGES = COCO_SKELETON   # shared with classification.py

NUM_NODES    = 17
NODE_FEAT_DIM = 3   # (x_norm, y_norm, conf)
NUM_CLASSES  = len(CLASSES)   # 5
LABEL_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_LABEL = {i: cls for i, cls in enumerate(CLASSES)}

# Symmetric left-right keypoint pairs (COCO 17-point order).
# Used to mirror a pose graph for horizontal flip augmentation.
_FLIP_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8),
    (9, 10), (11, 12), (13, 14), (15, 16),
]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_edge_index():
    """Bidirectional edge index tensor for COCO skeleton. Shape [2, 2*E]."""
    src, dst = [], []
    for a, b in _SKELETON_EDGES:
        src += [a, b]
        dst += [b, a]
    return torch.tensor([src, dst], dtype=torch.long)


_EDGE_INDEX = _build_edge_index()   # cached, same for every patch


def keypoints_to_graph(kps: np.ndarray, bbox: np.ndarray) -> Data:
    """
    Convert a [17, 3] keypoints array and [4] bbox to a PyG Data object.

    Coordinates are normalised by bbox width/height so the graph is
    position-invariant across different patch sizes.

    Args:
        kps  : np.ndarray [17, 3] — (x, y, conf) in pixel coords
        bbox : np.ndarray [4]     — (x1, y1, x2, y2) in pixel coords

    Returns:
        PyG Data with x [17, 3] and edge_index [2, 2*E].
    """
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)

    feats = np.stack([
        (kps[:, 0] - x1) / w,   # x_norm  ∈ [0, 1] roughly
        (kps[:, 1] - y1) / h,   # y_norm  ∈ [0, 1] roughly
        kps[:, 2],               # confidence ∈ [0, 1]
    ], axis=1).astype(np.float32)

    return Data(
        x          = torch.from_numpy(feats),
        edge_index = _EDGE_INDEX.clone(),
    )


def _flip_graph(data: Data) -> Data:
    """Return a horizontally flipped copy of a pose graph.

    Flips x-coordinates (x_norm → 1 − x_norm) and swaps left/right
    keypoint pairs so the graph remains geometrically consistent.
    """
    x = data.x.clone()
    x[:, 0] = 1.0 - x[:, 0]
    for i, j in _FLIP_PAIRS:
        x[[i, j]] = x[[j, i]]
    return Data(x=x, edge_index=data.edge_index.clone())


# ---------------------------------------------------------------------------
# Keypoint caching
# ---------------------------------------------------------------------------

def save_keypoints(keypoints_dict: dict, npz_path: str):
    """
    Save keypoints dict to .npz.

    keypoints_dict : { fname: {"kps": np.ndarray [17,3], "bbox": np.ndarray [4]} }
                     Value is None if no detection.
    """
    fnames = list(keypoints_dict.keys())
    kps_list  = []
    bbox_list = []
    has_det   = []

    for fname in fnames:
        entry = keypoints_dict[fname]
        if entry is None:
            kps_list.append(np.zeros((17, 3), dtype=np.float32))
            bbox_list.append(np.zeros(4, dtype=np.float32))
            has_det.append(False)
        else:
            kps_list.append(entry["kps"].astype(np.float32))
            bbox_list.append(entry["bbox"].astype(np.float32))
            has_det.append(True)

    np.savez_compressed(
        npz_path,
        fnames  = np.array(fnames),
        kps     = np.stack(kps_list),
        bbox    = np.stack(bbox_list),
        has_det = np.array(has_det),
    )


def load_keypoints(npz_path: str) -> dict:
    """
    Load keypoints dict from .npz.

    Returns { fname: {"kps": np.ndarray [17,3], "bbox": np.ndarray [4]} | None }
    """
    data    = np.load(npz_path, allow_pickle=False)
    fnames  = data["fnames"].tolist()
    kps     = data["kps"]
    bbox    = data["bbox"]
    has_det = data["has_det"]

    return {
        fname: ({"kps": kps[i], "bbox": bbox[i]} if has_det[i] else None)
        for i, fname in enumerate(fnames)
    }


def extract_and_save_keypoints(
    pose_model,
    patch_dir: str,
    npz_path: str,
    batch_size: int = 32,
) -> dict:
    """
    Run pose inference on all patches in patch_dir and save to npz_path.
    Returns the keypoints dict.
    """
    import glob
    image_paths = sorted(
        glob.glob(os.path.join(patch_dir, "*.jpg")) +
        glob.glob(os.path.join(patch_dir, "*.png"))
    )

    keypoints_dict = {}
    n_batches = (len(image_paths) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(image_paths), batch_size),
                  total=n_batches, desc="Extracting keypoints", unit="batch"):
        batch_paths   = image_paths[i: i + batch_size]
        batch_results = pose_model(batch_paths, verbose=False)

        for img_path, result in zip(batch_paths, batch_results):
            fname = os.path.basename(img_path)
            if result.keypoints is None or result.keypoints.data.shape[0] == 0:
                keypoints_dict[fname] = None
                continue

            boxes    = result.boxes.xyxy.cpu().numpy()
            areas    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = int(np.argmax(areas))

            keypoints_dict[fname] = {
                "kps":  result.keypoints.data[best_idx].cpu().numpy(),   # [17, 3]
                "bbox": boxes[best_idx],                                   # [4]
            }

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    save_keypoints(keypoints_dict, npz_path)
    print(f"[GCN] Saved keypoints for {len(keypoints_dict)} patches → {npz_path}")
    return keypoints_dict


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_labels_from_rule_dir(cls_dir: str) -> dict:
    """
    Load labels from an init_classifications directory.
    Returns { fname: class_string }.
    """
    labels = {}
    for cls in CLASSES:
        subdir = os.path.join(cls_dir, cls)
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            if fname.lower().endswith((".jpg", ".png")):
                labels[fname] = cls
    return labels


def load_labels_from_manual_annotations(ann_path: str) -> dict:
    """
    Load labels from an annotations.json produced by annotate.py.
    Returns { fname: class_string } — excludes 'bad_extraction' entries.
    """
    with open(ann_path) as f:
        raw = json.load(f)
    return {
        fname: entry["label"]
        for fname, entry in raw.items()
        if entry["label"] in CLASSES
    }


# ---------------------------------------------------------------------------
# GCN model
# ---------------------------------------------------------------------------

class PoseGCN(nn.Module):
    """
    Three-layer GCN with confidence-weighted global pooling for graph-level
    classification of pose keypoint graphs.

    Architecture:
        GCNConv(3 → hidden)      → ReLU → Dropout
        GCNConv(hidden → hidden) → ReLU → Dropout
        GCNConv(hidden → hidden) → ReLU → Dropout
        confidence-weighted mean pool → Linear(hidden → num_classes)

    The extra conv layer helps back-facing classes which need more structural
    reasoning across the skeleton graph.
    """

    def __init__(self, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.conv1   = GCNConv(NODE_FEAT_DIM, hidden)
        self.conv2   = GCNConv(hidden, hidden)
        self.conv3   = GCNConv(hidden, hidden)
        self.dropout = dropout
        self.head    = nn.Linear(hidden, NUM_CLASSES)

    def forward(self, data: Data) -> torch.Tensor:
        if data.x is None:
            raise ValueError("Input graph is missing node features (data.x)")
        if data.edge_index is None:
            raise ValueError("Input graph is missing edge indices (data.edge_index)")

        x, edge_index = data.x, data.edge_index
        batch_vec = getattr(data, "batch", None)
        if batch_vec is None:
            batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Node embeddings — three GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Confidence-weighted global pooling per graph in batch.
        # Use raw input confidence (data.x[:, 2]) so pooling weights are
        # independent of the learned embeddings in x.
        raw_conf = data.x[:, 2].unsqueeze(1)               # [num_nodes, 1]

        weighted = x * raw_conf
        conf_sum = scatter(raw_conf, batch_vec, dim=0, reduce="sum")
        feat_sum = scatter(weighted, batch_vec, dim=0, reduce="sum")
        graph_emb = feat_sum / (conf_sum + 1e-8)

        return self.head(graph_emb)                         # [batch_size, num_classes]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for labels_batch, graphs_batch in loader:
        graphs_batch = graphs_batch.to(device)
        labels_batch = labels_batch.to(device)
        optimiser.zero_grad()
        logits = model(graphs_batch)
        loss   = criterion(logits, labels_batch)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * labels_batch.size(0)
        correct    += (logits.argmax(1) == labels_batch).sum().item()
        n          += labels_batch.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for labels_batch, graphs_batch in loader:
        graphs_batch = graphs_batch.to(device)
        labels_batch = labels_batch.to(device)
        logits = model(graphs_batch)
        loss   = criterion(logits, labels_batch)
        total_loss += loss.item() * labels_batch.size(0)
        correct    += (logits.argmax(1) == labels_batch).sum().item()
        n          += labels_batch.size(0)
    return total_loss / n, correct / n


class _LabelledDataset(torch.utils.data.Dataset):
    """Pairs (label_idx, graph) for training. augment=True adds H-flipped copies."""
    def __init__(self, labelled_items, keypoints_dict, augment: bool = False):
        base = [
            (LABEL_TO_IDX[label], keypoints_to_graph(
                keypoints_dict[fname]["kps"],
                keypoints_dict[fname]["bbox"],
            ))
            for fname, label in labelled_items
            if keypoints_dict.get(fname) is not None
        ]
        self.items = base + [(lbl, _flip_graph(g)) for lbl, g in base] if augment else base

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        label_idx, graph = self.items[idx]
        return torch.tensor(label_idx, dtype=torch.long), graph


def _collate(batch):
    from torch_geometric.data import Batch
    labels = torch.stack([b[0] for b in batch])
    graphs = Batch.from_data_list([b[1] for b in batch])
    return labels, graphs


# ---------------------------------------------------------------------------
# Training curves plot
# ---------------------------------------------------------------------------

def _save_training_plots(history: list, plot_dir: str):
    """
    Save a 2-panel seaborn figure of training curves to plot_dir.

    history : list of dicts with keys epoch, tr_loss, va_loss, tr_acc, va_acc.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.DataFrame(history)

    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # — Loss panel —
    loss_df = pd.melt(
        df, id_vars="epoch",
        value_vars=["tr_loss", "va_loss"],
        var_name="split", value_name="loss",
    )
    loss_df["split"] = loss_df["split"].map({"tr_loss": "train", "va_loss": "val"})
    sns.lineplot(data=loss_df, x="epoch", y="loss", hue="split", ax=axes[0])
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")

    # — Accuracy panel —
    acc_df = pd.melt(
        df, id_vars="epoch",
        value_vars=["tr_acc", "va_acc"],
        var_name="split", value_name="accuracy",
    )
    acc_df["split"] = acc_df["split"].map({"tr_acc": "train", "va_acc": "val"})
    sns.lineplot(data=acc_df, x="epoch", y="accuracy", hue="split", ax=axes[1])
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)

    fig.suptitle("GCN training curves", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(plot_dir, "training_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GCN] Training curves saved → {out_path}")


def train_gcn(
    labelled_items: list[tuple[str, str]],
    keypoints_dict: dict,
    device: str,
    hidden: int       = 128,
    dropout: float    = 0.3,
    lr: float         = 3e-4,
    weight_decay: float = 1e-4,
    epochs: int       = 150,
    batch_size: int   = 32,
    train_split: float= 0.8,
    seed: int         = 42,
    save_plots: bool  = True,
    plot_dir: str     = None,
) -> tuple:
    """
    Train PoseGCN on labelled_items = [(fname, class_string), ...].
    Returns (model, per_class_val_acc). Saves training-curve plots when
    save_plots=True (plot_dir must be provided).
    """
    # Stratified train/val split
    n_total_labelled = len(labelled_items)
    fnames_l = [f for f, _ in labelled_items if keypoints_dict.get(f) is not None]
    labels_l = [l for f, l in labelled_items if keypoints_dict.get(f) is not None]
    n_no_det = n_total_labelled - len(fnames_l)

    f_train, f_val, l_train, l_val = train_test_split(
        fnames_l, labels_l,
        test_size   = 1 - train_split,
        stratify    = labels_l,
        random_state= seed,
    )

    train_ds = _LabelledDataset(list(zip(f_train, l_train)), keypoints_dict, augment=True)
    val_ds   = _LabelledDataset(list(zip(f_val,   l_val)),   keypoints_dict, augment=False)

    # Oversample minority classes so they appear proportionally each epoch.
    # label_counts computed over the full labelled set (train+val); we want
    # weights that are inversely proportional to per-class frequency.
    label_counts = {cls: 0 for cls in CLASSES}
    for l in l_train:
        label_counts[l] += 1
    sample_weights = [
        1.0 / max(label_counts[IDX_TO_LABEL[item[0].item()]], 1)
        for item in train_ds
    ]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(train_ds), replacement=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, collate_fn=_collate)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, collate_fn=_collate)

    print(f"[GCN] GCN training: {len(train_ds)} train  {len(val_ds)} val  "
          f"({n_no_det} skipped - no detection)")

    # Class weights for the loss — computed over the full labelled pool
    # (train+val) to keep the loss scale stable regardless of split.
    full_label_counts = {cls: 0 for cls in CLASSES}
    for l in labels_l:
        full_label_counts[l] += 1
    total = sum(full_label_counts.values())
    weights = torch.tensor(
        [total / (NUM_CLASSES * max(full_label_counts[c], 1)) for c in CLASSES],
        dtype=torch.float32,
    ).to(device)

    model     = PoseGCN(hidden=hidden, dropout=dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_acc  = -1.0
    best_state    = None
    history       = []

    for epoch in tqdm(range(1, epochs + 1), desc="Training GCN", unit="epoch"):
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimiser, criterion, device)
        va_loss, va_acc = _eval_epoch(model,  val_loader,   criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "tr_loss": tr_loss, "va_loss": va_loss,
                        "tr_acc": tr_acc, "va_acc": va_acc})

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            print(f"[GCN] epoch {epoch:>3}  "
                  f"train loss {tr_loss:.4f}  acc {tr_acc:.3f}  |  "
                  f"val loss {va_loss:.4f}  acc {va_acc:.3f}")

    if save_plots and plot_dir:
        _save_training_plots(history, plot_dir)

    print(f"[GCN] Best val accuracy: {best_val_acc:.3f}")
    if best_state is None:
        raise ValueError("No checkpoint captured during training")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Per-class accuracy on val set to detect majority-class collapse.
    model.eval()
    per_class_correct = {cls: 0 for cls in CLASSES}
    per_class_total = {cls: 0 for cls in CLASSES}
    with torch.no_grad():
        for labels_batch, graphs_batch in val_loader:
            graphs_batch = graphs_batch.to(device)
            preds = model(graphs_batch).argmax(1).cpu()
            for pred, gt in zip(preds.tolist(), labels_batch.tolist()):
                cls = IDX_TO_LABEL[gt]
                per_class_total[cls] += 1
                per_class_correct[cls] += int(pred == gt)
    print("[GCN] Val accuracy per class:")
    per_class_val_acc = {}
    for cls in CLASSES:
        n = per_class_total[cls]
        if n:
            acc = per_class_correct[cls] / n
            per_class_val_acc[cls] = {"correct": per_class_correct[cls], "total": n, "acc": acc}
            print(f"[GCN] {cls:<25} {per_class_correct[cls]}/{n}  ({100*acc:.1f}%)")
        else:
            per_class_val_acc[cls] = {"correct": 0, "total": 0, "acc": None}

    return model, per_class_val_acc


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: PoseGCN,
    all_fnames: list[str],
    keypoints_dict: dict,
    device: str,
    batch_size: int = 64,
) -> dict:
    """
    Run GCN inference over all patches.
    Patches with no detection are assigned 'others' without going through the model.

    Returns { fname: predicted_class_string }.
    """
    model.eval()

    results   = {}
    no_det    = [f for f in all_fnames if keypoints_dict.get(f) is None]
    has_det   = [f for f in all_fnames if keypoints_dict.get(f) is not None]

    for fname in no_det:
        results[fname] = "no_pose"

    # Build graphs for patches with detections
    graphs = [
        keypoints_to_graph(keypoints_dict[f]["kps"], keypoints_dict[f]["bbox"])
        for f in has_det
    ]

    for i in tqdm(range(0, len(graphs), batch_size),
                  total=(len(graphs) + batch_size - 1) // batch_size,
                  desc="GCN inference", unit="batch"):
        batch_fnames = has_det[i: i + batch_size]
        graph_slice = cast(list[Any], graphs[i: i + batch_size])
        batch_graphs = cast(Any, Batch.from_data_list(graph_slice)).to(device)

        logits = model(batch_graphs)
        preds   = logits.argmax(dim=1).cpu().tolist()

        for fname, pred in zip(batch_fnames, preds):
            results[fname] = IDX_TO_LABEL[pred]

    return results


# ---------------------------------------------------------------------------
# Save results (mirrors init_classifications format)
# ---------------------------------------------------------------------------

def save_gcn_results(
    results: dict,
    src_dir: str,
    save_dir: str,
    per_class_val_acc: dict = None,
) -> dict:
    """
    Copy patches into per-class subdirs under save_dir and write _summary.txt.

    Args:
        results           : { fname: class_string }
        src_dir           : directory containing the original patch files
        save_dir          : output directory (output/gcn_results/<run>/)
        per_class_val_acc : optional dict from train_gcn —
                            { cls: {"correct": int, "total": int, "acc": float|None} }

    Returns:
        summary dict { class_string: count }
    """
    for cls in CLASSES:
        os.makedirs(os.path.join(save_dir, cls), exist_ok=True)

    summary = {cls: 0 for cls in CLASSES}

    for fname, cls in tqdm(results.items(), desc="Saving GCN results", unit="patch"):
        src_path = os.path.join(src_dir, fname)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(save_dir, cls, fname))
        summary[cls] += 1

    total = sum(summary.values())
    summary_lines = [
        "GCN classification summary",
        f"Source patches : {src_dir}",
        f"Total patches  : {total}",
        "",
        f"{'Class':<25} {'Count':>6}  {'%':>6}",
        "-" * 42,
    ]
    for cls, count in sorted(summary.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        summary_lines.append(f"{cls:<25} {count:>6}  {pct:>5.1f}%")

    if per_class_val_acc:
        summary_lines += [
            "",
            "Val accuracy per class",
            f"{'Class':<25} {'Correct':>7}  {'Total':>5}  {'Acc':>6}",
            "-" * 48,
        ]
        for cls in CLASSES:
            entry = per_class_val_acc.get(cls, {})
            n = entry.get("total", 0)
            if n:
                correct = entry["correct"]
                acc = 100 * entry["acc"]
                summary_lines.append(f"{cls:<25} {correct:>7}  {n:>5}  {acc:>5.1f}%")
            else:
                summary_lines.append(f"{cls:<25} {'—':>7}  {'—':>5}  {'—':>6}")

    summary_text = "\n".join(summary_lines)
    print("[GCN] " + summary_text)
    summary_path = os.path.join(save_dir, "_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"[GCN] Summary saved to {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------

def reload_gcn_results(gcn_save_path: str) -> tuple[dict, dict]:
    """
    Reload GCN results from an existing gcn_results directory.
    Same interface as reload_classification_results.
    """
    from src.classification import reload_classification_results
    return reload_classification_results(gcn_save_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_gcn_pipeline(
    labelled_dir: str,
    cls_source: str,
    all_patches_dir: str,
    save_dir: str,
    pose_model_path: str,
    device: str,
    # GCN hyperparams
    hidden: int        = 128,
    dropout: float     = 0.3,
    lr: float          = 3e-4,
    weight_decay: float= 1e-4,
    epochs: int        = 150,
    batch_size: int    = 32,
    train_split: float = 0.8,
    seed: int          = 42,
    exclude_classes: list[str] | None = None,
    # Keypoint cache
    force_reextract_keypoints: bool = False,
    # Plots
    save_plots: bool = True,
) -> tuple[dict, dict, dict]:
    """
    Full GCN classification pipeline.

    Args:
        labelled_dir   : Path to init_classifications/<run>/ or
                         manual_annotated/<run>/ dir containing labels.
        cls_source     : "rule" — load from init_classifications subdir structure.
                         "manual" — load from annotations.json in labelled_dir.
        all_patches_dir: Path to extracted_humans/<ts>/ containing all patches
                         and (optionally) _keypoints.npz.
        save_dir       : Output dir for GCN results (output/gcn_results/<run>/).
        pose_model_path: Path to YOLO pose model weights.
        device         : "cuda" | "mps" | "cpu".
        exclude_classes: Optional class names to exclude from training labels.
        force_reextract_keypoints: Re-run pose inference even if .npz exists.
        save_plots     : If True (default), save training_curves.png to save_dir.

    Returns:
        results           : { fname: class_string }  — for ALL patches
        summary           : { class_string: count }
        per_class_val_acc : { cls: {correct, total, acc} } from training
    """
    import glob

    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Load labels for training ---
    print(f"\n[GCN] Loading labels from {labelled_dir} (source={cls_source})")
    if cls_source == "manual":
        ann_path = os.path.join(labelled_dir, "annotations.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"annotations.json not found in {labelled_dir}")
        labelled = load_labels_from_manual_annotations(ann_path)
    elif cls_source == "rule":
        labelled = load_labels_from_rule_dir(labelled_dir)
    else:
        raise ValueError(f"cls_source must be 'manual' or 'rule', got '{cls_source}'")

    print(f"[GCN] {len(labelled)} labelled patches loaded")
    if exclude_classes:
        excluded_set = set(exclude_classes)
        before_count = len(labelled)
        labelled = {
            fname: cls
            for fname, cls in labelled.items()
            if cls not in excluded_set
        }
        print(
            f"[GCN] Excluded {before_count - len(labelled)} labelled patches "
            f"from classes: {sorted(excluded_set)}"
        )
        if not labelled:
            raise ValueError(
                "No labelled patches remain after applying exclude_classes"
            )

    # --- 2. Load or extract keypoints ---
    npz_path = os.path.join(all_patches_dir, "_keypoints.npz")

    if os.path.exists(npz_path) and not force_reextract_keypoints:
        print(f"[GCN] Loading cached keypoints from {npz_path}")
        keypoints_dict = load_keypoints(npz_path)
    else:
        print(f"[GCN] Extracting keypoints (no cache found or force_reextract=True)")
        from ultralytics import YOLO
        pose_model = YOLO(pose_model_path)
        pose_model.to(device)
        keypoints_dict = extract_and_save_keypoints(
            pose_model, all_patches_dir, npz_path
        )

    # --- 3. All patch filenames ---
    all_fnames = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(all_patches_dir, "*.jpg")) +
                 glob.glob(os.path.join(all_patches_dir, "*.png"))
    )
    print(f"[GCN] Total patches to classify: {len(all_fnames)}")

    # --- 4. Train ---
    labelled_items = list(labelled.items())
    model, per_class_val_acc = train_gcn(
        labelled_items = labelled_items,
        keypoints_dict = keypoints_dict,
        device         = device,
        hidden         = hidden,
        dropout        = dropout,
        lr             = lr,
        weight_decay   = weight_decay,
        epochs         = epochs,
        batch_size     = batch_size,
        train_split    = train_split,
        seed           = seed,
        save_plots     = save_plots,
        plot_dir       = save_dir,
    )

    # Save model checkpoint (named after the run dir so it's traceable)
    save_name = os.path.basename(save_dir)
    ckpt_path = os.path.join(save_dir, f"gcn_model_{save_name}.pt")
    torch.save({"state_dict": model.state_dict(), "hidden": hidden}, ckpt_path)
    print(f"[GCN] Model checkpoint saved → {ckpt_path}")

    # --- 5. Inference on all patches ---
    results = run_inference(model, all_fnames, keypoints_dict, device)

    # --- 6. Save results ---
    summary = save_gcn_results(results, all_patches_dir, save_dir, per_class_val_acc)

    return results, summary, per_class_val_acc


# ---------------------------------------------------------------------------
# Ablation plot
# ---------------------------------------------------------------------------

def load_gcn_model(gcn_save_path: str, device: str) -> PoseGCN:
    """
    Load a trained PoseGCN checkpoint from a gcn_results directory.

    Searches for gcn_model_*.pt inside gcn_save_path. The hidden dimension is
    inferred from the checkpoint dict (new format); legacy bare state-dicts fall
    back to hidden=128.

    Args:
        gcn_save_path : directory containing a gcn_model_<run>.pt checkpoint.
        device        : "cuda" | "mps" | "cpu".

    Returns:
        PoseGCN in eval mode on the given device.
    """
    import glob as _glob
    matches = sorted(_glob.glob(os.path.join(gcn_save_path, "gcn_model_*.pt")))
    if not matches:
        raise FileNotFoundError(f"No gcn_model_*.pt checkpoint found in {gcn_save_path}")
    ckpt_path = matches[0]
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        hidden = payload["hidden"]
        state_dict = payload["state_dict"]
    else:
        hidden = 128  # legacy bare state-dict
        state_dict = payload
    model = PoseGCN(hidden=hidden)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[GCN] Loaded checkpoint from {ckpt_path} (hidden={hidden})")
    return model


def run_gcn_inference_pretrained(
    ckpt_path: str,
    extract_save_path: str,
    gcn_save_path: str,
    pose_model_path: str,
    device: str,
) -> tuple[dict, dict]:
    """
    Run GCN inference using a committed pretrained checkpoint on freshly
    extracted patches, without retraining.

    Copies ckpt_path into gcn_save_path (preserving filename), then classifies
    every patch. The hidden dimension is inferred from the checkpoint.

    Args:
        ckpt_path         : committed .pt checkpoint (e.g. checkpoints/gcn_model_<run>.pt).
        extract_save_path : extracted_humans/<run>/ directory.
        gcn_save_path     : output/gcn_results/<run>/ directory to write into.
        pose_model_path   : YOLO pose weights used to extract keypoints if not cached.
        device            : "cuda" | "mps" | "cpu".

    Returns:
        (results, summary) — same structure as run_gcn_pipeline.
    """
    import glob as _glob
    import shutil as _shutil
    from ultralytics import YOLO as _YOLO

    os.makedirs(gcn_save_path, exist_ok=True)
    _shutil.copy(ckpt_path, os.path.join(gcn_save_path, os.path.basename(ckpt_path)))

    npz_path = os.path.join(extract_save_path, "_keypoints.npz")
    if os.path.exists(npz_path):
        keypoints_dict = load_keypoints(npz_path)
        print(f"[GCN] Loaded cached keypoints ({len(keypoints_dict)} patches)")
    else:
        _pose_model = _YOLO(pose_model_path)
        _pose_model.to(device)
        keypoints_dict = extract_and_save_keypoints(_pose_model, extract_save_path, npz_path)

    gcn_model = load_gcn_model(gcn_save_path, device)
    all_fnames = sorted(
        os.path.basename(p)
        for p in _glob.glob(os.path.join(extract_save_path, "*.jpg"))
             + _glob.glob(os.path.join(extract_save_path, "*.png"))
    )
    results = run_inference(gcn_model, all_fnames, keypoints_dict, device)
    summary = save_gcn_results(results, extract_save_path, gcn_save_path, per_class_val_acc=None)
    print(f"[GCN] Inference complete (pretrained ckpt): {sum(summary.values())} patches classified")
    return results, summary


def plot_annotation_ablation(
    rule_per_class_val_acc: dict,
    manual_per_class_val_acc: dict,
    save_path: str,
):
    """
    Side-by-side per-class val accuracy barplot comparing GCN trained on
    rule-based vs manual annotations.  Saves PNG to save_path.

    Args:
        rule_per_class_val_acc   : per_class_val_acc dict from run_gcn_pipeline
                                   with cls_source='rule'.
        manual_per_class_val_acc : same structure for cls_source='manual'.
        save_path                : full path for the output PNG.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    rows = []
    for cls in CLASSES:
        for src, acc_dict in (("rule", rule_per_class_val_acc),
                              ("manual", manual_per_class_val_acc)):
            entry = acc_dict.get(cls, {})
            acc   = entry.get("acc")
            rows.append({
                "class":        cls,
                "label_source": src,
                "val_accuracy": acc * 100 if acc is not None else float("nan"),
            })

    df = pd.DataFrame(rows)

    sns.set_theme(style="darkgrid", palette="muted")
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=df,
        x="class",
        y="val_accuracy",
        hue="label_source",
        ax=ax,
    )

    # Annotate bars with value
    for bar in ax.patches:
        h = bar.get_height()
        if not (h != h):   # skip NaN bars
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.8,
                f"{h:.1f}%",
                ha="center", va="bottom",
                fontsize=8, color="#333333",
            )

    ax.set_title("GCN val accuracy per class: rule-based vs manual labels",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Val accuracy (%)")
    ax.set_ylim(0, 110)
    ax.legend(title="Label source")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GCN] Ablation plot saved → {save_path}")