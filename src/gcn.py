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
   # TODO: try attention pooling if more labelled data becomes available, or
   #       if confidence-weighted pooling proves insufficiently expressive.
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

from src.classification import CLASSES, ClassifierConfig, DEFAULT_CONFIG

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# COCO skeleton edges (0-indexed) — same as classification.py _SKELETON
_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

NUM_NODES    = 17
NODE_FEAT_DIM = 3   # (x_norm, y_norm, conf)
NUM_CLASSES  = len(CLASSES)   # 5
LABEL_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_LABEL = {i: cls for i, cls in enumerate(CLASSES)}


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
    print(f"Saved keypoints for {len(keypoints_dict)} patches → {npz_path}")
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
    Two-layer GCN with confidence-weighted global pooling for graph-level
    classification of pose keypoint graphs.

    Architecture:
        GCNConv(3 → hidden) → ReLU → Dropout
        GCNConv(hidden → hidden) → ReLU → Dropout
        confidence-weighted mean pool → Linear(hidden → num_classes)

    # TODO: consider attention pooling (e.g. GlobalAttention from torch_geometric)
    #       if more labelled data becomes available and confidence-weighted pooling
    #       proves insufficiently expressive.
    """

    def __init__(self, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1   = GCNConv(NODE_FEAT_DIM, hidden)
        self.conv2   = GCNConv(hidden, hidden)
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

        # Node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Confidence-weighted global pooling per graph in batch.
        conf = x[:, 2].unsqueeze(1)                         # [num_nodes, 1]

        weighted = x * conf
        conf_sum = scatter(conf, batch_vec, dim=0, reduce="sum")
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
    """Pairs (label_idx, graph) for training."""
    def __init__(self, labelled_items, keypoints_dict):
        self.items = [
            (LABEL_TO_IDX[label], keypoints_to_graph(
                keypoints_dict[fname]["kps"],
                keypoints_dict[fname]["bbox"],
            ))
            for fname, label in labelled_items
            if keypoints_dict.get(fname) is not None
        ]

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


def train_gcn(
    labelled_items: list[tuple[str, str]],
    keypoints_dict: dict,
    device: str,
    hidden: int       = 64,
    dropout: float    = 0.3,
    lr: float         = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int       = 60,
    batch_size: int   = 32,
    train_split: float= 0.8,
    seed: int         = 42,
) -> PoseGCN:
    """
    Train PoseGCN on labelled_items = [(fname, class_string), ...].
    Returns the trained model (best val-acc checkpoint).
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

    train_ds = _LabelledDataset(list(zip(f_train, l_train)), keypoints_dict)
    val_ds   = _LabelledDataset(list(zip(f_val,   l_val)),   keypoints_dict)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  collate_fn=_collate)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, collate_fn=_collate)

    print(f"GCN training: {len(train_ds)} train  {len(val_ds)} val  "
          f"({n_no_det} skipped - no detection)")

    # Class weights to handle imbalance
    label_counts = {cls: 0 for cls in CLASSES}
    for l in labels_l:
        label_counts[l] += 1
    total = sum(label_counts.values())
    weights = torch.tensor(
        [total / (NUM_CLASSES * max(label_counts[c], 1)) for c in CLASSES],
        dtype=torch.float32,
    ).to(device)

    model     = PoseGCN(hidden=hidden, dropout=dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_acc  = -1.0
    best_state    = None

    for epoch in tqdm(range(1, epochs + 1), desc="Training GCN", unit="epoch"):
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimiser, criterion, device)
        va_loss, va_acc = _eval_epoch(model,  val_loader,   criterion, device)
        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  epoch {epoch:>3}  "
                  f"train loss {tr_loss:.4f}  acc {tr_acc:.3f}  |  "
                  f"val loss {va_loss:.4f}  acc {va_acc:.3f}")

    print(f"Best val accuracy: {best_val_acc:.3f}")
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
    print("  Val accuracy per class:")
    for cls in CLASSES:
        n = per_class_total[cls]
        if n:
            acc = 100 * per_class_correct[cls] / n
            print(f"    {cls:<25} {per_class_correct[cls]}/{n}  ({acc:.1f}%)")

    return model


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
        results[fname] = "others"

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
) -> dict:
    """
    Copy patches into per-class subdirs under save_dir and write _summary.txt.

    Args:
        results  : { fname: class_string }
        src_dir  : directory containing the original patch files
        save_dir : output directory (output/gcn_results/<run>/)

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

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(os.path.join(save_dir, "_summary.txt"), "w") as f:
        f.write(summary_text + "\n")

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
    hidden: int        = 64,
    dropout: float     = 0.3,
    lr: float          = 1e-3,
    weight_decay: float= 1e-4,
    epochs: int        = 60,
    batch_size: int    = 32,
    train_split: float = 0.8,
    seed: int          = 42,
    # Keypoint cache
    force_reextract_keypoints: bool = False,
) -> tuple[dict, dict]:
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
        force_reextract_keypoints: Re-run pose inference even if .npz exists.

    Returns:
        results : { fname: class_string }  — for ALL patches
        summary : { class_string: count }
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
    model = train_gcn(
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
    )

    # Save model checkpoint
    ckpt_path = os.path.join(save_dir, "gcn_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[GCN] Model checkpoint saved → {ckpt_path}")

    # --- 5. Inference on all patches ---
    results = run_inference(model, all_fnames, keypoints_dict, device)

    # --- 6. Save results ---
    summary = save_gcn_results(results, all_patches_dir, save_dir)

    return results, summary