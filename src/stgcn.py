"""
src/stgcn.py

Spatio-Temporal GCN (ST-GCN, Yan et al. 2018) for video-consistent
patch classification in Q2.2.

Extends the per-frame PoseGCN (src/gcn.py) to temporal windows of T frames
by adding temporal edges connecting the same keypoint across consecutive
frames.  The GCNConv weights in PoseGCN are T-agnostic (they operate on
arbitrary graphs), so the same trained checkpoint classifies single-frame or
multi-frame graphs without modification — only the edge_index changes.

Graph for a T-frame window (one tracked person):
  nodes         : 17 * T  — node  t*17 + k  =  keypoint k at frame t
  spatial edges : COCO skeleton replicated per frame  (16 bones × 2 dirs × T)
  temporal edges: bidirectional (t*17+k) ↔ ((t+1)*17+k), k∈[0,17), t∈[0,T-2)

Node features : (x_norm, y_norm, conf) — bbox-normalised, matching gcn.py.

Tracking: a lightweight IoU-based greedy tracker groups per-frame detections
into tracks so consecutive frames of the same person can be windowed together.
No external dependency (no SORT/ByteTrack required).

Based on: Yan et al., "Spatial Temporal Graph Convolutional Networks for
Skeleton-Based Action Recognition", AAAI 2018.
Inspired by the mmskeleton reference implementation (open-mmlab/mmskeleton)
but re-implemented here from scratch using torch_geometric for DRY integration
with the existing PoseGCN training/inference pipeline.

Public API
----------
  build_st_edge_index(T)                                  -> LongTensor
  sequence_to_stgraph(kps_seq, bbox_seq)                  -> Data
  assign_track_ids(crop_metadata, iou_threshold)          -> dict
  run_stgcn_inference(model, crop_metadata,
                      keypoints_dict, device, T, ...)     -> dict
"""

from typing import Any, cast

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from src.gcn import IDX_TO_LABEL, NUM_NODES, _SKELETON_EDGES

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_st_edge_index(T: int) -> torch.Tensor:
    """
    Bidirectional spatial + temporal edge index for a T-frame pose sequence.

    Node layout: frame t, keypoint k  →  node index  t * NUM_NODES + k

    Spatial  : COCO skeleton within each frame     (T * 2*16 directed edges)
    Temporal : same keypoint, consecutive frames   (2*(T-1)*17 directed edges)

    Returns LongTensor [2, num_edges].
    """
    src, dst = [], []

    # Spatial edges — COCO skeleton replicated once per frame
    for t in range(T):
        off = t * NUM_NODES
        for a, b in _SKELETON_EDGES:
            src += [off + a, off + b]
            dst += [off + b, off + a]

    # Temporal edges — same keypoint between consecutive frames (bidirectional)
    for t in range(T - 1):
        for k in range(NUM_NODES):
            n1 = t * NUM_NODES + k
            n2 = (t + 1) * NUM_NODES + k
            src += [n1, n2]
            dst += [n2, n1]

    return torch.tensor([src, dst], dtype=torch.long)


def sequence_to_stgraph(
    kps_seq: np.ndarray,   # [T, 17, 3]  (x, y, conf) in pixel coords
    bbox_seq: np.ndarray,  # [T, 4]      (x1, y1, x2, y2) per frame
) -> Data:
    """
    Build a spatio-temporal PyG graph for a T-frame person-track window.

    Each frame's keypoints are normalised by that frame's bbox independently
    (matching gcn.keypoints_to_graph).  The T normalised frames are then
    concatenated into a single 17*T-node graph.

    Args:
        kps_seq  : [T, 17, 3] — (x, y, conf) in pixel coordinates
        bbox_seq : [T, 4]     — (x1, y1, x2, y2) in pixel coordinates

    Returns:
        PyG Data with  x [17*T, 3]  and  edge_index [2, E].
    """
    T = kps_seq.shape[0]
    feat_frames = []
    for t in range(T):
        x1, y1, x2, y2 = bbox_seq[t]
        w = max(float(x2 - x1), 1.0)
        h = max(float(y2 - y1), 1.0)
        kps = kps_seq[t]
        feats = np.stack([
            (kps[:, 0] - x1) / w,
            (kps[:, 1] - y1) / h,
            kps[:, 2],
        ], axis=1).astype(np.float32)        # [17, 3]
        feat_frames.append(feats)

    x = torch.from_numpy(np.concatenate(feat_frames, axis=0))  # [17*T, 3]
    return Data(x=x, edge_index=build_st_edge_index(T))


# ---------------------------------------------------------------------------
# Lightweight IoU tracker (no external dependency)
# ---------------------------------------------------------------------------

def _iou(b1, b2) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    a1 = max(b1[2] - b1[0], 0) * max(b1[3] - b1[1], 0)
    a2 = max(b2[2] - b2[0], 0) * max(b2[3] - b2[1], 0)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def assign_track_ids(
    crop_metadata: list[dict],
    iou_threshold: float = 0.3,
) -> dict[str, int]:
    """
    Greedy IoU tracker over crop_metadata (sorted by frame_idx).

    Each detection is matched to the highest-IoU unmatched detection from the
    previous frame; unmatched detections start a new track.

    Returns {crop_stem: track_id}.
    """
    by_frame: dict[int, list[dict]] = {}
    for m in crop_metadata:
        by_frame.setdefault(m["frame_idx"], []).append(m)

    next_id = 0
    stem_to_track: dict[str, int] = {}
    prev_tracks: list[tuple[int, tuple]] = []   # [(track_id, bbox)]

    for fidx in sorted(by_frame):
        matched = set()
        new_prev = []
        for det in by_frame[fidx]:
            bbox = tuple(det["bbox"])
            best_iou, best_tid = 0.0, -1
            for tid, prev_bbox in prev_tracks:
                if tid in matched:
                    continue
                s = _iou(bbox, prev_bbox)
                if s > best_iou:
                    best_iou, best_tid = s, tid
            if best_iou >= iou_threshold:
                tid = best_tid
                matched.add(tid)
            else:
                tid = next_id
                next_id += 1
            stem_to_track[det["crop_stem"]] = tid
            new_prev.append((tid, bbox))
        prev_tracks = new_prev

    return stem_to_track


# ---------------------------------------------------------------------------
# ST-GCN inference
# ---------------------------------------------------------------------------

def run_stgcn_inference(
    model,                        # PoseGCN — weights are T-agnostic
    crop_metadata: list[dict],    # [{frame_idx, crop_stem, bbox, ...}]
    keypoints_dict: dict,         # {"<stem>.jpg": {"kps": [17,3], "bbox": [4]}} | None
    device: str,
    T: int = 5,
    iou_threshold: float = 0.3,
    batch_size: int = 32,
) -> dict:
    """
    Classify each patch using a T-frame spatio-temporal window.

    For each detection the T most-recent frames of the same track are used to
    build an ST-graph; shorter tracks are front-padded with the earliest
    available frame.  Patches with no keypoint detection in their window fall
    back to zero-confidence keypoints (model typically predicts 'others').

    Args:
        model          : trained PoseGCN (from src.gcn).
        crop_metadata  : patch list from _extract_test_patches, ordered by frame.
        keypoints_dict : keyed by "<stem>.jpg"; None value = no detection.
        device         : torch device string.
        T              : temporal window length (frames).
        iou_threshold  : IoU threshold for cross-frame detection matching.
        batch_size     : graphs per forward pass.

    Returns:
        {"<stem>.jpg": predicted_class_string}
    """
    model.eval()

    stem_to_track = assign_track_ids(crop_metadata, iou_threshold)

    # Per-track ordered history: {track_id: [(stem, kps [17,3], bbox [4]), ...]}
    track_history: dict[int, list] = {}
    for m in sorted(crop_metadata, key=lambda x: x["frame_idx"]):
        stem  = m["crop_stem"]
        tid   = stem_to_track.get(stem, -1)
        entry = keypoints_dict.get(stem + ".jpg")
        if entry is not None:
            kps  = entry["kps"]
            bbox = entry["bbox"]
        else:
            bbox = np.array(m["bbox"], dtype=np.float32)
            kps  = np.zeros((NUM_NODES, 3), dtype=np.float32)
        track_history.setdefault(tid, []).append((stem, kps, bbox))

    # Map each stem to its position within its track
    stem_position: dict[str, int] = {}
    for entries in track_history.values():
        for pos, (stem, _, _) in enumerate(entries):
            stem_position[stem] = pos

    # Build one ST-graph per detection
    graphs: list[Data] = []
    stems_ordered: list[str] = []

    for m in crop_metadata:
        stem = m["crop_stem"]
        tid  = stem_to_track.get(stem, -1)
        pos  = stem_position.get(stem, 0)
        hist = track_history.get(tid)
        if not hist:
            hist = [(stem, np.zeros((NUM_NODES, 3), np.float32),
                     np.zeros(4, np.float32))]

        # T most-recent frames ending at current position; front-pad if needed
        start  = max(0, pos - T + 1)
        window = hist[start: pos + 1]
        if len(window) < T:
            window = [window[0]] * (T - len(window)) + window

        kps_seq  = np.stack([w[1] for w in window])   # [T, 17, 3]
        bbox_seq = np.stack([w[2] for w in window])   # [T, 4]
        graphs.append(sequence_to_stgraph(kps_seq, bbox_seq))
        stems_ordered.append(stem)

    results: dict[str, str] = {}

    with torch.no_grad():
        for i in tqdm(
            range(0, len(graphs), batch_size),
            total=(len(graphs) + batch_size - 1) // batch_size,
            desc="ST-GCN inference", unit="batch",
        ):
            batch_stems  = stems_ordered[i: i + batch_size]
            graph_slice  = cast(list[Any], graphs[i: i + batch_size])
            batch_graphs = cast(Any, Batch.from_data_list(graph_slice)).to(device)

            logits = model(batch_graphs)
            preds  = logits.argmax(dim=1).cpu().tolist()

            for stem, pred in zip(batch_stems, preds):
                results[stem + ".jpg"] = IDX_TO_LABEL[pred]

    return results
