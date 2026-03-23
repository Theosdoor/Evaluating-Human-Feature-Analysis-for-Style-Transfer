"""
src/baseline_model.py
Question 2.1 — Image Model Deployment

Pipeline
--------
1. ensure_pretrained_models  — download CUT pretrained checkpoints if absent.
2. build_frame_dataset       — extract full frames at 1.1 detection timestamps
                               into CUT's trainA/trainB layout.
3. finetune_cut_fullframe    — fine-tune the pretrained checkpoint on full
                               game/movie frames for a small number of epochs.
4. translate_test_video      — apply the (fine-tuned) model to every frame of
                               the test video and write the output mp4.
5. compute_metrics           — FID, KID, LPIPS for a translated image set.
6. save_comparison_grid      — side-by-side input/translated figure.
7. save_umap                 — VGG16 feature UMAP across domain groups.

Shared helpers (video I/O, CUT subprocess, fine-tuning) live in src/utils.py.

Public API imported in nb_main.py:
    ensure_pretrained_models, build_frame_dataset, finetune_cut_fullframe,
    translate_test_video, compute_metrics, save_comparison_grid, save_umap,
    make_inference_dataroot, PRETRAINED_MODELS
"""

import glob
import json
import os
import shutil
import subprocess
import tempfile

import cv2
import lpips as lpips_lib
import numpy as np
import torch
from cleanfid import fid
from tqdm import tqdm

from src.utils import extract_video_frames, write_video, run_cut_inference, finetune_cut


PRETRAINED_URL = "http://efrosgans.eecs.berkeley.edu/CUT/pretrained_models.tar"

PRETRAINED_MODELS = [
    "cityscapes_cut_pretrained",
    "cityscapes_fastcut_pretrained",
    "horse2zebra_cut_pretrained",
    "horse2zebra_fastcut_pretrained",
    "cat2dog_cut_pretrained",
    "cat2dog_fastcut_pretrained",
]


# ---------------------------------------------------------------------------
# Pretrained model setup
# ---------------------------------------------------------------------------

def ensure_pretrained_models(cut_dir: str) -> None:
    """Download and extract pretrained CUT checkpoints if not already present."""
    tar_path = os.path.join(cut_dir, "pretrained_models.tar")
    probe    = os.path.join(
        cut_dir, "checkpoints", "horse2zebra_cut_pretrained", "latest_net_G.pth"
    )

    if not os.path.exists(tar_path):
        print(f"[CUT] Downloading pretrained archive → {tar_path}")
        subprocess.run(["wget", "-O", tar_path, PRETRAINED_URL], check=True)

    if not os.path.exists(probe):
        print("[CUT] Extracting pretrained archive…")
        subprocess.run(["tar", "-xf", tar_path], check=True, cwd=cut_dir)
    else:
        print("[CUT] Pretrained checkpoints already present.")


# ---------------------------------------------------------------------------
# Dataset preparation — full frames at 1.1 detection timestamps
# ---------------------------------------------------------------------------

def build_frame_dataset(
    selected_detections: list[dict],
    train_paths: list[str],
    data_dir: str,
    n_per_domain: int = 500,
) -> tuple[str, str, str, str]:
    """
    Extract full frames at the timestamps from 1.1 and organise into
    CUT's trainA (game) / trainB (movie) layout.

    Returns (trainA_dir, trainB_dir, testA_dir, testB_dir).
    """
    game_paths  = {p for p in train_paths
                   if any(kw in p.lower() for kw in ["game", "mafia"])}
    movie_paths = {p for p in train_paths if p not in game_paths}

    trainA = os.path.join(data_dir, "trainA")
    trainB = os.path.join(data_dir, "trainB")
    testA  = os.path.join(data_dir, "testA")
    testB  = os.path.join(data_dir, "testB")

    print("[CUT] Building frame dataset…")
    a_rebuilt = _extract_frames(selected_detections, game_paths,  trainA, n_per_domain)
    b_rebuilt = _extract_frames(selected_detections, movie_paths, trainB, n_per_domain)

    _make_held_out_split(trainA, testA, force=a_rebuilt)
    _make_held_out_split(trainB, testB, force=b_rebuilt)

    print(f"[CUT] trainA (game):  {len(glob.glob(os.path.join(trainA, '*.jpg')))} frames")
    print(f"[CUT] trainB (movie): {len(glob.glob(os.path.join(trainB, '*.jpg')))} frames")
    return trainA, trainB, testA, testB


def _extract_frames(
    detections: list[dict],
    video_paths_for_domain: set,
    output_dir: str,
    n_frames: int,
) -> bool:
    """Extract n_frames full frames from the given domain's videos. Returns True if rebuilt."""
    os.makedirs(output_dir, exist_ok=True)
    existing = len(glob.glob(os.path.join(output_dir, "*.jpg")))
    if existing >= n_frames:
        print(f"[CUT] {os.path.basename(output_dir)}: {existing} frames present, skipping.")
        return False

    domain_dets = [d for d in detections if d["video_path"] in video_paths_for_domain]

    # Deduplicate by (video_path, frame_num) so multiple detections in one
    # frame don't produce filename collisions.
    seen: set = set()
    deduped = []
    for d in domain_dets:
        key = (d["video_path"], d["frame_num"])
        if key not in seen:
            seen.add(key)
            deduped.append(d)

    step        = max(1, len(deduped) // n_frames)
    domain_dets = deduped[::step][:n_frames]

    by_video: dict[str, list] = {}
    for det in domain_dets:
        by_video.setdefault(det["video_path"], []).append(det)

    saved = 0
    for vpath, dets in by_video.items():
        dets_sorted = sorted(dets, key=lambda d: d["frame_num"])
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"[CUT] Warning: cannot open {vpath}")
            continue
        cur = 0
        for det in tqdm(dets_sorted, desc=f"  {os.path.basename(vpath)}", leave=False):
            target = det["frame_num"]
            if target < cur:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                cur = target
            while cur < target:
                cap.grab()
                cur += 1
            ret, frame = cap.read()
            if not ret:
                continue
            cur += 1
            tag   = os.path.splitext(os.path.basename(vpath))[0]
            fname = f"{tag}_f{target:06d}.jpg"
            cv2.imwrite(
                os.path.join(output_dir, fname),
                cv2.resize(frame, (286, 286)),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            saved += 1
        cap.release()

    print(f"[CUT] Saved {saved} frames → {output_dir}")
    return True


def _make_held_out_split(
    src_dir: str,
    dst_dir: str,
    n: int = 200,
    force: bool = False,
) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    if not force and len(glob.glob(os.path.join(dst_dir, "*.jpg"))) >= n:
        return
    if force:
        for f in glob.glob(os.path.join(dst_dir, "*.jpg")):
            os.remove(f)
    frames = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))[-n:]
    for f in frames:
        shutil.copy(f, os.path.join(dst_dir, os.path.basename(f)))


def make_inference_dataroot(domain_dir: str, partner_dir: str) -> str:
    """
    Create a temp directory with trainA/trainB symlinks as required by
    CUT's dataloader.
    """
    tmp = tempfile.mkdtemp()
    os.symlink(os.path.abspath(domain_dir),  os.path.join(tmp, "trainA"))
    os.symlink(os.path.abspath(partner_dir), os.path.join(tmp, "trainB"))
    return tmp


# ---------------------------------------------------------------------------
# Fine-tuning wrapper for Q2.1  (full-frame game ↔ movie)
# ---------------------------------------------------------------------------

def finetune_cut_fullframe(
    cut_dir: str,
    pretrained_exp: str,
    finetune_exp: str,
    frame_dataroot: str,
    device: str,
    n_epochs: int = 20,
    n_epochs_decay: int = 10,
) -> None:
    """
    Fine-tune the pretrained CUT checkpoint on full game/movie frames.

    Copies pretrained_exp → finetune_exp before training so the original
    weights are never modified and 2.2 can branch from the same base
    independently.

    frame_dataroot should be the data_dir passed to build_frame_dataset,
    i.e. the directory containing trainA/ and trainB/.
    """
    finetune_cut(
        cut_dir        = cut_dir,
        pretrained_exp = pretrained_exp,
        finetune_exp   = finetune_exp,
        dataroot       = frame_dataroot,
        device         = device,
        n_epochs       = n_epochs,
        n_epochs_decay = n_epochs_decay,
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference(
    cut_dir: str,
    exp_name: str,
    input_dataroot: str,
    results_dir: str,
    direction: str,
    device: str,
) -> list[str]:
    """
    Thin wrapper around src.utils.run_cut_inference kept for API compatibility
    with nb_main.py import.
    """
    return run_cut_inference(
        cut_dir     = cut_dir,
        exp_name    = exp_name,
        dataroot    = input_dataroot,
        results_dir = results_dir,
        direction   = direction,
        device      = device,
    )


# ---------------------------------------------------------------------------
# Full-frame test video translation — Q2.1 baseline
# ---------------------------------------------------------------------------

def translate_test_video(
    cut_dir: str,
    exp_name: str,
    test_path: str,
    save_dir: str,
    device: str,
    output_name: str = "baseline_model.mp4",
) -> str:
    """
    Extract every frame of test_path, translate game→movie with CUT, and
    write the result to save_dir/<output_name>.

    Each call uses save_dir-scoped subdirectories throughout, so parallel
    calls with different save_dir values never share cached state.

    Returns the output video path.
    """
    cap = cv2.VideoCapture(test_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    frames_dir = os.path.join(save_dir, "test_frames")
    frame_paths = extract_video_frames(test_path, frames_dir, size=(256, 256))

    # Stage frames into CUT dataroot layout.
    infer_root = os.path.join(save_dir, "test_infer_dataroot")
    trainA_tmp = os.path.join(infer_root, "trainA")
    trainB_tmp = os.path.join(infer_root, "trainB")
    os.makedirs(trainA_tmp, exist_ok=True)
    os.makedirs(trainB_tmp, exist_ok=True)

    for p in frame_paths:
        dst = os.path.join(trainA_tmp, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy(p, dst)
    if not glob.glob(os.path.join(trainB_tmp, "*.jpg")):
        shutil.copy(frame_paths[0], os.path.join(trainB_tmp, "dummy.jpg"))

    print(f"[CUT] Translating {len(frame_paths)} test frames with {exp_name}…")
    fakes = run_cut_inference(
        cut_dir     = cut_dir,
        exp_name    = exp_name,
        dataroot    = infer_root,
        results_dir = os.path.join(save_dir, "results", "test_g2m"),
        direction   = "AtoB",
        device      = device,
    )
    print(f"[CUT] {len(fakes)} translated frames")

    return write_video(fakes, os.path.join(save_dir, output_name), fps)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    real_dir: str,
    fake_dir: str,
    input_paths: list[str],
    fake_paths: list[str],
    device: str,
    num_workers: int = 2,
) -> dict:
    """
    Compute FID, KID (via clean-fid) and LPIPS for one translation direction.

    real_dir    : real target-domain frames (FID/KID reference).
    fake_dir    : translated frames         (FID/KID query).
    input_paths : source-domain images      (LPIPS reference).
    fake_paths  : translated images         (LPIPS query).
    """
    real_images = glob.glob(os.path.join(real_dir, "*"))
    fake_images = glob.glob(os.path.join(fake_dir, "*"))
    if not real_images:
        raise RuntimeError(f"compute_metrics: real_dir is empty: {real_dir}")
    if not fake_images:
        raise RuntimeError(f"compute_metrics: fake_dir is empty: {fake_dir}")

    print("[CUT] Computing FID…")
    fid_score = fid.compute_fid(
        real_dir, fake_dir, device=device, num_workers=num_workers, verbose=False
    )
    print("[CUT] Computing KID…")
    kid_score = fid.compute_kid(
        real_dir, fake_dir, device=device, num_workers=num_workers, verbose=False
    )

    print("[CUT] Computing LPIPS…")
    loss_fn = lpips_lib.LPIPS(net="vgg").to(device)
    scores  = []
    for rp, fp in tqdm(
        list(zip(sorted(input_paths), sorted(fake_paths)))[:200],
        desc="  LPIPS", leave=False,
    ):
        ri = lpips_lib.im2tensor(lpips_lib.load_image(rp)).to(device)
        fi = lpips_lib.im2tensor(lpips_lib.load_image(fp)).to(device)
        if ri.shape != fi.shape:
            fi = torch.nn.functional.interpolate(
                fi, size=ri.shape[2:], mode="bilinear", align_corners=False
            )
        scores.append(loss_fn(ri, fi).item())

    return {
        "FID":               fid_score,
        "KID":               kid_score,
        "LPIPS (vs. input)": float(np.mean(scores)),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_comparison_grid(
    input_paths: list[str],
    fake_paths: list[str],
    title: str,
    out_path: str,
    n: int = 10,
) -> None:
    """Save a side-by-side grid of n [input | translated] pairs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reals = sorted(input_paths)[:n]
    fakes = sorted(fake_paths)[:n]

    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    axes[0, 0].set_title("Input",      fontsize=10)
    axes[0, 1].set_title("Translated", fontsize=10)
    for i, (rp, fp) in enumerate(zip(reals, fakes)):
        axes[i, 0].imshow(cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB))
        axes[i, 1].imshow(cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB))
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle(title, fontsize=12, y=1.001)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[CUT] Saved → {out_path}")


def save_umap(
    groups: list[list[str]],
    labels: list[str],
    colours: list[str],
    title: str,
    out_path: str,
    n_each: int = 150,
    device: str = "cpu",
) -> None:
    """Compute VGG16 features for each group of image paths and plot a UMAP."""
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feat_groups = [_vgg_features(g[:n_each], device) for g in groups]
    all_feats   = np.concatenate(feat_groups, axis=0)
    all_labels  = np.concatenate([
        np.full(len(fg), label) for fg, label in zip(feat_groups, labels)
    ])

    print("[CUT] Fitting UMAP…")
    reducer   = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(all_feats)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, colour in zip(labels, colours):
        mask = all_labels == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   label=label, alpha=0.5, s=10, c=colour)
    ax.legend(markerscale=3)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[CUT] Saved → {out_path}")


def evaluate_translation(
    cut_dir: str,
    exp_name: str,
    testA: str,
    testB: str,
    out_dir: str,
    device: str,
    tag: str,
) -> dict:
    """
    Run inference, metrics, and visualisations for both translation directions.

    Combines run_inference + compute_metrics + save_comparison_grid + save_umap
    so the same evaluation pattern isn't duplicated for each Q2 model.

    Args:
        cut_dir  : root of the CUT repo clone.
        exp_name : checkpoint to evaluate.
        testA    : directory of game test frames (domain A).
        testB    : directory of movie test frames (domain B).
        out_dir  : output root (e.g. output/q2_1/ or output/q2_2/).
        device   : "cuda" | "cpu".
        tag      : label used in print output and figure titles (e.g. "2.1").

    Returns:
        metrics dict {"game→movie": {...}, "movie→game": {...}}
    """
    results_dir = os.path.join(out_dir, "results")
    game_imgs  = glob.glob(os.path.join(testA, "*.jpg"))
    movie_imgs = glob.glob(os.path.join(testB, "*.jpg"))

    g2m_fakes = run_inference(cut_dir, exp_name, make_inference_dataroot(testA, testB), os.path.join(results_dir, "g2m"), "AtoB", device)
    m2g_fakes = run_inference(cut_dir, exp_name, make_inference_dataroot(testB, testA), os.path.join(results_dir, "m2g"), "BtoA", device)

    metrics = {
        "game→movie": compute_metrics(testB, os.path.join(results_dir, "g2m", "fake"), game_imgs,  g2m_fakes, device),
        "movie→game": compute_metrics(testA, os.path.join(results_dir, "m2g", "fake"), movie_imgs, m2g_fakes, device),
    }
    for direction, vals in metrics.items():
        print(f"[{tag}] {direction}:  " + "  ".join(f"{k}: {v:.4f}" for k, v in vals.items()))

    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    with open(os.path.join(viz_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    save_comparison_grid(game_imgs,  g2m_fakes, f"game → movie ({tag})", os.path.join(viz_dir, "comparison_g2m.png"))
    save_comparison_grid(movie_imgs, m2g_fakes, f"movie → game ({tag})", os.path.join(viz_dir, "comparison_m2g.png"))
    save_umap([game_imgs, movie_imgs, g2m_fakes], ["game (real)", "movie (real)", "game→movie (fake)"], ["steelblue", "tomato", "mediumpurple"], f"VGG feature UMAP: game→movie ({tag})", os.path.join(viz_dir, "umap_g2m.png"), device=device)
    save_umap([game_imgs, movie_imgs, m2g_fakes], ["game (real)", "movie (real)", "movie→game (fake)"], ["steelblue", "tomato", "seagreen"],    f"VGG feature UMAP: movie→game ({tag})", os.path.join(viz_dir, "umap_m2g.png"), device=device)

    return metrics


def _vgg_features(
    image_paths: list[str],
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    import torchvision.models as tvm
    import torchvision.transforms as T

    vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT).features.to(device).eval()
    tfm = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    feats = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch = [
                tfm(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
                for p in image_paths[i: i + batch_size]
            ]
            t   = torch.stack(batch).to(device)
            out = vgg(t)
            out = torch.nn.functional.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)