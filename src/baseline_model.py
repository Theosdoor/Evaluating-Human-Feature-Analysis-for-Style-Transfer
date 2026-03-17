"""
src/baseline_model.py
Question 2.1 — Image Model Deployment (CUT-based baseline)

Public API (called from nb_main.py):
    build_frame_dataset(selected_detections, train_paths, data_dir, n_per_domain)
    train_cut(cut_dir, data_dir, exp_name, direction, device, n_epochs, n_epochs_decay, batch_size)
    run_inference(cut_dir, exp_name, input_dataroot, results_dir, direction, device)
    translate_test_video(cut_dir, exp_name, test_path, save_dir, device)  -> video_path
    compute_metrics(real_dir, fake_dir, input_paths, fake_paths, device)  -> dict
    save_comparison_grid(input_paths, fake_paths, title, out_path, n)
    save_umap(groups, labels, colours, title, out_path)
    make_inference_dataroot(domain_dir, partner_dir)                       -> tmp_path
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import lpips as lpips_lib
import numpy as np
import torch
from cleanfid import fid
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def build_frame_dataset(selected_detections, train_paths, data_dir, n_per_domain=1000):
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

    print("Building CUT frame dataset…")
    _extract_frames(selected_detections, game_paths,  trainA, n_per_domain)
    _extract_frames(selected_detections, movie_paths, trainB, n_per_domain)

    _make_held_out_split(trainA, testA)
    _make_held_out_split(trainB, testB)

    print(f"  trainA (game):  {len(glob.glob(os.path.join(trainA, '*.jpg')))} frames")
    print(f"  trainB (movie): {len(glob.glob(os.path.join(trainB, '*.jpg')))} frames")
    return trainA, trainB, testA, testB


def _extract_frames(detections, video_paths_for_domain, output_dir, n_frames):
    os.makedirs(output_dir, exist_ok=True)
    existing = len(glob.glob(os.path.join(output_dir, "*.jpg")))
    if existing >= n_frames:
        print(f"  {os.path.basename(output_dir)}: {existing} frames present, skipping.")
        return

    domain_dets = [d for d in detections if d["video_path"] in video_paths_for_domain]
    step = max(1, len(domain_dets) // n_frames)
    domain_dets = domain_dets[::step][:n_frames]

    by_video = {}
    for det in domain_dets:
        by_video.setdefault(det["video_path"], []).append(det)

    saved = 0
    for vpath, dets in by_video.items():
        dets_sorted = sorted(dets, key=lambda d: d["frame_num"])
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  Warning: cannot open {vpath}")
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
            cv2.imwrite(os.path.join(output_dir, fname),
                        cv2.resize(frame, (286, 286)),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        cap.release()
    print(f"  Saved {saved} frames → {output_dir}")


def _make_held_out_split(src_dir, dst_dir, n=200):
    os.makedirs(dst_dir, exist_ok=True)
    if len(glob.glob(os.path.join(dst_dir, "*.jpg"))) >= n:
        return
    frames = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))[-n:]
    for f in frames:
        shutil.copy(f, os.path.join(dst_dir, os.path.basename(f)))


def make_inference_dataroot(domain_dir, partner_dir):
    """
    Create a temporary directory with trainA → domain_dir and trainB → partner_dir
    symlinks, as required by CUT's dataloader.
    """
    tmp = tempfile.mkdtemp()
    os.symlink(os.path.abspath(domain_dir),  os.path.join(tmp, "trainA"))
    os.symlink(os.path.abspath(partner_dir), os.path.join(tmp, "trainB"))
    return tmp


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_cut(
    cut_dir,
    data_dir,
    exp_name,
    direction,
    device,
    n_epochs=100,
    n_epochs_decay=100,
    batch_size=4,
):
    """
    Launch CUT training via subprocess.  Returns the checkpoint directory path.
    Skips if latest_net_G.pth already exists.

    direction : 'AtoB' (game→movie) or 'BtoA' (movie→game)
    """
    ckpt_dir = os.path.join(cut_dir, "checkpoints", exp_name)
    latest   = os.path.join(ckpt_dir, "latest_net_G.pth")

    if os.path.exists(latest):
        print(f"  Checkpoint exists, skipping training: {exp_name}")
        return ckpt_dir

    gpu_ids = "0" if device == "cuda" else "-1"
    cmd = [
        sys.executable, os.path.join(cut_dir, "train.py"),
        "--dataroot",       data_dir,
        "--name",           exp_name,
        "--model",          "cut",
        "--CUT_mode",       "CUT",
        "--direction",      direction,
        "--n_epochs",       str(n_epochs),
        "--n_epochs_decay", str(n_epochs_decay),
        "--batch_size",     str(batch_size),
        "--load_size",      "286",
        "--crop_size",      "256",
        "--gpu_ids",        gpu_ids,
        "--checkpoints_dir",os.path.join(cut_dir, "checkpoints"),
        "--no_html",
        "--display_id",     "0",
        "--nce_idt",        "True",
        "--save_epoch_freq","10",
    ]
    print(f"  Training CUT ({direction})…")
    subprocess.run(cmd, check=True, cwd=cut_dir)
    return ckpt_dir


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(cut_dir, exp_name, input_dataroot, results_dir, direction, device):
    """
    Run CUT test.py on input_dataroot.  Copies fake_B outputs into
    results_dir/fake/ and returns a sorted list of those paths.
    """
    fake_dir = os.path.join(results_dir, "fake")
    os.makedirs(fake_dir, exist_ok=True)

    existing = sorted(glob.glob(os.path.join(fake_dir, "*.jpg")))
    if existing:
        print(f"  Inference cache hit: {fake_dir} ({len(existing)} images)")
        return existing

    gpu_ids = "0" if device == "cuda" else "-1"
    cmd = [
        sys.executable, os.path.join(cut_dir, "test.py"),
        "--dataroot",        input_dataroot,
        "--name",            exp_name,
        "--model",           "cut",
        "--direction",       direction,
        "--results_dir",     os.path.join(cut_dir, "results"),
        "--checkpoints_dir", os.path.join(cut_dir, "checkpoints"),
        "--gpu_ids",         gpu_ids,
        "--load_size",       "256",
        "--crop_size",       "256",
        "--no_flip",
        "--num_test",        "9999",
        "--phase",           "test",
        "--eval",
    ]
    subprocess.run(cmd, check=True, cwd=cut_dir)

    raw_dir = os.path.join(cut_dir, "results", exp_name, "test_latest", "images")
    for p in glob.glob(os.path.join(raw_dir, "*fake_B*")):
        dst = os.path.join(fake_dir, os.path.basename(p).replace("_fake_B", ""))
        if not os.path.exists(dst):
            shutil.copy(p, dst)

    return sorted(glob.glob(os.path.join(fake_dir, "*.jpg")))


def translate_test_video(cut_dir, exp_name, test_path, save_dir, device):
    """
    Extract every frame of test_path, translate game→movie with CUT,
    and write the result to save_dir/baseline_model.mp4.

    Returns the output video path.
    """
    frames_dir = os.path.join(save_dir, "2_1_test_frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap   = cv2.VideoCapture(test_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_paths = []
    existing = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if len(existing) >= total > 0:
        print(f"  Test frames already extracted ({len(existing)}), skipping.")
        frame_paths = existing
        cap.release()
    else:
        for i in tqdm(range(total), desc="Extracting test frames"):
            ret, frame = cap.read()
            if not ret:
                break
            p = os.path.join(frames_dir, f"frame_{i:05d}.jpg")
            cv2.imwrite(p, cv2.resize(frame, (256, 256)),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_paths.append(p)
        cap.release()
        print(f"  {len(frame_paths)} test frames extracted")

    # Build a temporary dataroot for CUT inference
    infer_root = os.path.join(save_dir, "2_1_test_infer_dataroot")
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

    print("Translating test frames (game→movie)…")
    fakes = run_inference(
        cut_dir, exp_name, infer_root,
        os.path.join(save_dir, "2_1_results", "test_g2m"),
        "AtoB", device,
    )
    print(f"  {len(fakes)} translated frames")

    video_path = os.path.join(save_dir, "baseline_model.mp4")
    fakes_sorted = sorted(fakes)
    if fakes_sorted:
        sample = cv2.imread(fakes_sorted[0])
        h, w   = sample.shape[:2]
        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        for fp in tqdm(fakes_sorted, desc="Writing baseline_model.mp4"):
            frame = cv2.imread(fp)
            if frame is not None:
                writer.write(frame)
        writer.release()
        print(f"  Baseline video → {video_path}")
    else:
        print("  Warning: no translated frames — video not written.")

    return video_path


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(real_dir, fake_dir, input_paths, fake_paths, device):
    """
    Compute FID, KID (via torch-fidelity) and LPIPS (vs. input) for one direction.

    real_dir    : directory of real target-domain frames (for FID/KID)
    fake_dir    : directory of translated frames         (for FID/KID)
    input_paths : list of source-domain image paths      (for LPIPS)
    fake_paths  : list of translated image paths         (for LPIPS, same order)

    Returns a dict with keys FID, KID, LPIPS.
    """
    print("  Computing FID…")
    fid_score = fid.compute_fid(real_dir, fake_dir, device=device, verbose=False)
    print("  Computing KID…")
    kid_score = fid.compute_kid(real_dir, fake_dir, device=device, verbose=False)
    fid_kid = {
        "frechet_inception_distance": fid_score,
        "kernel_inception_distance_mean": kid_score,
    }

    print("  Computing LPIPS…")
    loss_fn = lpips_lib.LPIPS(net="vgg").to(device)
    scores  = []
    pairs   = list(zip(sorted(input_paths), sorted(fake_paths)))[:200]
    for rp, fp in tqdm(pairs, desc="  LPIPS", leave=False):
        ri = lpips_lib.im2tensor(lpips_lib.load_image(rp)).to(device)
        fi = lpips_lib.im2tensor(lpips_lib.load_image(fp)).to(device)
        if ri.shape != fi.shape:
            fi = torch.nn.functional.interpolate(
                fi, size=ri.shape[2:], mode="bilinear", align_corners=False
            )
        scores.append(loss_fn(ri, fi).item())

    return {
        "FID":               fid_kid["frechet_inception_distance"],
        "KID":               fid_kid["kernel_inception_distance_mean"],
        "LPIPS (vs. input)": float(np.mean(scores)),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_comparison_grid(input_paths, fake_paths, title, out_path, n=10):
    """
    Save a side-by-side grid of n [input | translated] pairs.
    """
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
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


def save_umap(groups, labels, colours, title, out_path, n_each=150, device="cpu"):
    """
    Compute VGG16 features for each group of image paths and plot a UMAP.

    groups  : list of lists of image paths
    labels  : list of string labels (same length as groups)
    colours : list of colour strings
    """
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feat_groups = [_vgg_features(g[:n_each], device) for g in groups]
    all_feats   = np.concatenate(feat_groups, axis=0)
    all_labels  = np.concatenate([
        np.full(len(fg), label) for fg, label in zip(feat_groups, labels)
    ])

    print("  Fitting UMAP…")
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
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


def _vgg_features(image_paths, device, batch_size=32):
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
            batch = []
            for p in image_paths[i : i + batch_size]:
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                batch.append(tfm(img))
            t = torch.stack(batch).to(device)
            out = vgg(t)
            out = torch.nn.functional.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


