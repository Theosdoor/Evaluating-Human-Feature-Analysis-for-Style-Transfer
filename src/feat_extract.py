"""
feat_extract.py

Human patch extraction pipeline.
- Stores video_path + frame_num instead of raw frames to avoid memory blowout
- Computes blur score inline during extraction, before discarding the frame
- Caches cropped patch in detection dict to eliminate the second video pass in save_patches
- Per-source temporal gap enforcement in diverse_sampling (O(n) not O(n²))
- Frame change detection to skip near-duplicate frames before running YOLO
- GPU batched YOLO inference: buffers `yolo_batch_size` eligible frames then
  runs a single model call, keeping the 2080 Ti feed efficiently
- cap.grab() (no decode) used for frames deep inside the yolo_interval gap
- Async patch saving via ThreadPoolExecutor to overlap disk I/O with work
"""

import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def frame_difference(prev_gray, curr_gray):
    """Mean absolute difference between consecutive grayscale frames."""
    if prev_gray is None:
        return float('inf')
    diff = cv2.absdiff(prev_gray, curr_gray)
    return diff.mean()


def compute_blur_score(patch):
    """Laplacian variance of a BGR patch. Higher = sharper."""
    if patch is None or patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _process_batch(model, frame_batch, frame_meta, img_area, blur_thresh, video_path):
    """
    Run YOLO on a batch of frames and return detection dicts.

    frame_batch : list of BGR numpy arrays
    frame_meta  : list of (frame_count,) tuples aligned with frame_batch
    """
    detections = []
    half = 'cuda' in str(model.device)  # use fp16 on GPU
    results = model(frame_batch, classes=[0], verbose=False, half=half)

    for batch_idx, (result, (frame_count,)) in enumerate(zip(results, frame_meta)):
        if result.boxes is None:
            continue
        frame_h, frame_w = result.orig_shape
        for box in result.boxes:
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            patch = frame_batch[batch_idx][y1:y2, x1:x2]
            blur = compute_blur_score(patch)
            area = (x2 - x1) * (y2 - y1)
            relative_area = area / img_area

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            dist = np.sqrt((center_x - frame_w / 2) ** 2 + (center_y - frame_h / 2) ** 2)
            max_dist = np.sqrt((frame_w / 2) ** 2 + (frame_h / 2) ** 2)
            centering = 1.0 - dist / max_dist

            detections.append({
                'video_path': video_path,
                'frame_num': frame_count,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2),
                'area': area,
                'relative_area': relative_area,
                'blur_score': blur,
                'blur_thresh': blur_thresh,
                'centering': centering,
                'patch': patch.copy(),  # cached crop — avoids second video pass
            })

    return detections


def extract_humans_from_video(
    model,
    video_path,
    yolo_interval=10,
    scene_change_threshold=8.0,
    blur_threshold_film=40.0,
    blur_threshold_game=100.0,
    yolo_batch_size=8,
):
    """
    Extract human detections from a video file.

    GPU speedup — batched YOLO calls:
      Instead of running model(frame) once per frame, eligible frames are
      buffered until `yolo_batch_size` are ready, then dispatched as one
      model(batch) call.  This keeps the 2080 Ti feed efficiently without
      holding many frames in RAM simultaneously.

    Uses two-stage frame filtering:
      1. Skip frames where inter-frame difference is below scene_change_threshold
      2. Run YOLO every `yolo_interval` frames at minimum

    Blur is scored inline per-patch so we don't need to store full frames.
    Film footage threshold is lower than game footage (inferred from filename).

    Returns a list of detection dicts (no raw frame data stored).
    """
    is_film = any(kw in video_path.lower() for kw in ['movie', 'film', 'godfather', 'irishman', 'sopranos'])
    blur_thresh = blur_threshold_film if is_film else blur_threshold_game

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_area = frame_h * frame_w

    frame_count = 0
    last_yolo_frame = -yolo_interval
    prev_gray = None
    detections = []

    # Batch buffers
    frame_buf = []   # BGR frames ready for YOLO
    meta_buf  = []   # (frame_count,) per buffered frame

    def flush_batch():
        if not frame_buf:
            return
        detections.extend(_process_batch(model, frame_buf, meta_buf, img_area, blur_thresh, video_path))
        frame_buf.clear()
        meta_buf.clear()

    progress_total = total_frames if total_frames > 0 else None
    with tqdm(total=progress_total, desc=f"Extracting {os.path.basename(video_path)}", unit="frame") as pbar:
        while cap.isOpened():
            frames_since_yolo = frame_count - last_yolo_frame

            # If we're deep inside the interval gap, skip decoding entirely.
            # We lose diff-check continuity for those frames but avoid decode cost.
            skip_ahead = max(0, yolo_interval - frames_since_yolo - 2)
            if skip_ahead > 0 and prev_gray is not None:
                for _ in range(skip_ahead):
                    if not cap.grab():
                        break
                frame_count += skip_ahead
                pbar.update(skip_ahead)

            ret, frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = frame_difference(prev_gray, curr_gray)
            prev_gray = curr_gray

            frames_since_yolo = frame_count - last_yolo_frame
            if frames_since_yolo >= yolo_interval and diff >= scene_change_threshold:
                last_yolo_frame = frame_count
                frame_buf.append(frame.copy())
                meta_buf.append((frame_count,))

                if len(frame_buf) >= yolo_batch_size:
                    flush_batch()

            frame_count += 1
            pbar.update(1)

    # Process any remaining frames in the buffer
    flush_batch()
    cap.release()
    return detections


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_detection(det):
    """
    Composite quality score in [0, 1].

    Weights:
      0.30 — YOLO confidence
      0.25 — relative size (penalise tiny or overly large crops)
      0.20 — sharpness (relative to per-source blur threshold)
      0.15 — centering
      0.10 — aspect ratio (tall crops preferred over wide)
    """
    score = 0.0

    # 1. Confidence
    score += det['confidence'] * 0.30

    # 2. Relative size — soft penalty outside [0.04, 0.45]
    ra = det['relative_area']
    if 0.04 <= ra <= 0.45:
        size_score = 1.0 - abs(ra - 0.15) / 0.30
        score += max(0.0, size_score) * 0.25

    # 3. Sharpness — normalise against threshold so film/game are comparable
    blur_norm = min(det['blur_score'] / (det['blur_thresh'] * 3), 1.0)
    score += blur_norm * 0.20

    # 4. Centering
    score += det['centering'] * 0.15

    # 5. Aspect ratio — humans should be taller than wide
    x1, y1, x2, y2 = det['bbox']
    w, h = x2 - x1, y2 - y1
    if w > 0:
        ar = h / w
        if ar >= 1.0:
            ar_score = min((ar - 1.0) / 2.5, 1.0)
            score += ar_score * 0.10

    return round(score, 4)


# ---------------------------------------------------------------------------
# Diverse sampling
# ---------------------------------------------------------------------------

def diverse_sampling(detections, target_count=1000, temporal_gap=30):
    """
    Greedy diverse sampling with per-source temporal gap enforcement.

    Works in O(n): tracks last selected frame per video source rather than
    scanning the full selected set each iteration.

    detections should already be sorted by score descending.
    """
    selected = []
    last_frame_per_source = {}

    for det in tqdm(detections, desc="Selecting diverse detections", unit="det"):
        if len(selected) >= target_count:
            break

        source = det['video_path']
        last = last_frame_per_source.get(source, -(temporal_gap + 1))

        if abs(det['frame_num'] - last) >= temporal_gap:
            selected.append(det)
            last_frame_per_source[source] = det['frame_num']

    return selected


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _write_patch(args):
    """Worker function: crop + save a single patch. Used by ThreadPoolExecutor."""
    frame, det, idx, output_dir = args
    # Use pre-cached patch if available (fast path — no video re-read needed)
    if det.get('patch') is not None:
        patch = det['patch']
    else:
        x1, y1, x2, y2 = det['bbox']
        patch = frame[y1:y2, x1:x2]
    if patch is None or patch.size == 0:
        return False

    video_path = det['video_path']
    source_tag = os.path.splitext(os.path.basename(video_path))[0]
    filename = (
        f"human_{idx:04d}_{source_tag}"
        f"_f{det['frame_num']:06d}"
        f"_conf{det['confidence']:.2f}"
        f"_score{det['score']:.2f}.jpg"
    )
    cv2.imwrite(
        os.path.join(output_dir, filename),
        patch,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    return True


def save_patches(detections, output_dir, io_workers=4):
    """
    Save patches to disk.

    Fast path: if detections carry a 'patch' key (set during extraction),
    patches are written directly without re-opening any video file.

    Fallback: re-reads source videos sequentially for any detections that
    lack a cached patch (e.g. loaded from a previous run's metadata).

    Writes are dispatched to a ThreadPoolExecutor to overlap disk I/O.
    """
    os.makedirs(output_dir, exist_ok=True)

    cached   = [(i, d) for i, d in enumerate(detections) if d.get('patch') is not None]
    uncached = [(i, d) for i, d in enumerate(detections) if d.get('patch') is None]

    saved = 0
    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        # --- Fast path: patches already in memory ---
        if cached:
            futures = [
                executor.submit(_write_patch, (None, det, idx, output_dir))
                for idx, det in tqdm(cached, desc="Writing cached patches", unit="patch")
            ]
            for f in as_completed(futures):
                if f.result():
                    saved += 1

        # --- Fallback: re-read videos for uncached detections ---
        if uncached:
            by_video = {}
            for i, det in uncached:
                by_video.setdefault(det['video_path'], []).append((i, det))

            for video_path, items in tqdm(by_video.items(), desc="Saving patches (video re-read)", unit="video"):
                items_sorted = sorted(items, key=lambda x: x[1]['frame_num'])

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Warning: could not open {video_path} for saving patches")
                    continue

                futures = []
                current_frame = 0
                for idx, det in tqdm(items_sorted, desc=f"Patches {os.path.basename(video_path)}", unit="patch", leave=False):
                    target_frame = det['frame_num']

                    if target_frame < current_frame:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        current_frame = target_frame

                    while current_frame < target_frame:
                        cap.grab()
                        current_frame += 1

                    ret, frame = cap.read()
                    if not ret:
                        continue
                    current_frame += 1

                    futures.append(executor.submit(_write_patch, (frame.copy(), det, idx, output_dir)))

                cap.release()

                for f in as_completed(futures):
                    if f.result():
                        saved += 1

    print(f"Saved {saved} patches to {output_dir}")
    return saved