"""
scripts/annotate.py

Flask-based interactive annotation tool for manually labelling human patches.

Speed optimisations
-------------------
1. CV2-only render_diagnostic (no matplotlib) — ~10-20x faster per frame.
2. Background prefetch: a thread pool renders the next PREFETCH_AHEAD images
   while you are looking at the current one.
3. Cache lives in the classification dir (.diag_cache/) so it persists across
   annotation sessions on the same cls-dir.
4. Progressive loading: raw patch shown immediately while diagnostic renders
   in the background, polled every 200ms and faded in when ready.

Usage
-----
    python3 scripts/annotate.py --cls-dir output/classifications/20260314-195748-5
    # open http://localhost:5000

    Optional:
      --class-target N   Per-class-per-domain annotation target shown in the
                         progress tracker (default: 200; set 0 for counts only).

Keys (in browser)
-----------------
    y / Y          Accept current prediction
    n / N          Open reclassify panel, then 1-6 to pick class
    b / B          Go back one image
    q / Q          Save and exit
    Escape         Close reclassify panel

Output
------
    output/manual_annotated/<run>/annotations.json
    output/manual_annotated/<run>/<class>/   (written on exit or via /finish)
"""

import argparse
import json
import os
import queue
import shutil
import sys
import threading
from collections import Counter

import cv2

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.classification import CLASSES, DEFAULT_CONFIG, render_diagnostic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_LABELS       = CLASSES + ["bad_extraction"]
PREFETCH_AHEAD   = 20
PREFETCH_WORKERS = 2


# ---------------------------------------------------------------------------
# Domain helper
# ---------------------------------------------------------------------------

def _domain_from_fname(fname):
    fl = fname.lower()
    if "mafia" in fl or "game" in fl:
        return "game"
    if any(k in fl for k in ("godfather", "irishman", "sopranos", "movie")):
        return "movie"
    return "unknown"


# ---------------------------------------------------------------------------
# Diagnostic cache helpers
# ---------------------------------------------------------------------------

def _render_and_cache(pose_model, src_path, label, diag_cache_dir, cfg=DEFAULT_CONFIG):
    """Render and cache a diagnostic image. Returns the cache path or None."""
    img_bgr = cv2.imread(src_path)
    if img_bgr is None:
        return None
    try:
        result   = pose_model(img_bgr, verbose=False)[0]
        diag     = render_diagnostic(result, img_bgr, cfg, predicted_class=label)
        out_path = os.path.join(diag_cache_dir, os.path.basename(src_path) + ".jpg")
        os.makedirs(diag_cache_dir, exist_ok=True)
        cv2.imwrite(out_path, diag, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return out_path
    except Exception as e:
        print(f"  render error for {src_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Prefetch engine
# ---------------------------------------------------------------------------

class Prefetcher:
    def __init__(self, pose_model, diag_cache_dir, n_workers=2):
        self.pose_model     = pose_model
        self.diag_cache_dir = diag_cache_dir
        self._q             = queue.Queue()
        self._in_flight     = set()
        self._lock          = threading.Lock()
        self._threads       = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(n_workers)
        ]
        for t in self._threads:
            t.start()

    def stop(self):
        for _ in self._threads:
            self._q.put(None)

    def schedule(self, items):
        for fname, src_path, label in items:
            cp = os.path.join(self.diag_cache_dir, fname + ".jpg")
            with self._lock:
                if os.path.exists(cp) or fname in self._in_flight:
                    continue
                self._in_flight.add(fname)
            self._q.put((fname, src_path, label))

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            fname, src_path, label = item
            _render_and_cache(self.pose_model, src_path, label, self.diag_cache_dir)
            with self._lock:
                self._in_flight.discard(fname)
            self._q.task_done()


# ---------------------------------------------------------------------------
# Annotation state
# ---------------------------------------------------------------------------

class AnnotationState:
    def __init__(self, patches, annotations, json_path, out_dir,
                 diag_cache_dir, class_target=200):
        self.patches        = patches          # list of (abs_path, rule_label)
        self.annotations    = annotations      # {fname: {label, source}}
        self.json_path      = json_path
        self.out_dir        = out_dir
        self.diag_cache_dir = diag_cache_dir
        self.class_target   = class_target

        self.fname_to_src = {os.path.basename(p): (p, lbl) for p, lbl in patches}
        self.all_fnames   = [os.path.basename(p) for p, _ in patches]

        # All 5 classes active by default
        self.active_classes = set(CLASSES)

        self.queue = []
        self.idx   = 0
        self._rebuild_queue()

        self.pose_model  = None
        self.prefetcher  = None

    # ── Queue management ────────────────────────────────────────────────────

    def _rebuild_queue(self):
        """Rebuild from unannotated patches in active classes. Resets idx."""
        self.queue = [
            f for f in self.all_fnames
            if f not in self.annotations
            and self.fname_to_src[f][1] in self.active_classes
        ]
        self.idx = 0

    def set_filter(self, active_classes):
        self.active_classes = set(active_classes) & set(CLASSES)
        self._rebuild_queue()
        self._schedule_prefetch()

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def n_total(self):
        return sum(
            1 for f in self.all_fnames
            if self.fname_to_src[f][1] in self.active_classes
        )

    @property
    def n_done(self):
        return sum(
            1 for f in self.annotations
            if self.fname_to_src.get(f, (None, None))[1] in self.active_classes
        )

    @property
    def finished(self):
        return self.idx >= len(self.queue)

    def current_fname(self):
        return None if self.finished else self.queue[self.idx]

    # ── Actions ─────────────────────────────────────────────────────────────

    def accept(self, fname, label, source):
        self.annotations[fname] = {"label": label, "source": source}
        self._save()
        self.idx += 1
        self._schedule_prefetch()

    def go_back(self):
        if self.idx > 0:
            self.idx -= 1
            prev = self.queue[self.idx]
            if prev in self.annotations:
                del self.annotations[prev]
                self._save()
            self._schedule_prefetch()
            return prev
        return None

    # ── Stats ────────────────────────────────────────────────────────────────

    def class_counts(self):
        """
        Per-class stats for the tracker.

        Returns dict keyed by label:
          annotated_game   : int
          annotated_movie  : int
          annotated_total  : int
          remaining        : unannotated patches still in queue for this
                             rule-based label (0 if class is inactive)
          target           : int | None
          active           : bool
        """
        ann_game  = Counter()
        ann_movie = Counter()
        for fname, entry in self.annotations.items():
            lbl    = entry["label"]
            domain = _domain_from_fname(fname)
            if domain == "game":
                ann_game[lbl] += 1
            else:
                ann_movie[lbl] += 1

        # Remaining in active queue from this point forward
        remaining_active = Counter(
            self.fname_to_src[f][1]
            for f in self.queue[self.idx:]
        )
        # Remaining for inactive classes (outside current queue)
        remaining_inactive = Counter(
            self.fname_to_src[f][1]
            for f in self.all_fnames
            if f not in self.annotations
            and self.fname_to_src[f][1] not in self.active_classes
        )

        target = self.class_target if self.class_target > 0 else None
        result = {}
        for lbl in ALL_LABELS:
            is_active = lbl in self.active_classes
            rem = (remaining_active if is_active else remaining_inactive).get(lbl, 0)
            result[lbl] = {
                "annotated_game":  ann_game.get(lbl, 0),
                "annotated_movie": ann_movie.get(lbl, 0),
                "annotated_total": ann_game.get(lbl, 0) + ann_movie.get(lbl, 0),
                "remaining":       rem,
                "target":          target if lbl != "bad_extraction" else None,
                "active":          is_active,
            }
        return result

    # ── Persistence ─────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        with open(self.json_path, "w") as f:
            json.dump(self.annotations, f, indent=2)

    def write_output_dirs(self):
        for label in ALL_LABELS:
            os.makedirs(os.path.join(self.out_dir, label), exist_ok=True)
        copied = 0
        for src_path, _ in self.patches:
            fname = os.path.basename(src_path)
            entry = self.annotations.get(fname)
            if entry is None:
                continue
            dst = os.path.join(self.out_dir, entry["label"], fname)
            if not os.path.exists(dst):
                shutil.copy2(src_path, dst)
                copied += 1
        return copied

    # ── Diagnostic cache ─────────────────────────────────────────────────────

    def _cache_path(self, fname):
        return os.path.join(self.diag_cache_dir, fname + ".jpg")

    def diag_ready(self, fname):
        return os.path.exists(self._cache_path(fname))

    def get_diag_path(self, fname, label):
        cp = self._cache_path(fname)
        if os.path.exists(cp):
            return cp
        src_path, _ = self.fname_to_src[fname]
        return _render_and_cache(self.pose_model, src_path, label, self.diag_cache_dir)

    def invalidate_diag(self, fname):
        cp = self._cache_path(fname)
        if os.path.exists(cp):
            os.remove(cp)

    def _schedule_prefetch(self):
        if self.prefetcher is None:
            return
        items = []
        for i in range(self.idx, min(self.idx + PREFETCH_AHEAD, len(self.queue))):
            fname              = self.queue[i]
            src_path, rule_lbl = self.fname_to_src[fname]
            label              = self.annotations.get(fname, {}).get("label", rule_lbl)
            items.append((fname, src_path, label))
        self.prefetcher.schedule(items)

    def load_model(self, model_path):
        from ultralytics import YOLO
        self.pose_model = YOLO(model_path)
        self.prefetcher = Prefetcher(self.pose_model, self.diag_cache_dir,
                                     n_workers=PREFETCH_WORKERS)
        self._schedule_prefetch()

    def check_and_skip_no_pose(self, fname):
        """
        Run inference on a single patch. If no keypoints are detected,
        auto-accept it as 'bad_extraction' and advance the queue.
        Returns True if the patch was skipped, False otherwise.
        """
        src_path, _ = self.fname_to_src[fname]
        img_bgr = cv2.imread(src_path)
        if img_bgr is None:
            return False
        res = self.pose_model(img_bgr, verbose=False)[0]
        no_det = res.keypoints is None or res.keypoints.data.shape[0] == 0
        if no_det:
            self.annotations[fname] = {"label": "bad_extraction", "source": "auto_no_pose"}
            self._save()
            self.idx += 1
            self._schedule_prefetch()
            print(f"  [skip] {fname}: no pose detected -> bad_extraction")
            return True
        return False


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

STATE = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_patches(cls_dir):
    patches = []
    for cls in CLASSES:
        subdir = os.path.join(cls_dir, cls)
        if not os.path.isdir(subdir):
            continue
        for fname in sorted(os.listdir(subdir)):
            if fname.lower().endswith((".jpg", ".png")):
                patches.append((os.path.join(subdir, fname), cls))
    patches.sort(key=lambda x: os.path.basename(x[0]))
    return patches


def load_annotations(json_path):
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

from flask import Flask, Response, jsonify, render_template_string, request, send_file

app = Flask(__name__)
app.secret_key = "acv_annotator"

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ACV Annotator</title>
<style>
  @import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;700;800&display=swap");
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0d0d0f; --surface: #141417; --border: #252529;
    --accent: #e8ff47; --accent2: #47d4ff; --danger: #ff4757;
    --text: #e8e8ec; --muted: #6b6b78;
    --mono: "JetBrains Mono", monospace; --sans: "Syne", sans-serif;
    --ctrl-h: 56px;
  }

  html, body {
    background: var(--bg); color: var(--text); font-family: var(--mono);
    height: 100%; overflow: hidden;
  }

  .shell {
    display: grid;
    grid-template-rows: auto 1fr var(--ctrl-h);
    height: 100vh;
    overflow: hidden;
  }

  /* ── Header ── */
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 20px; border-bottom: 1px solid var(--border);
    background: var(--surface); z-index: 10; flex-shrink: 0;
  }
  .logo { font-family: var(--sans); font-weight: 800; font-size: 0.95rem; color: var(--accent); }
  .progress-wrap { display: flex; align-items: center; gap: 12px; }
  .progress-text { font-size: 0.70rem; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; }
  .progress-bar-outer { width: 160px; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .progress-bar-inner { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.3s ease; }
  .fname { font-size: 0.66rem; color: var(--muted); max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* ── Scrollable main ── */
  main {
    overflow-y: auto;
    display: flex; flex-direction: column; align-items: center;
    padding: 12px 20px 8px; gap: 10px;
  }

  /* ── Progress tracker ── */
  .tracker {
    width: 100%; max-width: 1100px;
    border: 1px solid var(--border); border-radius: 4px;
    background: var(--surface); padding: 10px 14px;
    display: flex; flex-direction: column; gap: 6px;
    flex-shrink: 0;
  }
  .tracker-header {
    display: flex; align-items: baseline; justify-content: space-between;
    margin-bottom: 2px;
  }
  .tracker-title {
    font-size: 0.60rem; text-transform: uppercase;
    letter-spacing: 0.10em; color: var(--muted);
  }
  .tracker-hint {
    font-size: 0.58rem; color: var(--muted); font-style: italic;
  }
  .tracker-target { font-size: 0.60rem; color: var(--muted); }

  /* cls-row: [toggle btn] [bars block] [done badge] */
  .cls-row {
    display: grid;
    grid-template-columns: 118px 1fr 44px;
    align-items: center; gap: 8px;
  }

  .cls-toggle {
    font-family: var(--mono); font-size: 0.58rem; font-weight: 600;
    letter-spacing: 0.03em; padding: 3px 7px;
    border: 1px solid var(--border); border-radius: 3px;
    background: var(--surface); color: var(--text);
    cursor: pointer; white-space: nowrap;
    transition: background 0.15s, border-color 0.15s, opacity 0.2s;
    display: flex; align-items: center; justify-content: space-between; gap: 4px;
    width: 100%;
  }
  .cls-toggle:hover { background: #1e1e24; border-color: #3a3a42; }
  .cls-toggle.inactive {
    opacity: 0.35; background: #0a0a0c;
    border-color: #1a1a1e; color: var(--muted);
  }
  .cls-toggle .rem-pill {
    font-size: 0.52rem; padding: 1px 4px; border-radius: 2px;
    background: var(--border); color: var(--muted); white-space: nowrap;
    flex-shrink: 0;
  }

  /* bars block: stacked game + movie rows */
  .bars-block { display: flex; flex-direction: column; gap: 2px; }
  .domain-row {
    display: grid;
    grid-template-columns: 32px 1fr 56px;
    align-items: center; gap: 5px;
  }
  .domain-label { font-size: 0.54rem; color: var(--muted); text-align: right; }
  .bar-outer { height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .bar-inner { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
  .bar-inner.game  { background: #47d4ff; }
  .bar-inner.movie { background: #b47dff; }
  .bar-inner.done  { background: #4cda74; }
  .bar-count { font-size: 0.56rem; color: var(--text); text-align: right; white-space: nowrap; }

  .done-badge {
    font-size: 0.50rem; font-weight: 700; letter-spacing: 0.08em;
    padding: 1px 5px; border-radius: 3px;
    background: #1a3a10; color: #4cda74;
    visibility: hidden; text-align: center;
  }
  .done-badge.visible { visibility: visible; }

  /* ── Prediction badge ── */
  .prediction-badge {
    display: flex; flex-direction: column; align-items: center; gap: 3px;
    padding: 8px 18px; border: 1px solid var(--border); border-radius: 4px;
    background: var(--surface); text-align: center; min-width: 240px;
  }
  .badge-row { display: flex; align-items: baseline; gap: 8px; }
  .badge-label { color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.56rem; }
  .badge-rule  { color: var(--muted); font-size: 0.70rem; }
  .badge-current { color: var(--accent2); font-weight: 700; font-size: 0.95rem; }
  .badge-divider { width: 100%; height: 1px; background: var(--border); margin: 1px 0; }

  /* ── Images ── */
  .img-stack { width: 100%; max-width: 1400px; display: flex; flex-direction: column; gap: 6px; }
  .img-raw {
    width: 100%; max-height: 130px; object-fit: contain;
    border: 1px solid var(--border); border-radius: 4px; background: #000;
  }
  .img-diag-wrap {
    width: 100%; border: 1px solid var(--border); border-radius: 4px;
    overflow: hidden; background: #000; position: relative; min-height: 48px;
  }
  .img-diag { width: 100%; height: auto; display: block; opacity: 0; transition: opacity 0.3s; }
  .diag-overlay {
    position: absolute; inset: 0; display: flex; align-items: center;
    justify-content: center; color: var(--muted); font-size: 0.76rem;
    letter-spacing: 0.1em; pointer-events: none;
  }
  .diag-overlay.hidden { opacity: 0; }

  /* ── Reclassify panel — floats above sticky bar ── */
  .reclassify-panel {
    display: none; flex-direction: column; gap: 8px;
    position: fixed;
    bottom: calc(var(--ctrl-h) + 8px); left: 50%; transform: translateX(-50%);
    width: min(860px, calc(100vw - 40px));
    padding: 12px 14px; border: 1px solid var(--border);
    border-radius: 4px; background: var(--surface);
    z-index: 25; box-shadow: 0 -4px 24px rgba(0,0,0,0.7);
  }
  .reclassify-panel.open { display: flex; }
  .reclassify-title { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); }
  .cls-btn-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
  .cls-btn { justify-content: flex-start; padding: 7px 12px; font-size: 0.70rem; }
  .cls-btn.bad { color: var(--danger); border-color: #3a1a1a; }
  .cls-btn.bad:hover { background: #1e0f0f; }

  /* ── Sticky controls bar ── */
  .controls-bar {
    height: var(--ctrl-h);
    display: flex; align-items: center; justify-content: center; gap: 10px;
    background: var(--surface); border-top: 1px solid var(--border);
    padding: 0 20px; z-index: 20; flex-shrink: 0;
  }

  button {
    font-family: var(--mono); font-size: 0.76rem; font-weight: 600; letter-spacing: 0.06em;
    padding: 9px 20px; border: 1px solid var(--border); border-radius: 4px;
    background: var(--surface); color: var(--text); cursor: pointer;
    transition: background 0.15s, border-color 0.15s, transform 0.1s;
    display: flex; align-items: center; gap: 7px; white-space: nowrap;
  }
  button:hover  { background: #1e1e24; border-color: #3a3a42; }
  button:active { transform: scale(0.97); }
  .btn-accept { background: #1a2400; border-color: #3a5200; color: var(--accent); }
  .btn-accept:hover { background: #223000; border-color: var(--accent); }
  .btn-back   { color: var(--muted); }
  .btn-finish { color: var(--danger); border-color: #3a1a1a; }
  .btn-finish:hover { background: #1e0f0f; border-color: var(--danger); }
  .kbd {
    display: inline-block; background: var(--border); border-radius: 3px;
    padding: 1px 5px; font-size: 0.64rem; color: var(--muted);
  }

  .done-screen {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 14px; text-align: center; flex: 1; padding: 40px 0;
  }
  .done-title { font-family: var(--sans); font-size: 1.8rem; font-weight: 800; color: var(--accent); }
  .done-sub   { font-size: 0.78rem; color: var(--muted); line-height: 1.7; }

  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner {
    width: 16px; height: 16px; border: 2px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  .toast {
    position: fixed; bottom: calc(var(--ctrl-h) + 10px); right: 20px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 4px; padding: 8px 16px; font-size: 0.73rem;
    opacity: 0; transform: translateY(6px);
    transition: opacity 0.2s, transform 0.2s;
    pointer-events: none; z-index: 100;
  }
  .toast.show { opacity: 1; transform: translateY(0); }
  .toast.ok  { border-color: #3a5200; color: var(--accent); }
  .toast.err { border-color: #3a1a1a; color: var(--danger); }
</style>
</head>
<body>
<div class="shell">

  <header>
    <span class="logo">ACV ANNOTATOR</span>
    <div class="progress-wrap">
      <span class="progress-text" id="hdr-progress">{{ n_done }} / {{ n_total }}</span>
      <div class="progress-bar-outer">
        <div class="progress-bar-inner" id="hdr-bar" style="width:{{ pct }}%"></div>
      </div>
    </div>
    <span class="fname">{{ fname or "" }}</span>
  </header>

  <main>
  {% if finished %}
    <div class="done-screen">
      <div class="done-title">All done.</div>
      <div class="done-sub">{{ n_done }} patches annotated.<br>Click below to write output directories.</div>
    </div>
  {% else %}

    <!-- Tracker always first in scroll area -->
    <div class="tracker" id="tracker">
      <div class="tracker-header">
        <span class="tracker-title">
          Annotations per class
          <span class="tracker-target" id="tracker-target"></span>
        </span>
        <span class="tracker-hint">click class name to skip / include</span>
      </div>
      <!-- rows injected by JS -->
    </div>

    <div class="prediction-badge">
      <div class="badge-row"><span class="badge-label">current label</span></div>
      <span class="badge-current" id="current-label">{{ current_label }}</span>
      <div class="badge-divider"></div>
      <div class="badge-row">
        <span class="badge-label">rule-based</span>
        <span class="badge-rule">{{ rule_label }}</span>
      </div>
    </div>

    <div class="img-stack">
      <img class="img-raw" src="/raw/{{ fname }}" alt="raw patch">
      <div class="img-diag-wrap">
        <div class="diag-overlay" id="diag-overlay">
          <div class="spinner"></div>&nbsp;&nbsp;rendering diagnostic...
        </div>
        <img class="img-diag" id="diag-img" src="" alt="diagnostic"
             onload="diagLoaded()" onerror="diagError()">
      </div>
    </div>

  {% endif %}
  </main>

  <!-- Reclassify panel — fixed above sticky bar -->
  <div class="reclassify-panel" id="reclassify-panel">
    <div class="reclassify-title">Reclassify as</div>
    <div class="cls-btn-grid">
      <button class="cls-btn" onclick="reclassify('full_body_front')"><span class="kbd">1</span> full_body_front</button>
      <button class="cls-btn" onclick="reclassify('full_body_back')"><span class="kbd">2</span> full_body_back</button>
      <button class="cls-btn" onclick="reclassify('head_shoulder_front')"><span class="kbd">3</span> head_shoulder_front</button>
      <button class="cls-btn" onclick="reclassify('head_shoulder_back')"><span class="kbd">4</span> head_shoulder_back</button>
      <button class="cls-btn" onclick="reclassify('others')"><span class="kbd">5</span> others</button>
      <button class="cls-btn bad" onclick="reclassify('bad_extraction')"><span class="kbd">6</span> bad_extraction</button>
    </div>
  </div>

  <!-- Sticky controls bar -->
  <div class="controls-bar">
  {% if finished %}
    <button class="btn-finish" onclick="finish()">Write outputs &amp; exit</button>
  {% else %}
    <button class="btn-accept" onclick="accept()"><span class="kbd">Y</span> Accept</button>
    <button onclick="toggleReclassify()"><span class="kbd">N</span> Reclassify</button>
    <button class="btn-back"   onclick="goBack()"><span class="kbd">B</span> Back</button>
    <button class="btn-finish" onclick="finish()"><span class="kbd">Q</span> Save &amp; exit</button>
  {% endif %}
  </div>

</div>
<div class="toast" id="toast"></div>

<script>
const FNAME        = {{ fname | tojson }};
const RULE_LABEL   = {{ rule_label | tojson }};
let   currentLabel = {{ current_label | tojson }};
let   busy         = false;
let   pollTimer    = null;

let activeClasses = new Set({{ active_classes | tojson }});

// ── Keyboard shortcuts ──────────────────────────────────────────────────────
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const open = document.getElementById("reclassify-panel")?.classList.contains("open");
  if (open) {
    const map = {
      "1": "full_body_front", "2": "full_body_back",
      "3": "head_shoulder_front", "4": "head_shoulder_back",
      "5": "others", "6": "bad_extraction",
    };
    if (map[e.key]) { reclassify(map[e.key]); return; }
    if (e.key === "Escape" || e.key.toLowerCase() === "n") toggleReclassify();
  } else {
    if (e.key.toLowerCase() === "y") accept();
    if (e.key.toLowerCase() === "n") toggleReclassify();
    if (e.key.toLowerCase() === "b") goBack();
    if (e.key.toLowerCase() === "q") finish();
  }
});

// ── Diagnostic polling ──────────────────────────────────────────────────────
function startDiagPoll(fname, label) {
  const img     = document.getElementById("diag-img");
  const overlay = document.getElementById("diag-overlay");
  if (!img) return;
  img.style.opacity = "0";
  if (overlay) overlay.classList.remove("hidden");
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  function tryLoad() {
    fetch("/diag_ready/" + encodeURIComponent(fname))
      .then(r => r.json())
      .then(d => {
        if (d.ready) {
          img.src = "/diag/" + encodeURIComponent(fname)
            + "?label=" + encodeURIComponent(label) + "&t=" + Date.now();
        } else {
          pollTimer = setTimeout(tryLoad, 200);
        }
      })
      .catch(() => { pollTimer = setTimeout(tryLoad, 500); });
  }
  tryLoad();
}
function diagLoaded() {
  document.getElementById("diag-img").style.opacity = "1";
  document.getElementById("diag-overlay")?.classList.add("hidden");
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
}
function diagError() {
  const o = document.getElementById("diag-overlay");
  if (o) o.innerHTML = "<span style='color:var(--danger)'>Render failed.</span>";
}
if (FNAME) startDiagPoll(FNAME, currentLabel);

// ── POST helper ─────────────────────────────────────────────────────────────
function post(url, body) {
  if (busy) return Promise.resolve(null);
  busy = true;
  return fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(r => r.json()).finally(() => { busy = false; });
}

// ── Actions ─────────────────────────────────────────────────────────────────
function accept() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  post("/accept", { fname: FNAME, label: currentLabel })
    .then(d => { if (d?.redirect) window.location = d.redirect; });
}

function reclassify(label) {
  currentLabel = label;
  const el = document.getElementById("current-label");
  if (el) el.textContent = label;
  document.getElementById("reclassify-panel").classList.remove("open");
  post("/invalidate", { fname: FNAME })
    .then(() => {
      startDiagPoll(FNAME, label);
      return post("/accept", { fname: FNAME, label: label, source: "manual" });
    })
    .then(d => {
      showToast("-> " + label, "ok");
      setTimeout(() => { if (d?.redirect) window.location = d.redirect; }, 500);
    });
}

function toggleReclassify() {
  document.getElementById("reclassify-panel").classList.toggle("open");
}

function goBack() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  post("/back", {}).then(d => { if (d?.redirect) window.location = d.redirect; });
}

function finish() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  post("/finish", {}).then(d => {
    if (d?.message) showToast(d.message, "ok");
    setTimeout(() => window.location = "/", 1200);
  });
}

// ── Class filter toggle ─────────────────────────────────────────────────────
function toggleClass(cls) {
  if (activeClasses.has(cls)) {
    if (activeClasses.size <= 1) {
      showToast("At least one class must stay active", "err");
      return;
    }
    activeClasses.delete(cls);
  } else {
    activeClasses.add(cls);
  }
  // Disable all toggle buttons while request is in-flight
  document.querySelectorAll(".cls-toggle").forEach(b => b.disabled = true);
  post("/api/set_filter", { active_classes: [...activeClasses] })
    .then(d => {
      if (d?.redirect) window.location = d.redirect;
      else updateTracker();
    })
    .finally(() => {
      document.querySelectorAll(".cls-toggle").forEach(b => b.disabled = false);
    });
}

// ── Tracker rendering ───────────────────────────────────────────────────────
const CLS_LABELS = [
  "full_body_front", "full_body_back",
  "head_shoulder_front", "head_shoulder_back",
  "others", "bad_extraction",
];
const CLS_SHORT = {
  "full_body_front":     "fb_front",
  "full_body_back":      "fb_back",
  "head_shoulder_front": "hs_front",
  "head_shoulder_back":  "hs_back",
  "others":              "others",
  "bad_extraction":      "bad_extr",
};

function updateTracker() {
  fetch("/api/class_counts")
    .then(r => r.json())
    .then(data => {
      const container = document.getElementById("tracker");
      if (!container) return;

      const header = container.querySelector(".tracker-header");
      container.innerHTML = "";
      if (header) container.appendChild(header);

      // Set target label once
      let targetSet = false;

      for (const lbl of CLS_LABELS) {
        const info   = data[lbl] || {};
        const gCount = info.annotated_game  || 0;
        const mCount = info.annotated_movie || 0;
        const rem    = info.remaining       || 0;
        const target = info.target          || null;
        const active = info.active !== false;

        if (target !== null && !targetSet) {
          const tl = document.getElementById("tracker-target");
          if (tl) { tl.textContent = ` — target: ${target}/domain`; targetSet = true; }
        }

        const pctG  = target ? Math.min(100, gCount / target * 100) : 0;
        const pctM  = target ? Math.min(100, mCount / target * 100) : 0;
        const doneG = target !== null && gCount >= target;
        const doneM = target !== null && mCount >= target;

        const row = document.createElement("div");
        row.className = "cls-row";

        // Toggle button
        const btn = document.createElement("button");
        btn.className = "cls-toggle" + (active ? "" : " inactive");
        btn.title     = active ? "Click to skip this class" : "Click to include this class";
        btn.onclick   = () => toggleClass(lbl);

        const nameSpan = document.createElement("span");
        nameSpan.textContent = CLS_SHORT[lbl] || lbl;

        const pill = document.createElement("span");
        pill.className   = "rem-pill";
        pill.textContent = rem > 0 ? `${rem}` : "0";

        btn.appendChild(nameSpan);
        btn.appendChild(pill);

        // Bars block
        const barsBlock = document.createElement("div");
        barsBlock.className = "bars-block";

        for (const [domainKey, count, pct, done] of [
          ["game",  gCount, pctG, doneG],
          ["movie", mCount, pctM, doneM],
        ]) {
          const dr = document.createElement("div");
          dr.className = "domain-row";

          const dl = document.createElement("span");
          dl.className   = "domain-label";
          dl.textContent = domainKey;

          const bo = document.createElement("div");
          bo.className = "bar-outer";
          const bi = document.createElement("div");
          bi.className = "bar-inner " + domainKey + (done ? " done" : "");
          bi.style.width = pct.toFixed(1) + "%";
          bo.appendChild(bi);

          const bc = document.createElement("span");
          bc.className   = "bar-count";
          bc.textContent = target !== null ? `${count}/${target}` : `${count}`;

          dr.appendChild(dl);
          dr.appendChild(bo);
          dr.appendChild(bc);
          barsBlock.appendChild(dr);
        }

        // Done badge
        const badge = document.createElement("span");
        badge.className   = "done-badge" + (doneG && doneM ? " visible" : "");
        badge.textContent = "DONE";

        row.appendChild(btn);
        row.appendChild(barsBlock);
        row.appendChild(badge);
        container.appendChild(row);
      }

      // Sync local activeClasses from server truth
      activeClasses = new Set(
        CLS_LABELS.filter(l => (data[l]?.active !== false) && l !== "bad_extraction")
      );
    })
    .catch(() => {});
}

updateTracker();
setInterval(updateTracker, 3000);

function showToast(msg, type) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "toast " + type + " show";
  setTimeout(() => t.classList.remove("show"), 2200);
}
</script>
</body>
</html>
"""


def _ctx():
    s           = STATE
    fname       = s.current_fname()
    rule_label  = s.fname_to_src[fname][1] if fname else ""
    current_lbl = s.annotations.get(fname, {}).get("label", rule_label) if fname else ""
    pct         = round(s.n_done / s.n_total * 100, 1) if s.n_total else 0
    return dict(
        fname=fname,
        rule_label=rule_label,
        current_label=current_lbl,
        n_done=s.n_done,
        n_total=s.n_total,
        pct=pct,
        finished=s.finished,
        out_dir=s.out_dir,
        active_classes=list(s.active_classes),
    )


@app.route("/")
def index():
    # Skip any consecutive no-pose patches at the front of the queue.
    if STATE.pose_model is not None:
        while not STATE.finished:
            fname = STATE.current_fname()
            if fname and fname not in STATE.annotations:
                if STATE.check_and_skip_no_pose(fname):
                    continue   # auto-accepted — check the next one
            break
    STATE._schedule_prefetch()
    return render_template_string(HTML, **_ctx())


@app.route("/raw/<fname>")
def raw(fname):
    src_path, _ = STATE.fname_to_src.get(fname, (None, None))
    if not src_path or not os.path.exists(src_path):
        return Response(status=404)
    return send_file(src_path, mimetype="image/jpeg")


@app.route("/diag_ready/<fname>")
def diag_ready(fname):
    return jsonify(ready=STATE.diag_ready(fname))


@app.route("/diag/<fname>")
def diag(fname):
    label      = request.args.get("label", "")
    cache_path = STATE.get_diag_path(fname, label)
    if not cache_path or not os.path.exists(cache_path):
        return Response(status=404)
    return send_file(cache_path, mimetype="image/jpeg")


@app.route("/accept", methods=["POST"])
def accept():
    data   = request.get_json()
    fname  = data["fname"]
    label  = data["label"]
    source = data.get(
        "source",
        "auto" if label == STATE.fname_to_src[fname][1] else "manual",
    )
    STATE.accept(fname, label, source)
    return jsonify(redirect="/")


@app.route("/invalidate", methods=["POST"])
def invalidate():
    fname = request.get_json().get("fname")
    if fname:
        STATE.invalidate_diag(fname)
    return jsonify(ok=True)


@app.route("/back", methods=["POST"])
def back():
    STATE.go_back()
    return jsonify(redirect="/")


@app.route("/api/class_counts")
def api_class_counts():
    return jsonify(STATE.class_counts())


@app.route("/api/set_filter", methods=["POST"])
def api_set_filter():
    data           = request.get_json()
    active_classes = data.get("active_classes", list(CLASSES))
    STATE.set_filter(active_classes)
    fname = STATE.current_fname()
    return jsonify(redirect="/" if fname is not None else None)


def _shutdown_server():
    import time, signal
    time.sleep(1.2)
    os.kill(os.getpid(), signal.SIGTERM)


@app.route("/finish", methods=["POST"])
def finish():
    if STATE.prefetcher:
        STATE.prefetcher.stop()
    STATE._save()
    copied = STATE.write_output_dirs()
    msg = f"Saved {STATE.n_done} annotations. Wrote {copied} files."
    print(f"\n{msg}\nAnnotations -> {STATE.json_path}\nOutput dirs -> {STATE.out_dir}")
    threading.Thread(target=_shutdown_server, daemon=True).start()
    return jsonify(message=msg, redirect="/")


def main():
    parser = argparse.ArgumentParser(
        description="Flask annotation tool for human patch labelling."
    )
    parser.add_argument("--cls-dir",      required=True)
    parser.add_argument("--out-dir",      default=None)
    parser.add_argument("--pose-model",   default=None)
    parser.add_argument("--port",         type=int, default=5000)
    parser.add_argument("--host",         default="127.0.0.1")
    parser.add_argument("--class-target", type=int, default=200,
                        help="Per-class-per-domain annotation target "
                             "(default: 200; 0 = counts only, no target bars)")
    args = parser.parse_args()

    cls_dir  = os.path.abspath(args.cls_dir)
    run_name = os.path.basename(os.path.normpath(cls_dir))
    out_dir  = (
        os.path.abspath(args.out_dir) if args.out_dir
        else os.path.join(PROJECT_ROOT, "output", "manual_annotated", run_name)
    )
    pose_model = args.pose_model or os.path.join(PROJECT_ROOT, "models", "yolo26m-pose.pt")

    if not os.path.isdir(cls_dir):
        print(f"Error: cls-dir not found: {cls_dir}"); sys.exit(1)
    if not os.path.exists(pose_model):
        print(f"Error: pose model not found: {pose_model}"); sys.exit(1)

    patches        = collect_patches(cls_dir)
    json_path      = os.path.join(out_dir, "annotations.json")
    annotations    = load_annotations(json_path)
    diag_cache_dir = os.path.join(cls_dir, ".diag_cache")

    global STATE
    STATE = AnnotationState(
        patches, annotations, json_path, out_dir, diag_cache_dir,
        class_target=args.class_target,
    )
    STATE.load_model(pose_model)  # also starts prefetch

    print(f"\nACV Annotator")
    print(f"  cls-dir      : {cls_dir}")
    print(f"  out-dir      : {out_dir}")
    print(f"  diag-cache   : {diag_cache_dir}")
    print(f"  patches      : {STATE.n_total}  (todo: {len(STATE.queue)}  done: {STATE.n_done})")
    print(f"  class-target : {args.class_target or 'none (counts only)'}")
    print(f"  prefetch     : next {PREFETCH_AHEAD} images, {PREFETCH_WORKERS} workers")
    print(f"\n  -> http://{args.host}:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()