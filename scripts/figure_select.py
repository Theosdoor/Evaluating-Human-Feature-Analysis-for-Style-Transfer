"""
scripts/figure_select.py

Flask-based figure selection tool for Q2.1 and Q2.2 paper figures.

Tabs
----
  2.1  — original + translated pairs; mark as success, failure, or skip
  2.2  — original + 2.1-output + 2.2-output triples; mark as selected or skip

Keys (in browser)
-----------------
  s / S        (2.1 tab) success
  f / F        (2.1 tab) failure
  y / Y        (2.2 tab) select
  Space / →    jump forward N frames (default 30; set with --skip-interval)
  b / B        go back one
  q / Q        export figures and open export summary

Output
------
  paper/figs/q21_success.png   — 10-row × 2-col grid  [orig | translated]
  paper/figs/q21_failure.png   — 10-row × 2-col grid  [orig | translated]
  paper/figs/q22_compare.png   — 10-row × 3-col grid  [orig | 2.1 | 2.2]

  Selections are auto-saved to output/figure_select/selections.json.

Usage
-----
  # Minimal — auto-discovers images from standard CUT output layout:
  python3 scripts/figure_select.py --q21-dir output/q2_1/20260325-120108 --q22-dir output/q2_2/20260325-115117

  # Override individual image dirs if layout differs:
  python3 scripts/figure_select.py \\
      --q21-orig-dir output/q2_1/.../cut_data/testA \\
      --q21-fake-dir output/q2_1/.../results/test_g2m/fake_M \\
      --q22-orig-dir output/q2_2/.../cut_data/testA \\
      --q22-21-dir  output/q2_2/.../results/test_g2m/fake_M \\
      --q22-22-dir  output/q2_2/.../enh_frames
      
      
python3 scripts/figure_select.py --q21-orig-dir output/q2_1/20260325-120108/results/test_g2m/cut_raw/cut_finetuned_fullframe/train_latest/images/real_A --q21-fake-dir output/q2_1/20260325-120108/results/test_g2m/cut_raw/cut_finetuned_fullframe/train_latest/images/fake_B


"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, jsonify, render_template_string, request, send_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_21 = 10   # desired per category (success / failure)
TARGET_22 = 10   # desired selected triples

# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def _find_images(directory: str) -> list[str]:
    if not directory or not os.path.isdir(directory):
        return []
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)


def _match_pairs(orig_paths: list[str], fake_paths: list[str]) -> list[tuple[str, str]]:
    """Match originals to fakes by basename."""
    fake_by_name = {os.path.basename(p): p for p in fake_paths}
    pairs = []
    for op in orig_paths:
        name = os.path.basename(op)
        if name in fake_by_name:
            pairs.append((op, fake_by_name[name]))
        else:
            stem = os.path.splitext(name)[0]
            for fn, fp in fake_by_name.items():
                if os.path.splitext(fn)[0] == stem:
                    pairs.append((op, fp))
                    break
    return pairs


def _match_triples(
    orig_paths: list[str],
    fake21_paths: list[str],
    fake22_paths: list[str],
) -> list[tuple[str, str, str]]:
    """Match originals, 2.1-fakes, 2.2-fakes by basename."""
    by21 = {os.path.basename(p): p for p in fake21_paths}
    by22 = {os.path.basename(p): p for p in fake22_paths}
    triples = []
    for op in orig_paths:
        name = os.path.basename(op)
        if name in by21 and name in by22:
            triples.append((op, by21[name], by22[name]))
    return triples


def _auto_discover_q21(run_dir: str):
    """Return (orig_dir, fake_dir) from a standard Q2.1 run directory."""
    orig_dir = os.path.join(run_dir, "cut_data", "testA")
    # CUT writes to results/<exp>/test_latest/images/fake_B; utils.py copies to
    # results/test_g2m/fake_M  (game→movie direction)
    for candidate in (
        os.path.join(run_dir, "results", "test_g2m", "fake_M"),
        os.path.join(run_dir, "results", "test_m2g", "fake_G"),
        os.path.join(run_dir, "fake_M"),
        os.path.join(run_dir, "fake_G"),
    ):
        if os.path.isdir(candidate):
            return orig_dir, candidate
    return orig_dir, None


def _auto_discover_q22(run_dir: str):
    """Return (orig_dir, fake21_dir, fake22_dir) from a standard Q2.2 run directory."""
    orig_dir = os.path.join(run_dir, "cut_data", "testA")
    fake21_dir, fake22_dir = None, None
    for candidate in (
        os.path.join(run_dir, "results", "test_g2m", "fake_M"),
        os.path.join(run_dir, "fake_M"),
    ):
        if os.path.isdir(candidate):
            fake21_dir = candidate
            break
    for candidate in (
        os.path.join(run_dir, "enh_frames"),
        os.path.join(run_dir, "enhanced"),
        os.path.join(run_dir, "composited"),
    ):
        if os.path.isdir(candidate):
            fake22_dir = candidate
            break
    return orig_dir, fake21_dir, fake22_dir


# ---------------------------------------------------------------------------
# Figure grid generation
# ---------------------------------------------------------------------------

def _load_resize(path: str, size: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        placeholder = np.full((*size[::-1], 3), 30, dtype=np.uint8)
        cv2.putText(placeholder, "?", (size[0]//2 - 8, size[1]//2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        return placeholder
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def make_grid(
    image_groups: list[list[str]],
    cell_w: int = 512,
    cell_h: int = 288,
    padding: int = 4,
    bg: int = 20,
) -> np.ndarray:
    """
    Build a grid image.  Each element of image_groups is one row; each element
    within the sub-list is one column image path.
    Returns a BGR numpy array suitable for cv2.imwrite.
    """
    if not image_groups:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    n_rows = len(image_groups)
    n_cols = max(len(g) for g in image_groups)
    p = padding
    canvas = np.full(
        (n_rows * cell_h + (n_rows + 1) * p,
         n_cols * cell_w + (n_cols + 1) * p,
         3),
        bg, dtype=np.uint8,
    )
    for r, group in enumerate(image_groups):
        for c, path in enumerate(group):
            img = _load_resize(path, (cell_w, cell_h))
            y = p + r * (cell_h + p)
            x = p + c * (cell_w + p)
            canvas[y:y + cell_h, x:x + cell_w] = img
    return canvas


# ---------------------------------------------------------------------------
# Selection state
# ---------------------------------------------------------------------------

class SelectionState:
    def __init__(
        self,
        pairs_21:   list[tuple[str, str]],
        triples_22: list[tuple[str, str, str]],
        json_path:  str,
        out_dir:    str,
    ):
        self.pairs_21   = pairs_21
        self.triples_22 = triples_22
        self.json_path  = json_path
        self.out_dir    = out_dir

        # {str(idx): label}  — stored as string keys for JSON compat
        self._sel: dict[str, dict[str, str]] = {"q21": {}, "q22": {}}
        self._idx: dict[str, int]            = {"q21": 0,  "q22": 0}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if not os.path.exists(self.json_path):
            return
        try:
            with open(self.json_path) as f:
                data = json.load(f)
            self._sel = data.get("selections", {"q21": {}, "q22": {}})
            # Advance indices so we resume after last annotated item
            for tab, items in [("q21", self.pairs_21), ("q22", self.triples_22)]:
                sel = self._sel.get(tab, {})
                for i in range(len(items)):
                    if str(i) in sel:
                        self._idx[tab] = i + 1
                    else:
                        break
        except Exception as e:
            print(f"[figsel] Warning: could not load {self.json_path}: {e}")

    def _save(self):
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        with open(self.json_path, "w") as f:
            json.dump({"selections": self._sel}, f, indent=2)

    # ── Accessors ────────────────────────────────────────────────────────────

    def current_idx(self, tab: str) -> int:
        return self._idx[tab]

    def total(self, tab: str) -> int:
        return len(self.pairs_21) if tab == "q21" else len(self.triples_22)

    def get_label(self, tab: str, idx: int) -> str | None:
        return self._sel.get(tab, {}).get(str(idx))

    def counts_21(self):
        vals = self._sel.get("q21", {}).values()
        return {
            "success": sum(1 for v in vals if v == "success"),
            "failure": sum(1 for v in vals if v == "failure"),
            "skip":    sum(1 for v in vals if v == "skip"),
            "target":  TARGET_21,
        }

    def counts_22(self):
        vals = self._sel.get("q22", {}).values()
        return {
            "selected": sum(1 for v in vals if v == "selected"),
            "skip":     sum(1 for v in vals if v == "skip"),
            "target":   TARGET_22,
        }

    # ── Mutations ────────────────────────────────────────────────────────────

    def annotate(self, tab: str, idx: int, label: str):
        if tab not in self._sel:
            self._sel[tab] = {}
        self._sel[tab][str(idx)] = label
        self._save()
        if idx >= self._idx[tab]:
            self._idx[tab] = idx + 1

    def go_back(self, tab: str):
        if self._idx[tab] > 0:
            self._idx[tab] -= 1
            prev = str(self._idx[tab])
            if prev in self._sel.get(tab, {}):
                del self._sel[tab][prev]
                self._save()

    def jump(self, tab: str, n: int):
        """Advance index by n without recording any label."""
        total = len(self.pairs_21) if tab == "q21" else len(self.triples_22)
        self._idx[tab] = min(self._idx[tab] + n, total)

    # ── Export ───────────────────────────────────────────────────────────────

    def export(self, cell_w_21=512, cell_h_21=288, cell_wh_22=256) -> dict[str, str]:
        """Build figure grids and save to self.out_dir.  Returns {name: path}."""
        os.makedirs(self.out_dir, exist_ok=True)
        results = {}

        sel21 = self._sel.get("q21", {})
        success_pairs = [
            self.pairs_21[int(i)] for i, v in sorted(sel21.items(), key=lambda x: int(x[0]))
            if v == "success"
        ][:TARGET_21]
        failure_pairs = [
            self.pairs_21[int(i)] for i, v in sorted(sel21.items(), key=lambda x: int(x[0]))
            if v == "failure"
        ][:TARGET_21]

        sel22 = self._sel.get("q22", {})
        selected_triples = [
            self.triples_22[int(i)] for i, v in sorted(sel22.items(), key=lambda x: int(x[0]))
            if v == "selected"
        ][:TARGET_22]

        for name, groups, cw, ch in [
            ("q21_success", [[op, fp] for op, fp in success_pairs], cell_w_21, cell_h_21),
            ("q21_failure", [[op, fp] for op, fp in failure_pairs], cell_w_21, cell_h_21),
            ("q22_compare", [[op, f21, f22] for op, f21, f22 in selected_triples], cell_wh_22, cell_wh_22),
        ]:
            if not groups:
                print(f"[figsel] Skip {name}: no selections")
                continue
            grid = make_grid(groups, cell_w=cw, cell_h=ch)
            path = os.path.join(self.out_dir, f"{name}.png")
            cv2.imwrite(path, grid)
            print(f"[figsel] Saved {path}  ({len(groups)} rows)")
            results[name] = path

        return results


# ---------------------------------------------------------------------------
# Global state  (set up in main)
# ---------------------------------------------------------------------------

STATE: SelectionState | None = None

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = "acv_figsel_2025"

# ── HTML ────────────────────────────────────────────────────────────────────

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ACV Figure Selector</title>
<style>
  @import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;700;800&display=swap");
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:      #0d0d0f;
    --surface: #141417;
    --border:  #252529;
    --accent:  #e8ff47;
    --green:   #4dff91;
    --red:     #ff4757;
    --blue:    #47d4ff;
    --text:    #e8e8ec;
    --muted:   #6b6b78;
    --mono:    "JetBrains Mono", monospace;
    --sans:    "Syne", sans-serif;
  }
  html, body { background: var(--bg); color: var(--text); font-family: var(--mono);
               height: 100%; overflow: hidden; }

  /* ── Layout ── */
  .shell { display: grid; grid-template-rows: auto auto 1fr auto; height: 100vh; overflow: hidden; }

  /* ── Header ── */
  header { display: flex; align-items: center; justify-content: space-between;
           padding: 10px 20px; border-bottom: 1px solid var(--border);
           background: var(--surface); flex-shrink: 0; }
  .logo  { font-family: var(--sans); font-weight: 800; font-size: 1rem; color: var(--accent); }
  .fname { font-size: 0.65rem; color: var(--muted); max-width: 400px;
           overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* ── Tabs ── */
  .tabs { display: flex; border-bottom: 1px solid var(--border);
          background: var(--surface); flex-shrink: 0; }
  .tab  { padding: 10px 28px; font-size: 0.75rem; font-weight: 600; cursor: pointer;
          letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);
          border-bottom: 2px solid transparent; transition: all 0.15s; user-select: none; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  /* ── Image area ── */
  .viewer { display: flex; align-items: center; justify-content: center;
            gap: 12px; padding: 20px; overflow: hidden; flex: 1 1 0; min-height: 0; }
  .img-wrap { display: flex; flex-direction: column; align-items: center; gap: 6px;
              flex: 1 1 0; max-width: 600px; min-width: 0; }
  .img-label { font-size: 0.60rem; letter-spacing: 0.12em; text-transform: uppercase;
               color: var(--muted); }
  .img-wrap img { width: 100%; height: auto; max-height: calc(100vh - 220px);
                  object-fit: contain; border: 1px solid var(--border); border-radius: 4px;
                  transition: opacity 0.2s; }

  /* ── Controls bar ── */
  .controls { display: flex; align-items: center; justify-content: space-between;
              padding: 0 20px; height: 56px; border-top: 1px solid var(--border);
              background: var(--surface); flex-shrink: 0; }
  .btn-group { display: flex; gap: 8px; align-items: center; }
  .btn  { display: flex; align-items: center; gap: 7px; padding: 7px 16px;
          font-family: var(--mono); font-size: 0.72rem; font-weight: 600;
          border: 1px solid var(--border); border-radius: 4px; cursor: pointer;
          background: transparent; color: var(--text); transition: all 0.12s;
          letter-spacing: 0.04em; }
  .btn:hover { background: var(--border); }
  .btn.success { border-color: var(--green); color: var(--green); }
  .btn.success:hover { background: rgba(77,255,145,0.12); }
  .btn.failure { border-color: var(--red);   color: var(--red); }
  .btn.failure:hover { background: rgba(255,71,87,0.12); }
  .btn.select  { border-color: var(--blue);  color: var(--blue); }
  .btn.select:hover  { background: rgba(71,212,255,0.12); }
  .btn.skip    { border-color: var(--muted); color: var(--muted); }
  .btn.export  { border-color: var(--accent); color: var(--accent); }
  .btn.export:hover  { background: rgba(232,255,71,0.12); }
  .key-badge { font-size: 0.60rem; background: var(--border); padding: 1px 5px;
               border-radius: 3px; color: var(--muted); }

  /* ── Progress pills ── */
  .pills { display: flex; gap: 10px; align-items: center; }
  .pill  { font-size: 0.62rem; padding: 3px 10px; border-radius: 20px;
           border: 1px solid var(--border); color: var(--muted); letter-spacing: 0.06em; }
  .pill.green { border-color: var(--green); color: var(--green); }
  .pill.red   { border-color: var(--red);   color: var(--red); }
  .pill.blue  { border-color: var(--blue);  color: var(--blue); }

  /* ── Empty state ── */
  .empty { display: flex; flex-direction: column; align-items: center; justify-content: center;
           flex: 1 1 0; gap: 12px; }
  .empty h2 { font-family: var(--sans); font-size: 1.2rem; color: var(--muted); }
  .empty p  { font-size: 0.72rem; color: var(--muted); }

  /* ── Export overlay ── */
  .overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7);
             display: none; align-items: center; justify-content: center; z-index: 100; }
  .overlay.visible { display: flex; }
  .modal { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
           padding: 28px 32px; min-width: 440px; max-width: 600px; }
  .modal h2 { font-family: var(--sans); font-size: 1rem; color: var(--accent); margin-bottom: 16px; }
  .modal .result-row { font-size: 0.72rem; padding: 5px 0; border-bottom: 1px solid var(--border);
                       color: var(--text); }
  .modal .result-row span { color: var(--muted); font-size: 0.65rem; display: block; margin-top: 2px; }
  .modal .close-btn { margin-top: 18px; }
  .progress-row { display: flex; align-items: center; gap: 10px; font-size: 0.65rem; color: var(--muted); }
  .pbar-outer { flex: 1; height: 3px; background: var(--border); border-radius: 2px; }
  .pbar-inner { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.3s; }

  /* ── Toast ── */
  .toast { position: fixed; bottom: 70px; left: 50%; transform: translateX(-50%);
           background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
           padding: 8px 18px; font-size: 0.70rem; color: var(--text);
           opacity: 0; transition: opacity 0.2s; pointer-events: none; z-index: 50; }
  .toast.show { opacity: 1; }
</style>
</head>
<body>
<div class="shell" id="shell">
  <!-- Header -->
  <header>
    <span class="logo">ACV · Figure Selector</span>
    <span class="fname" id="fname">–</span>
    <div class="pills" id="pills"></div>
  </header>

  <!-- Tabs -->
  <div class="tabs">
    <div class="tab active" id="tab-q21" onclick="switchTab('q21')">2.1 — Pairs</div>
    <div class="tab"        id="tab-q22" onclick="switchTab('q22')">2.2 — Triples</div>
  </div>

  <!-- Viewer -->
  <div id="viewer" class="viewer"></div>

  <!-- Controls -->
  <div class="controls">
    <div class="btn-group" id="btn-group-q21">
      <button class="btn success" onclick="annotate('success')">
        <span class="key-badge">S</span> Success
      </button>
      <button class="btn failure" onclick="annotate('failure')">
        <span class="key-badge">F</span> Failure
      </button>
      <button class="btn skip"    onclick="jumpForward()">
        <span class="key-badge">Space</span> Skip +{{ skip_interval }}
      </button>
      <button class="btn"         onclick="goBack()">
        <span class="key-badge">B</span> Back
      </button>
    </div>
    <div class="btn-group" id="btn-group-q22" style="display:none">
      <button class="btn select" onclick="annotate('selected')">
        <span class="key-badge">Y</span> Select
      </button>
      <button class="btn skip"   onclick="jumpForward()">
        <span class="key-badge">Space</span> Skip +{{ skip_interval }}
      </button>
      <button class="btn"        onclick="goBack()">
        <span class="key-badge">B</span> Back
      </button>
    </div>
    <div class="btn-group">
      <button class="btn export" onclick="exportFigures()">
        <span class="key-badge">Q</span> Export &amp; Save
      </button>
    </div>
  </div>
</div>

<!-- Export overlay -->
<div class="overlay" id="overlay">
  <div class="modal">
    <h2>Export complete</h2>
    <div id="export-results"></div>
    <button class="btn close-btn" onclick="closeOverlay()">Close</button>
  </div>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<script>
// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let currentTab = 'q21';
let state = { q21: { idx: 0, total: 0 }, q22: { idx: 0, total: 0 } };
let toastTimer = null;
const skipInterval = {{ skip_interval }};

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tab-q21').classList.toggle('active', tab === 'q21');
  document.getElementById('tab-q22').classList.toggle('active', tab === 'q22');
  document.getElementById('btn-group-q21').style.display = tab === 'q21' ? '' : 'none';
  document.getElementById('btn-group-q22').style.display = tab === 'q22' ? '' : 'none';
  loadCurrent();
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------
function imgUrl(tab, idx, slot) {
  return `/api/image/${tab}/${idx}/${slot}?t=${Date.now()}`;
}

function loadCurrent() {
  fetch(`/api/state/${currentTab}`)
    .then(r => r.json())
    .then(data => {
      state[currentTab] = data;
      renderViewer(data);
      renderPills(data);
      document.getElementById('fname').textContent =
        data.fnames ? data.fnames.join('  ·  ') : '–';
    });
}

function renderViewer(data) {
  const v = document.getElementById('viewer');
  if (data.finished) {
    v.innerHTML = `<div class="empty">
      <h2>All done</h2>
      <p>${data.idx} images reviewed · press Q to export</p>
    </div>`;
    return;
  }
  if (data.total === 0) {
    v.innerHTML = `<div class="empty">
      <h2>No images found</h2>
      <p>Check your --${currentTab === 'q21' ? 'q21' : 'q22'}-* arguments</p>
    </div>`;
    return;
  }

  const slots = currentTab === 'q21'
    ? [['orig', 'Original'], ['fake', 'Translated']]
    : [['orig', 'Original'], ['fake21', '2.1 Output'], ['fake22', '2.2 Output']];

  v.innerHTML = slots.map(([slot, label]) => `
    <div class="img-wrap">
      <div class="img-label">${label}</div>
      <img id="img-${slot}" src="${imgUrl(currentTab, data.idx, slot)}"
           alt="${label}" onerror="this.style.opacity='0.2'">
    </div>
  `).join('');

  // Highlight already-labelled images if going back
  if (data.current_label) {
    showToast(`Previously: ${data.current_label}`, 1200);
  }
}

function renderPills(data) {
  const pills = document.getElementById('pills');
  if (!data.counts) { pills.innerHTML = ''; return; }
  const c = data.counts;
  if (currentTab === 'q21') {
    pills.innerHTML = `
      <span class="pill">${data.idx}/${data.total}</span>
      <span class="pill green">✓ ${c.success}/${c.target} success</span>
      <span class="pill red">✗ ${c.failure}/${c.target} failure</span>`;
  } else {
    pills.innerHTML = `
      <span class="pill">${data.idx}/${data.total}</span>
      <span class="pill blue">◆ ${c.selected}/${c.target} selected</span>`;
  }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------
function annotate(label) {
  fetch('/api/annotate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ tab: currentTab, label }),
  })
  .then(r => r.json())
  .then(() => {
    showToast(label, 600);
    loadCurrent();
  });
}

function goBack() {
  fetch('/api/back', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ tab: currentTab }),
  })
  .then(r => r.json())
  .then(() => loadCurrent());
}

function jumpForward() {
  fetch('/api/jump', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ tab: currentTab, n: skipInterval }),
  })
  .then(r => r.json())
  .then(() => {
    showToast(`+${skipInterval}`, 400);
    loadCurrent();
  });
}

function exportFigures() {
  fetch('/api/export', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      const res = document.getElementById('export-results');
      if (Object.keys(data.results).length === 0) {
        res.innerHTML = '<div class="result-row">Nothing exported (no selections yet)</div>';
      } else {
        res.innerHTML = Object.entries(data.results).map(([name, path]) =>
          `<div class="result-row">${name}<span>${path}</span></div>`
        ).join('');
      }
      document.getElementById('overlay').classList.add('visible');
    });
}

function closeOverlay() {
  document.getElementById('overlay').classList.remove('visible');
}

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------
function showToast(msg, duration=1000) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), duration);
}

// ---------------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------------
document.addEventListener('keydown', e => {
  if (document.getElementById('overlay').classList.contains('visible')) {
    if (e.key === 'Escape') closeOverlay();
    return;
  }
  const k = e.key;
  if (k === '1') { switchTab('q21'); return; }
  if (k === '2') { switchTab('q22'); return; }
  if (currentTab === 'q21') {
    if (k === 's' || k === 'S') { annotate('success'); return; }
    if (k === 'f' || k === 'F') { annotate('failure'); return; }
  }
  if (currentTab === 'q22') {
    if (k === 'y' || k === 'Y') { annotate('selected'); return; }
    if (k === 'n' || k === 'N') { annotate('skip');    return; }
  }
  if (k === ' ' || k === 'ArrowRight') { e.preventDefault(); jumpForward(); return; }
  if (k === 'b' || k === 'B') { goBack(); return; }
  if (k === 'q' || k === 'Q') { exportFigures(); return; }
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
loadCurrent();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML, skip_interval=STATE.skip_interval if STATE else 30)


@app.route("/api/state/<tab>")
def api_state(tab):
    if STATE is None or tab not in ("q21", "q22"):
        return jsonify({"error": "not ready"}), 500

    idx   = STATE.current_idx(tab)
    total = STATE.total(tab)

    if tab == "q21":
        items  = STATE.pairs_21
        counts = STATE.counts_21()
    else:
        items  = STATE.triples_22
        counts = STATE.counts_22()

    finished = idx >= total
    fnames   = None
    if not finished and items:
        group  = items[idx]
        fnames = [os.path.basename(p) for p in group]

    return jsonify({
        "tab":           tab,
        "idx":           idx,
        "total":         total,
        "finished":      finished,
        "fnames":        fnames,
        "counts":        counts,
        "current_label": STATE.get_label(tab, idx - 1) if idx > 0 else None,
    })


@app.route("/api/image/<tab>/<int:idx>/<slot>")
def api_image(tab, idx, slot):
    if STATE is None:
        return "not ready", 500

    try:
        if tab == "q21":
            pair = STATE.pairs_21[idx]
            path = pair[0] if slot == "orig" else pair[1]
        else:
            triple = STATE.triples_22[idx]
            path   = {"orig": triple[0], "fake21": triple[1], "fake22": triple[2]}.get(slot)
    except (IndexError, KeyError):
        return "not found", 404

    if not path or not os.path.exists(path):
        # Return a small placeholder image
        placeholder = np.full((288, 512, 3), 30, dtype=np.uint8)
        cv2.putText(placeholder, f"missing: {slot}", (20, 144),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        _, buf = cv2.imencode(".jpg", placeholder)
        return buf.tobytes(), 200, {"Content-Type": "image/jpeg"}

    return send_file(os.path.abspath(path), mimetype="image/jpeg" if path.lower().endswith(".jpg") else "image/png")


@app.route("/api/annotate", methods=["POST"])
def api_annotate():
    if STATE is None:
        return jsonify({"error": "not ready"}), 500
    data  = request.get_json()
    tab   = data.get("tab")
    label = data.get("label")
    idx   = STATE.current_idx(tab)
    total = STATE.total(tab)
    if idx >= total:
        return jsonify({"status": "finished"})
    STATE.annotate(tab, idx, label)
    return jsonify({"status": "ok", "idx": idx, "label": label})


@app.route("/api/back", methods=["POST"])
def api_back():
    if STATE is None:
        return jsonify({"error": "not ready"}), 500
    tab = request.get_json().get("tab")
    STATE.go_back(tab)
    return jsonify({"status": "ok", "idx": STATE.current_idx(tab)})


@app.route("/api/jump", methods=["POST"])
def api_jump():
    if STATE is None:
        return jsonify({"error": "not ready"}), 500
    data = request.get_json()
    tab = data.get("tab")
    n   = int(data.get("n", STATE.skip_interval))
    STATE.jump(tab, n)
    return jsonify({"status": "ok", "idx": STATE.current_idx(tab)})


@app.route("/api/export", methods=["POST"])
def api_export():
    if STATE is None:
        return jsonify({"error": "not ready"}), 500
    results = STATE.export()
    return jsonify({"status": "ok", "results": results})


# ---------------------------------------------------------------------------
# Argument parsing + main
# ---------------------------------------------------------------------------

def _resolve_q21_dirs(args) -> tuple[str | None, str | None]:
    orig_dir = getattr(args, "q21_orig_dir", None)
    fake_dir = getattr(args, "q21_fake_dir", None)
    if not orig_dir and args.q21_dir:
        orig_dir, auto_fake = _auto_discover_q21(args.q21_dir)
        if not fake_dir:
            fake_dir = auto_fake
    return orig_dir, fake_dir


def _resolve_q22_dirs(args) -> tuple[str | None, str | None, str | None]:
    orig_dir  = getattr(args, "q22_orig_dir",  None)
    fake21_dir = getattr(args, "q22_21_dir",   None)
    fake22_dir = getattr(args, "q22_22_dir",   None)
    if not orig_dir and args.q22_dir:
        auto_orig, auto_21, auto_22 = _auto_discover_q22(args.q22_dir)
        if not orig_dir:   orig_dir   = auto_orig
        if not fake21_dir: fake21_dir = auto_21
        if not fake22_dir: fake22_dir = auto_22
    return orig_dir, fake21_dir, fake22_dir


def main():
    global STATE

    parser = argparse.ArgumentParser(
        description="Figure selector for Q2.1 (pairs) and Q2.2 (triples)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Q2.1
    parser.add_argument("--q21-dir",      default=None,
                        help="Q2.1 run dir (auto-discovers orig/fake subdirs)")
    parser.add_argument("--q21-orig-dir", default=None, help="Override: original frames dir")
    parser.add_argument("--q21-fake-dir", default=None, help="Override: translated frames dir")
    # Q2.2
    parser.add_argument("--q22-dir",      default=None,
                        help="Q2.2 run dir (auto-discovers orig/fake21/fake22 subdirs)")
    parser.add_argument("--q22-orig-dir", default=None, help="Override: original frames dir")
    parser.add_argument("--q22-21-dir",   default=None, help="Override: 2.1-translated frames dir")
    parser.add_argument("--q22-22-dir",   default=None, help="Override: 2.2-translated frames dir")
    # Output
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "paper", "figs"),
                        help="Directory to write figure PNGs (default: paper/figs)")
    parser.add_argument("--sel-dir", default=os.path.join(PROJECT_ROOT, "output", "figure_select"),
                        help="Directory for selections JSON (default: output/figure_select)")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--skip-interval", type=int, default=30,
                        help="Number of frames to jump forward on Space/→ (default: 30)")
    args = parser.parse_args()

    # Discover dirs
    orig21_dir, fake21_dir = _resolve_q21_dirs(args)
    orig22_dir, f21_dir, f22_dir = _resolve_q22_dirs(args)

    # Load images
    orig21   = _find_images(orig21_dir)
    fake21   = _find_images(fake21_dir)
    orig22   = _find_images(orig22_dir)
    fake21_2 = _find_images(f21_dir)
    fake22   = _find_images(f22_dir)

    pairs_21   = _match_pairs(orig21, fake21)
    triples_22 = _match_triples(orig22, fake21_2, fake22)

    # Report
    print(f"[figsel] Q2.1 pairs:   {len(pairs_21)}")
    print(f"         orig_dir  = {orig21_dir}")
    print(f"         fake_dir  = {fake21_dir}")
    print(f"[figsel] Q2.2 triples: {len(triples_22)}")
    print(f"         orig_dir  = {orig22_dir}")
    print(f"         fake21    = {f21_dir}")
    print(f"         fake22    = {f22_dir}")
    print(f"[figsel] Selections → {args.sel_dir}/selections.json")
    print(f"[figsel] Figures    → {args.out_dir}")
    print(f"[figsel] Open http://localhost:{args.port}")

    json_path = os.path.join(args.sel_dir, "selections.json")
    STATE = SelectionState(pairs_21, triples_22, json_path, args.out_dir)
    STATE.skip_interval = args.skip_interval
    print(f"[figsel] Skip interval: {args.skip_interval} frames (Space/→)")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
