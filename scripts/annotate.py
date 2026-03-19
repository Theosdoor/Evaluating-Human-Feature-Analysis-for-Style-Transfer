"""
scripts/annotate.py

Flask-based interactive annotation tool for manually labelling human patches.

Loads patches from an existing classification directory (output/classifications/<run>),
renders the deep diagnostic once per image (cached to disk), and serves a local
web UI for accept/reclassify/back/quit.

Usage
-----
    python3 scripts/annotate.py --cls-dir output/classifications/20260314-195748-5

    Then open http://localhost:5000 in your browser.

    # Resume a partially-annotated session
    python3 scripts/annotate.py --cls-dir output/classifications/20260314-195748-5 \\
        --out-dir output/manual_annotated/20260314-195748-5

Output
------
    output/manual_annotated/<run>/annotations.json
        { filename: { "label": str, "source": "auto" | "manual" } }

    output/manual_annotated/<run>/<class>/
        Copies of annotated patches in per-class subdirs, mirroring the
        classification output format so the downstream pipeline can consume
        them without changes.  Written on exit via the /finish endpoint.
"""

import argparse
import json
import os
import shutil
import sys
import threading

import cv2

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.classification import (
    CLASSES,
    DEFAULT_CONFIG,
    render_diagnostic,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_LABELS = CLASSES + ['bad_extraction']

# ---------------------------------------------------------------------------
# State — single global object (single-user local tool)
# ---------------------------------------------------------------------------

class AnnotationState:
    def __init__(self, patches, annotations, json_path, out_dir, diag_cache_dir):
        # patches: list of (abs_path, rule_label)
        self.patches        = patches
        self.annotations    = annotations
        self.json_path      = json_path
        self.out_dir        = out_dir
        self.diag_cache_dir = diag_cache_dir

        self.all_fnames    = [os.path.basename(p) for p, _ in patches]
        self.fname_to_src  = {os.path.basename(p): (p, lbl) for p, lbl in patches}

        # Work queue: only unannotated images, in order
        self.queue = [f for f in self.all_fnames if f not in annotations]
        self.idx   = 0
        self.pose_model     = None
        self._model_lock    = threading.Lock()

    def load_model(self, model_path):
        with self._model_lock:
            if self.pose_model is None:
                from ultralytics import YOLO
                self.pose_model = YOLO(model_path)

    @property
    def n_total(self):
        return len(self.all_fnames)

    @property
    def n_done(self):
        return len(self.annotations)

    @property
    def finished(self):
        return self.idx >= len(self.queue)

    def current_fname(self):
        return None if self.finished else self.queue[self.idx]

    def save(self):
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        with open(self.json_path, "w") as f:
            json.dump(self.annotations, f, indent=2)

    def accept(self, fname, label, source):
        self.annotations[fname] = {"label": label, "source": source}
        self.save()
        self.idx += 1

    def go_back(self):
        if self.idx > 0:
            self.idx -= 1
            prev = self.queue[self.idx]
            if prev in self.annotations:
                del self.annotations[prev]
                self.save()
            return prev
        return None

    def diag_cache_path(self, fname):
        return os.path.join(self.diag_cache_dir, fname)

    def get_or_render_diag(self, fname, label):
        cache_path = self.diag_cache_path(fname)
        if os.path.exists(cache_path):
            return cache_path
        src_path, _ = self.fname_to_src[fname]
        img_bgr = cv2.imread(src_path)
        if img_bgr is None:
            return None
        result = self.pose_model(img_bgr, verbose=False)[0]
        diag   = render_diagnostic(result, img_bgr, DEFAULT_CONFIG, predicted_class=label)
        os.makedirs(self.diag_cache_dir, exist_ok=True)
        cv2.imwrite(cache_path, diag, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return cache_path

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


STATE: AnnotationState | None = None

# ---------------------------------------------------------------------------
# Patch collection + annotation persistence
# ---------------------------------------------------------------------------

def collect_patches(cls_dir):
    patches = []
    for cls in CLASSES:
        subdir = os.path.join(cls_dir, cls)
        if not os.path.isdir(subdir):
            continue
        for fname in sorted(os.listdir(subdir)):
            if fname.lower().endswith(('.jpg', '.png')):
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

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ACV Annotator</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #0d0d0f;
    --surface: #141417;
    --border:  #252529;
    --accent:  #e8ff47;
    --accent2: #47d4ff;
    --danger:  #ff4757;
    --text:    #e8e8ec;
    --muted:   #6b6b78;
    --mono:    'JetBrains Mono', monospace;
    --sans:    'Syne', sans-serif;
  }

  html, body { background: var(--bg); color: var(--text); font-family: var(--mono); min-height: 100vh; }

  .shell { display: grid; grid-template-rows: auto 1fr auto; min-height: 100vh; }

  /* Header */
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 28px; border-bottom: 1px solid var(--border);
    background: var(--surface); position: sticky; top: 0; z-index: 10;
  }
  .logo { font-family: var(--sans); font-weight: 800; font-size: 1rem; letter-spacing: -0.02em; color: var(--accent); }
  .progress-wrap { display: flex; align-items: center; gap: 14px; }
  .progress-text { font-size: 0.72rem; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; }
  .progress-bar-outer { width: 180px; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .progress-bar-inner { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.3s ease; }
  .fname { font-size: 0.7rem; color: var(--muted); max-width: 340px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* Main */
  main { display: flex; flex-direction: column; align-items: center; padding: 24px 28px; gap: 20px; }

  .prediction-badge {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 18px; border: 1px solid var(--border);
    border-radius: 4px; background: var(--surface); font-size: 0.75rem;
  }
  .badge-label { color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.65rem; }
  .badge-value { color: var(--accent2); font-weight: 600; }

  .diag-wrap { width: 100%; max-width: 1400px; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; background: #000; }
  .diag-wrap img { width: 100%; height: auto; display: block; }
  .diag-loading { width: 100%; height: 320px; display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 0.8rem; letter-spacing: 0.1em; }

  /* Controls */
  .controls { display: flex; flex-direction: column; align-items: center; gap: 14px; width: 100%; max-width: 700px; }
  .btn-row { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; }

  button {
    font-family: var(--mono); font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em;
    padding: 10px 22px; border: 1px solid var(--border); border-radius: 4px;
    background: var(--surface); color: var(--text); cursor: pointer;
    transition: background 0.15s, border-color 0.15s, color 0.15s, transform 0.1s;
    display: flex; align-items: center; gap: 8px;
  }
  button:hover  { background: #1e1e24; border-color: #3a3a42; }
  button:active { transform: scale(0.97); }

  .btn-accept { background: #1a2400; border-color: #3a5200; color: var(--accent); }
  .btn-accept:hover { background: #223000; border-color: var(--accent); }
  .btn-back   { color: var(--muted); }
  .btn-finish { color: var(--danger); border-color: #3a1a1a; }
  .btn-finish:hover { background: #1e0f0f; border-color: var(--danger); }

  .kbd { display: inline-block; background: var(--border); border-radius: 3px; padding: 1px 6px; font-size: 0.68rem; color: var(--muted); }

  /* Reclassify panel */
  .reclassify-panel {
    display: none; flex-direction: column; gap: 10px; width: 100%;
    padding: 16px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface);
  }
  .reclassify-panel.open { display: flex; }
  .reclassify-title { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 4px; }
  .cls-btn-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .cls-btn { justify-content: flex-start; padding: 9px 14px; font-size: 0.73rem; }
  .cls-btn.bad { color: var(--danger); border-color: #3a1a1a; }
  .cls-btn.bad:hover { background: #1e0f0f; }

  /* Done screen */
  .done-screen { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; min-height: 60vh; text-align: center; }
  .done-title { font-family: var(--sans); font-size: 2rem; font-weight: 800; color: var(--accent); }
  .done-sub { font-size: 0.8rem; color: var(--muted); line-height: 1.7; }

  /* Footer */
  footer { padding: 10px 28px; border-top: 1px solid var(--border); font-size: 0.65rem; color: var(--muted); letter-spacing: 0.06em; display: flex; justify-content: space-between; }

  /* Spinner */
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner { width: 18px; height: 18px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.7s linear infinite; flex-shrink: 0; }

  /* Toast */
  .toast { position: fixed; bottom: 24px; right: 24px; background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 10px 18px; font-size: 0.75rem; opacity: 0; transform: translateY(8px); transition: opacity 0.2s, transform 0.2s; pointer-events: none; z-index: 100; }
  .toast.show { opacity: 1; transform: translateY(0); }
  .toast.ok  { border-color: #3a5200; color: var(--accent); }
  .toast.err { border-color: #3a1a1a; color: var(--danger); }
</style>
</head>
<body>
<div class="shell">

<header>
  <span class="logo">ACV · ANNOTATOR</span>
  <div class="progress-wrap">
    <span class="progress-text">{{ n_done }} / {{ n_total }}</span>
    <div class="progress-bar-outer">
      <div class="progress-bar-inner" style="width:{{ pct }}%"></div>
    </div>
  </div>
  <span class="fname">{{ fname or '' }}</span>
</header>

<main>
{% if finished %}
  <div class="done-screen">
    <div class="done-title">All done.</div>
    <div class="done-sub">{{ n_done }} patches annotated.<br>Click below to write output directories and close the server.</div>
    <button class="btn-finish" onclick="finish()">Write outputs &amp; exit</button>
  </div>
{% else %}
  <div class="prediction-badge">
    <span class="badge-label">rule-based</span>
    <span class="badge-value" id="rule-label">{{ rule_label }}</span>
    <span style="color:var(--border)">│</span>
    <span class="badge-label">current</span>
    <span class="badge-value" id="current-label">{{ current_label }}</span>
  </div>

  <div class="diag-wrap">
    <div class="diag-loading" id="diag-loading"><div class="spinner"></div>&nbsp;&nbsp;rendering diagnostic…</div>
    <img id="diag-img" src="" alt="diagnostic" style="display:none" onload="imgLoaded()" onerror="imgError()">
  </div>

  <div class="controls">
    <div class="btn-row">
      <button class="btn-accept" onclick="accept()"><span class="kbd">Y</span> Accept</button>
      <button onclick="toggleReclassify()"><span class="kbd">N</span> Reclassify</button>
      <button class="btn-back"   onclick="goBack()"><span class="kbd">B</span> Back</button>
      <button class="btn-finish" onclick="finish()"><span class="kbd">Q</span> Save &amp; exit</button>
    </div>

    <div class="reclassify-panel" id="reclassify-panel">
      <div class="reclassify-title">Reclassify as —</div>
      <div class="cls-btn-grid">
        <button class="cls-btn" onclick="reclassify('full_body_front')"><span class="kbd">1</span> full_body_front</button>
        <button class="cls-btn" onclick="reclassify('full_body_back')"><span class="kbd">2</span> full_body_back</button>
        <button class="cls-btn" onclick="reclassify('head_shoulder_front')"><span class="kbd">3</span> head_shoulder_front</button>
        <button class="cls-btn" onclick="reclassify('head_shoulder_back')"><span class="kbd">4</span> head_shoulder_back</button>
        <button class="cls-btn" onclick="reclassify('others')"><span class="kbd">5</span> others</button>
        <button class="cls-btn bad" onclick="reclassify('bad_extraction')"><span class="kbd">6</span> bad_extraction</button>
      </div>
    </div>
  </div>
{% endif %}
</main>

<footer>
  <span>output → {{ out_dir }}</span>
  <span>annotations.json auto-saved on every action</span>
</footer>
</div>

<div class="toast" id="toast"></div>

<script>
const FNAME        = {{ fname | tojson }};
const RULE_LABEL   = {{ rule_label | tojson }};
let   currentLabel = {{ current_label | tojson }};
let   busy         = false;

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const open = document.getElementById('reclassify-panel')?.classList.contains('open');
  if (open) {
    const map = {'1':'full_body_front','2':'full_body_back','3':'head_shoulder_front','4':'head_shoulder_back','5':'others','6':'bad_extraction'};
    if (map[e.key]) reclassify(map[e.key]);
    if (e.key === 'Escape' || e.key.toLowerCase() === 'n') toggleReclassify();
  } else {
    if (e.key.toLowerCase() === 'y') accept();
    if (e.key.toLowerCase() === 'n') toggleReclassify();
    if (e.key.toLowerCase() === 'b') goBack();
    if (e.key.toLowerCase() === 'q') finish();
  }
});

function loadDiag(fname, label) {
  const img = document.getElementById('diag-img');
  const ldr = document.getElementById('diag-loading');
  img.style.display = 'none';
  ldr.style.display = 'flex';
  img.src = `/diag/${encodeURIComponent(fname)}?label=${encodeURIComponent(label)}&t=${Date.now()}`;
}

function imgLoaded() {
  document.getElementById('diag-loading').style.display = 'none';
  document.getElementById('diag-img').style.display = 'block';
}

function imgError() {
  document.getElementById('diag-loading').innerHTML = '<span style="color:var(--danger)">Failed to render diagnostic.</span>';
}

if (FNAME) loadDiag(FNAME, currentLabel);

function post(url, body) {
  if (busy) return Promise.resolve(null);
  busy = true;
  return fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) })
    .then(r => r.json())
    .finally(() => { busy = false; });
}

function accept() {
  post('/accept', {fname: FNAME, label: currentLabel})
    .then(d => { if (d?.redirect) window.location = d.redirect; });
}

function reclassify(label) {
  currentLabel = label;
  document.getElementById('current-label').textContent = label;
  document.getElementById('reclassify-panel').classList.remove('open');
  post('/invalidate', {fname: FNAME}).then(() => loadDiag(FNAME, label));
  post('/accept', {fname: FNAME, label: label, source: 'manual'})
    .then(d => {
      showToast(`→ ${label}`, 'ok');
      setTimeout(() => { if (d?.redirect) window.location = d.redirect; }, 600);
    });
}

function toggleReclassify() {
  document.getElementById('reclassify-panel').classList.toggle('open');
}

function goBack() {
  post('/back', {}).then(d => { if (d?.redirect) window.location = d.redirect; });
}

function finish() {
  post('/finish', {}).then(d => {
    if (d?.message) showToast(d.message, 'ok');
    setTimeout(() => window.location = '/', 1200);
  });
}

function showToast(msg, type) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = `toast ${type} show`;
  setTimeout(() => t.classList.remove('show'), 2000);
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _ctx():
    s           = STATE
    fname       = s.current_fname()
    rule_label  = s.fname_to_src[fname][1] if fname else ""
    current_lbl = s.annotations.get(fname, {}).get("label", rule_label) if fname else ""
    pct         = round(s.n_done / s.n_total * 100, 1) if s.n_total else 0
    return dict(fname=fname, rule_label=rule_label, current_label=current_lbl,
                n_done=s.n_done, n_total=s.n_total, pct=pct,
                finished=s.finished, out_dir=s.out_dir)


@app.route("/")
def index():
    return render_template_string(HTML, **_ctx())


@app.route("/diag/<fname>")
def diag(fname):
    label      = request.args.get("label", "")
    cache_path = STATE.get_or_render_diag(fname, label)
    if not cache_path or not os.path.exists(cache_path):
        return Response(status=404)
    return send_file(cache_path, mimetype="image/jpeg")


@app.route("/accept", methods=["POST"])
def accept():
    data   = request.get_json()
    fname  = data["fname"]
    label  = data["label"]
    source = data.get("source", "auto" if label == STATE.fname_to_src[fname][1] else "manual")
    STATE.accept(fname, label, source)
    return jsonify(redirect="/")


@app.route("/invalidate", methods=["POST"])
def invalidate():
    fname = request.get_json().get("fname")
    if fname:
        path = STATE.diag_cache_path(fname)
        if os.path.exists(path):
            os.remove(path)
    return jsonify(ok=True)


@app.route("/back", methods=["POST"])
def back():
    STATE.go_back()
    return jsonify(redirect="/")


@app.route("/finish", methods=["POST"])
def finish():
    STATE.save()
    copied = STATE.write_output_dirs()
    msg    = f"Saved {STATE.n_done} annotations. Wrote {copied} files."
    print(f"\n{msg}")
    print(f"Annotations → {STATE.json_path}")
    print(f"Output dirs → {STATE.out_dir}")
    def _shutdown():
        import time, signal
        time.sleep(1.2)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_shutdown, daemon=True).start()
    return jsonify(message=msg, redirect="/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Flask annotation tool for human patch labelling.")
    parser.add_argument("--cls-dir",    required=True)
    parser.add_argument("--out-dir",    default=None)
    parser.add_argument("--pose-model", default=None)
    parser.add_argument("--port",       type=int, default=5000)
    parser.add_argument("--host",       default="127.0.0.1")
    args = parser.parse_args()

    cls_dir    = os.path.abspath(args.cls_dir)
    run_name   = os.path.basename(os.path.normpath(cls_dir))
    out_dir    = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(PROJECT_ROOT, "output", "manual_annotated", run_name)
    pose_model = args.pose_model or os.path.join(PROJECT_ROOT, "models", "yolo26m-pose.pt")

    if not os.path.isdir(cls_dir):   print(f"Error: cls-dir not found: {cls_dir}");   sys.exit(1)
    if not os.path.exists(pose_model): print(f"Error: pose model not found: {pose_model}"); sys.exit(1)

    patches     = collect_patches(cls_dir)
    json_path   = os.path.join(out_dir, "annotations.json")
    annotations = load_annotations(json_path)
    diag_cache  = os.path.join(out_dir, ".diag_cache")

    global STATE
    STATE = AnnotationState(patches, annotations, json_path, out_dir, diag_cache)
    STATE.load_model(pose_model)

    print(f"\nACV Annotator")
    print(f"  cls-dir : {cls_dir}")
    print(f"  out-dir : {out_dir}")
    print(f"  patches : {STATE.n_total}  (todo: {len(STATE.queue)}  done: {STATE.n_done})")
    print(f"\n  → http://{args.host}:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()