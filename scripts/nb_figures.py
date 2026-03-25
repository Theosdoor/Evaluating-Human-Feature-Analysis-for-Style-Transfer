# nb_figures.py — standalone figure generation for the paper
# Run cells individually with #%% notation (Spyder / VS Code Jupyter)
# Outputs go to figures/ (git-tracked) and paper/figs/ (copy manually or symlink)

# %%
# ---------------------------------------------------------------------------
# Imports and global config
# ---------------------------------------------------------------------------

import re
import sys
import json
import base64
import io
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import umap as umap_lib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_train_split, CLASSES, _load_dino_model, _embed_patches

FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DINO_CKPT = ROOT / "models/dinov2_vitb14_reg4_pretrain.pt"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# ---------------------------------------------------------------------------
# 1.1 — Histogram of detection scores (before / after selection)
#
# Source : output/extracted_humans/20260324-185427/_summary.txt  (before stats)
#          output/extracted_humans/20260324-185427/*.jpg          (after scores)
# Save to: figures/score_histogram.pdf
# ---------------------------------------------------------------------------

EXTRACT_DIR = ROOT / "output/extracted_humans/20260324-185427"

# --- parse "after" scores from filenames -----------------------------------
score_re = re.compile(r"_score(\d+\.\d+)\.jpg$")
after_scores = []
for p in EXTRACT_DIR.glob("human_*.jpg"):
    m = score_re.search(p.name)
    if m:
        after_scores.append(float(m.group(1)))
after_scores = np.array(after_scores)

# --- parse "before" aggregate stats from _summary.txt ----------------------
summary_text = (EXTRACT_DIR / "_summary.txt").read_text()
det_block = summary_text.split("Detection score stats:")[1]

def _stat(label, text):
    m = re.search(rf"{label}:\s+([\d.]+)", text)
    return float(m.group(1)) if m else None

before_mean = _stat("mean", det_block)
n_raw       = int(re.search(r"Total raw detections:\s+(\d+)", summary_text).group(1))
n_selected  = len(after_scores)

print(f"Raw mean: {before_mean:.3f} (n={n_raw})   Selected mean: {after_scores.mean():.3f} (n={n_selected})")

# --- plot ------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

fig, ax = plt.subplots(figsize=(4.5, 2.8))

sns.histplot(after_scores, bins=30, ax=ax, color="#1565C0", alpha=0.75,
             kde=True, line_kws={"linewidth": 1.8}, label=f"Selected (n={n_selected})")

ax.axvline(before_mean, color="#E53935", linestyle="--", linewidth=1.5,
           label=f"All detections mean = {before_mean:.2f} (n={n_raw})")
ax.axvline(after_scores.mean(), color="#1565C0", linestyle="--", linewidth=1.5,
           label=f"Selected mean = {after_scores.mean():.2f}")

ax.set_xlabel("Composite quality score")
ax.set_ylabel("Count")
ax.legend(fontsize=8.5, framealpha=0.9)
ax.set_xlim(0.05, 0.80)
sns.despine(ax=ax)
fig.tight_layout()

out_path = FIGURES_DIR / "score_histogram.pdf"  # PDF = vector, preferred for LaTeX \includegraphics
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()

# %%
# ---------------------------------------------------------------------------
# 1.3 — Interactive UMAP (Plotly, self-contained HTML)
#
# Source : output/train_select/<RUN_ID>/train_split.json
# Save to: figures/dino_umap_<RUN_ID>_interactive.html
#
# Hover over points to see class / domain / filename.
# Click a point to show the patch image in a floating panel.
# ---------------------------------------------------------------------------

RUN_ID = "20260325-105918"

# --- load split -------------------------------------------------------
SPLIT_DIR = ROOT / "output/train_select" / RUN_ID
game_paths, movie_paths = load_train_split(str(SPLIT_DIR))

all_paths = game_paths + movie_paths
domains   = ["game"] * len(game_paths) + ["movie"] * len(movie_paths)
labels    = [pathlib.Path(p).parent.name for p in all_paths]
print(f"Loaded {len(game_paths)} game + {len(movie_paths)} movie patches")

# --- DINOv2 embed -----------------------------------------------------
model = _load_dino_model(str(DINO_CKPT), DEVICE)
embs  = _embed_patches(all_paths, model, DEVICE)
del model
torch.cuda.empty_cache() if DEVICE == "cuda" else None

# --- UMAP -------------------------------------------------------------
print("[DATA] Computing UMAP projection...")
xy = umap_lib.UMAP(n_components=2, random_state=42).fit_transform(embs)

# --- encode click images (128×128 JPEG) --------------------------------
def _b64(path, size=(128, 128), quality=65):
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

print("[DATA] Encoding patch thumbnails...")
click_b64 = [_b64(p) for p in tqdm(all_paths, unit="img")]

# --- colour palettes --------------------------------------------------
cls_order  = [c for c in CLASSES if c != "others"] + ["others"]
PALETTE    = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#aaaaaa"]
cls_colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cls_order)}
dom_colors = {"game": "#1f77b4", "movie": "#ff7f0e"}

# --- build figure -----------------------------------------------------
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["By class", "By domain"],
    horizontal_spacing=0.04,
)

_ht = (
    "<b>%{customdata[1]}</b> · %{customdata[2]}<br>"
    "<span style='font-size:10px;color:#666'>%{customdata[3]}</span>"
    "<extra></extra>"
)

# panel 1 — by class
for cls in cls_order:
    mask = [i for i, l in enumerate(labels) if l == cls]
    if not mask:
        continue
    fig.add_trace(go.Scatter(
        x=xy[mask, 0], y=xy[mask, 1],
        mode="markers",
        name=cls,
        marker=dict(color=cls_colors[cls], size=4, opacity=0.7),
        customdata=[[i, labels[i], domains[i], pathlib.Path(all_paths[i]).name]
                    for i in mask],
        hovertemplate=_ht,
        legendgroup=cls,
        showlegend=True,
    ), row=1, col=1)

# panel 2 — by domain
for dom in ["game", "movie"]:
    mask = [i for i, d in enumerate(domains) if d == dom]
    if not mask:
        continue
    fig.add_trace(go.Scatter(
        x=xy[mask, 0], y=xy[mask, 1],
        mode="markers",
        name=dom,
        marker=dict(color=dom_colors[dom], size=4, opacity=0.7),
        customdata=[[i, labels[i], domains[i], pathlib.Path(all_paths[i]).name]
                    for i in mask],
        hovertemplate=_ht,
        legendgroup=dom,
        showlegend=True,
    ), row=1, col=2)

fig.update_layout(
    title=dict(text=f"DINOv2 patch embeddings — {RUN_ID}", font=dict(size=14)),
    width=1400, height=680,
    plot_bgcolor="#f8f8f8",
    paper_bgcolor="#ffffff",
    legend=dict(itemsizing="constant"),
    hoverlabel=dict(bgcolor="white", font_size=12),
)
fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

# --- assemble self-contained HTML -------------------------------------
plot_div  = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="umap-plot")
images_js = json.dumps(click_b64)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DINO UMAP — {RUN_ID}</title>
<style>
  body {{ margin: 0; padding: 10px; font-family: sans-serif; background: #f0f0f0; }}
  #img-panel {{
    position: fixed; bottom: 20px; right: 20px;
    background: #fff; border: 1.5px solid #bbb; border-radius: 10px;
    padding: 10px 12px; box-shadow: 4px 4px 14px rgba(0,0,0,0.25);
    display: none; z-index: 9999; max-width: 310px; text-align: center;
  }}
  #img-panel img   {{ max-width: 256px; border-radius: 4px; margin-top: 6px; }}
  #img-panel .meta {{ font-size: 11px; color: #444; margin-top: 5px; word-break: break-all; }}
  #img-panel button {{
    margin-top: 7px; font-size: 11px; cursor: pointer;
    border: 1px solid #bbb; border-radius: 4px; padding: 2px 10px; background: #f5f5f5;
  }}
</style>
</head>
<body>
{plot_div}
<div id="img-panel">
  <img id="img-display" src="" alt="patch" />
  <div class="meta" id="img-meta"></div>
  <div><button onclick="document.getElementById('img-panel').style.display='none'">close ✕</button></div>
</div>
<script>
const _imgs = {images_js};
document.getElementById("umap-plot").on("plotly_click", function(data) {{
  const cd  = data.points[0].customdata;
  document.getElementById("img-display").src = "data:image/jpeg;base64," + _imgs[cd[0]];
  document.getElementById("img-meta").textContent = cd[1] + " · " + cd[2] + "  |  " + cd[3];
  document.getElementById("img-panel").style.display = "block";
}});
</script>
</body>
</html>"""

out_path = FIGURES_DIR / f"dino_umap_{RUN_ID}_interactive.html"
out_path.write_text(html)
print(f"Saved: {out_path}")

# %%
