# nb_figures.py — standalone figure generation for the paper
# Run cells individually with #%% notation (Spyder / VS Code Jupyter)
# Outputs go to figures/ (git-tracked) and paper/figs/ (copy manually or symlink)

# %%
import re
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# 1.1 — Histogram of detection scores (before / after selection)
#
# Source : output/extracted_humans/20260324-185427/_summary.txt  (before stats)
#          output/extracted_humans/20260324-185427/*.jpg          (after scores)
# Save to: figures/score_histogram.pdf
# ---------------------------------------------------------------------------

EXTRACT_DIR = pathlib.Path("../output/extracted_humans/20260324-185427")
FIGURES_DIR = pathlib.Path("../figures")
FIGURES_DIR.mkdir(exist_ok=True)

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
