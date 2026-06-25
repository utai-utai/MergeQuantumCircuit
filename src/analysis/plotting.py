"""
Shared figure styling for the paper.  `set_house_style()` applies the
PRX-Quantum-tuned Computer-Modern serif look used by every figure; `PALETTE`
holds the consistent colour assignments; `savefig()` writes the 400-dpi raster
plus the vector PDF that LaTeX ingests.
"""
import os

from src import RESULT_DIR

# Consistent colours across all figures.
PALETTE = {
    "method": "#1f3b73",     # subspace / method (deep blue)
    "global": "#c0392b",     # global / plain PQC (red)
    "truncation": "#2a8c4a",
    "reference": "#7f8c8d",
    "floor": "#1e8449",
}


def set_house_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.minor.size": 3, "ytick.minor.size": 3,
        "font.size": 12,
    })
    return plt


def savefig(fig, stem, dpi=400):
    """Save <stem>.png into result/."""
    png = os.path.join(RESULT_DIR, stem + ".png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    print("saved ->", os.path.relpath(png, os.path.dirname(RESULT_DIR)))
    return png
