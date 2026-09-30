"""Shared plotting style for the benchmark figures (matplotlib).

Colours are the validated categorical slots (blue, orange, aqua, yellow, magenta, violet, green, red) on a light surface;
text uses ink tokens, never series colours; gridlines are recessive; marks are thin with a 2 px surface gap.
The CUDA programs write CSV under stats/; the plot_*.py scripts read those files and write PNGs under figures/.
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK1, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2dd"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK1, "axes.titlecolor": INK1, "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2.0, "lines.markersize": 6, "figure.dpi": 100, "savefig.dpi": 150,
    "font.family": ["DejaVu Sans"],
})


def color(i):
    return SERIES[i % len(SERIES)]


def out_dir(root):
    d = os.path.join(root, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def finish(fig, path, subtitle=None):
    """Save and report. A subtitle (one sentence saying how to read the chart) sits under the title."""
    if subtitle:                       # the "how to read this chart" line sits under the chart
        fig.text(0.012, -0.035, subtitle, fontsize=9, color=INK2, ha="left", va="top", transform=fig.transFigure)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)


def bar_labels(ax, bars, fmt="{:.0f}", pad=2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + pad, fmt.format(h), ha="center", va="bottom", fontsize=8, color=INK2)
