#!/usr/bin/env python3
"""Figures for 01_dcs_compute and 04_ch9_pitfalls from stats/01_dcs.csv and stats/04_ch9_pitfalls.csv."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import viz  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
STATS, FIG = os.path.join(ROOT, "stats"), viz.out_dir(ROOT)


def fig_dcs():
    d = pd.read_csv(os.path.join(STATS, "01_dcs.csv"))
    fig, ax = plt.subplots(figsize=(10, 4.8)); w = 0.36
    for i, scale in enumerate(["reduced", "full"]):
        s = d[d.scale == scale]
        bars = ax.bar(np.arange(len(s)) + (i - 0.5) * w, s.g_evals_per_s, w * 0.92, color=viz.color(i), label={"reduced": "reduced: 100x100x10 lattice, 4,000 atoms", "full": "full: 100^3 lattice, 100,000 atoms"}[scale])
        viz.bar_labels(ax, bars, "{:.0f}", 1)
    ax.set_xticks(range(3), ["v1  1 point/thread", "v2  4 points/thread", "v3  8 points/thread, coalesced"]); ax.set_ylabel("G atom-point evaluations per second")
    ax.set_title("Direct Coulomb summation: the book's three kernels on this GPU"); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    viz.finish(fig, f"{FIG}/01_dcs_versions.png", "Higher is better. The book's G80 numbers were 18.6 / 33.4 / 39.5 G evals/s (v3 was its fastest; here v2 wins because one slice launches too few blocks for v3).")


def fig_pitfalls():
    d = pd.read_csv(os.path.join(STATS, "04_ch9_pitfalls.csv"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    a = d[d.section == "A_crossover_ms"]
    for i, s in enumerate(["cpu", "gpu_v1", "gpu_v2", "gpu_v3"]):
        x = a[a.series == s]; a1.plot(x.param, x.value, marker="o", color=viz.color(i), label={"cpu": "CPU (1 core)", "gpu_v1": "GPU v1", "gpu_v2": "GPU v2", "gpu_v3": "GPU v3"}[s], markeredgecolor=viz.SURFACE, markeredgewidth=1.5)
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlabel("number of atoms (one 100x100 slice)"); a1.set_ylabel("time (ms)"); a1.set_title("When does the GPU win?"); a1.legend(fontsize=8)
    e = d[d.section == "E_chunk_size_ms"]
    bars = a2.bar([str(int(p)) for p in e.param], e.value, color=viz.color(0), width=0.6); viz.bar_labels(a2, bars, "{:.0f}", 2)
    a2.set_xlabel("atoms per constant-memory chunk (64 KB = 4000)"); a2.set_ylabel("time for 40,000 atoms x 10 slices (ms)"); a2.set_title("The 64 KB limit: smaller chunks, more launches"); a2.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    viz.finish(fig, f"{FIG}/04_ch9_pitfalls.png", "Left: warm GPU vs single-core CPU. Right: the same arithmetic split into more and smaller chunks.")


if __name__ == "__main__":
    for f in (fig_dcs, fig_pitfalls):
        try:
            f()
        except FileNotFoundError as e:
            print("skipped", f.__name__, "-", e)
