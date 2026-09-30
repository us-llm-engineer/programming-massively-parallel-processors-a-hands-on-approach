#!/usr/bin/env python3
"""Figures for the MRI programs (08a-08d) from stats/08*.csv -> figures/08_*.png."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import viz  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
STATS, FIG = os.path.join(ROOT, "stats"), viz.out_dir(ROOT)
rd = lambda n: pd.read_csv(os.path.join(STATS, n))


def fig_ladder():
    d = rd("08a_mri_ladder.csv"); r = d[d.scale == "reduced"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [v.split(" (")[0].replace("constant memory, ", "const, ").replace(" + hardware", "\n+ hardware") for v in r.version]
    bars = ax.bar(range(len(r)), r.gpairs_per_s, color=[viz.color(i) for i in range(len(r))], width=0.6)
    viz.bar_labels(ax, bars, "{:.1f}", 0.5)
    f = d[d.scale == "full"]
    if len(f):
        ax.axhline(float(f.gpairs_per_s.iloc[0]), color=viz.INK2, lw=1, ls="--")
        ax.text(len(r) - 0.5, float(f.gpairs_per_s.iloc[0]) + 1, f"v5 at full scale (128^3 x 284,592): {float(f.gpairs_per_s.iloc[0]):.0f} G pairs/s", ha="right", fontsize=9, color=viz.INK2)
    ax.set_xticks(range(len(r)), labels, fontsize=9); ax.set_ylabel("G voxel-sample pairs per second"); ax.set_title("MRI F^H d: the book's optimisation ladder on this GPU"); ax.grid(axis="x", visible=False)
    viz.finish(fig, f"{FIG}/08_ladder.png", "64^3 voxels x 8,192 samples. Higher is better. The dashed line is the full-scale run of the fastest version.")


def fig_mapping():
    d = rd("08b_mri_mapping.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    names = [n.replace("Option ", "Opt ").split(":")[0] + "\n" + n.split(":")[1].strip() for n in d.option]
    bars = ax.barh(names, d.ms, color=[viz.color(i) for i in range(len(d))], height=0.55); ax.invert_yaxis()
    for b, v, r in zip(bars, d.ms, d.vs_option3):
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f} ms  ({r:.1f}x)", va="center", fontsize=9, color=viz.INK2)
    ax.set_xlabel("time (ms)"); ax.set_xlim(0, d.ms.max() * 1.25); ax.grid(axis="y", visible=False); ax.set_title("Mapping the two loops to threads")
    viz.finish(fig, f"{FIG}/08_thread_mapping.png", "32,768 voxels x 4,096 samples. Option 3 (one thread per voxel, no atomics) is the reference (1.0x).")


def fig_trig():
    d = rd("08c_trig_accuracy.csv"); fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (c, lab) in enumerate([("raw_hw_max_err", "hardware, raw argument"), ("reduced_hw_max_err", "hardware, reduced to one cycle"), ("accurate_max_err", "accurate sinf / cosf")]):
        ax.plot(d.max_angle, d[c], marker="o", color=viz.color(i), label=lab, markeredgecolor=viz.SURFACE, markeredgewidth=1.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("largest |angle| in radians"); ax.set_ylabel("max abs error of sin and cos"); ax.set_title("Hardware trigonometry: accuracy against the argument size")
    ax.legend(loc="upper left")
    viz.finish(fig, f"{FIG}/08_trig_accuracy.png", "1,000,000 random angles per point against a double-precision reference.")


def fig_tuning():
    d = rd("08c_tuning.csv")
    best = d.groupby(["block", "chunk"]).gpairs_per_s.max().unstack()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    im = a1.imshow(best.values, cmap="Blues", aspect="auto", origin="lower")
    a1.set_xticks(range(len(best.columns)), best.columns); a1.set_yticks(range(len(best.index)), best.index)
    for i in range(best.shape[0]):
        for j in range(best.shape[1]):
            a1.text(j, i, f"{best.values[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if best.values[i, j] > best.values.max() * 0.6 else viz.INK1)
    a1.set_xlabel("k-space samples per constant-memory chunk"); a1.set_ylabel("threads per block"); a1.set_title("Best of 4 unroll factors (G pairs/s)"); a1.grid(False)
    for i, u in enumerate(sorted(d.unroll.unique())):
        s = d[(d.unroll == u) & (d.block == 128)].sort_values("chunk")
        a2.plot(s.chunk, s.gpairs_per_s, marker="o", color=viz.color(i), label=f"unroll {u}", markeredgecolor=viz.SURFACE, markeredgewidth=1.5)
    a2.set_xscale("log", base=2); a2.set_xlabel("chunk size (samples)"); a2.set_ylabel("G pairs/s"); a2.set_title("Block 128: chunk size x unroll"); a2.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    viz.finish(fig, f"{FIG}/08_tuning.png", "All 100 combinations of block size, chunk size and unroll factor (64^3 voxels x 8,192 samples).")


def fig_parboil():
    d = rd("08d_mri_q.csv"); fig, ax = plt.subplots(figsize=(10, 4.6))
    variants = list(dict.fromkeys(d.variant)); x = np.arange(2); w = 0.26
    for i, v in enumerate(variants):
        s = d[d.variant == v].set_index("dataset").reindex(["small", "large"])
        bars = ax.bar(x + (i - 1) * w, s.gpairs_per_s, w * 0.92, color=viz.color(i), label=v); viz.bar_labels(ax, bars, "{:.0f}", 1)
    ax.set_xticks(x, ["small: 3,072 samples x 32,768 voxels", "large: 2,048 samples x 262,144 voxels"]); ax.set_ylabel("G pairs/s"); ax.set_ylim(0, d.gpairs_per_s.max() * 1.2)
    ax.set_title("MRI-Q on the Parboil datasets"); ax.legend(loc="upper left", fontsize=8); ax.grid(axis="x", visible=False)
    viz.finish(fig, f"{FIG}/08_parboil_q.png", "Real Parboil mri-q input files. Higher is better; all variants verified against a double-precision reference.")


if __name__ == "__main__":
    for f in (fig_ladder, fig_mapping, fig_trig, fig_tuning, fig_parboil):
        try:
            f()
        except FileNotFoundError as e:
            print("skipped", f.__name__, "-", e)
