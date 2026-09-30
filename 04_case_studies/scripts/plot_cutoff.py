#!/usr/bin/env python3
"""Figures for the cutoff-summation programs (05a-05d, 07) from stats/*.csv -> figures/05_*.png and 07_*.png."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import viz  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
STATS, FIG = os.path.join(ROOT, "stats"), viz.out_dir(ROOT)
rd = lambda name: pd.read_csv(os.path.join(STATS, name))


def fig_scaling():
    a, b, c = rd("05a_cutoff.csv"), rd("05b_cutoff.csv"), rd("05c_cutoff.csv")
    series = [
        ("CPU atom-centric (1 core)", a[a.algorithm == "cpu_atom_centric"]),
        ("GPU atom-centric (atomicAdd)", a[a.algorithm == "gpu_atom_centric"]),
        ("GPU DirectSum + cutoff test", a[a.algorithm == "gpu_directsum"]),
        ("GPU LargeBin", b[b.algorithm == "gpu_largebin"]),
        ("GPU SmallBin", c[c.algorithm == "gpu_smallbin"]),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, (name, s) in enumerate(series):
        ax.plot(s.volume_A3, s.ms, marker="o", color=viz.color(i), label=name, markeredgecolor=viz.SURFACE, markeredgewidth=1.5)
    v = np.array([1e3, 5e6]); anchor = a[a.algorithm == "gpu_directsum"]
    if len(anchor):
        v0, t0 = anchor.volume_A3.iloc[0], anchor.ms.iloc[0]
        ax.plot(v, t0 * (v / v0), color=viz.INK2, lw=0.8, ls=":"); ax.plot(v, t0 * (v / v0) ** 2, color=viz.INK2, lw=0.8, ls=":")
        ax.text(v[1] * 0.9, t0 * (v[1] / v0) * 0.6, "linear (O(V))", fontsize=8, color=viz.INK2, ha="right")
        ax.text(v[1] * 0.9, t0 * (v[1] / v0) ** 2 * 0.0004, "quadratic (O(V^2))", fontsize=8, color=viz.INK2, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(7e2, 8e6); ax.set_ylim(0.05, 2e5)
    ax.set_xlabel("simulated volume (cubic Angstrom)"); ax.set_ylabel("time (ms)")
    ax.set_title("Cutoff summation: how each version scales with volume")
    ax.legend(loc="upper left")
    viz.finish(fig, f"{FIG}/05_scaling.png", "Atom density fixed, so atoms grow with volume. Dotted lines: slope 1 (linear) and 2 (quadratic). GPU rows include the host work each algorithm needs.")


def fig_binsweep():
    c = rd("05c_cutoff.csv"); s = c[c.algorithm == "gpu_smallbin_binsweep"].sort_values("bin_edge_A")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(s)); w = 0.55
    parts = [("host binning", s.host_bin_ms), ("bin upload", s.h2d_ms), ("GPU kernel", s.kernel_ms), ("CPU overflow pass", s.overflow_cpu_ms)]
    bottom = np.zeros(len(s))
    for i, (name, vals) in enumerate(parts):
        ax.bar(x, vals, w, bottom=bottom, color=viz.color(i), label=name, edgecolor=viz.SURFACE, linewidth=1.5); bottom += vals.values
    ax.set_xticks(x, [f"{e:.0f} A\n{100 * o / n:.1f} % overflow" for e, o, n in zip(s.bin_edge_A, s.overflow_atoms, s.atoms)])
    ax.set_yscale("log"); ax.set_ylabel("time (ms, log scale)"); ax.set_xlabel("bin edge (capacity fixed at 8 atoms per bin)")
    ax.set_title("SmallBin: the bin size decides how much lands on the CPU"); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    viz.finish(fig, f"{FIG}/05_smallbin_binsweep.png", "L = 40 A. Small bins waste GPU time on nearly empty bins; large bins overflow atoms to the (slower) CPU.")


def fig_overlap():
    o = rd("05d_overlap.csv"); e = o[o.section == "edge_sweep"].sort_values("bin_edge_A")
    fig, ax = plt.subplots(figsize=(10, 4.8)); x = np.arange(len(e)); w = 0.26
    ax.bar(x - w, e.serial_ms, w, color=viz.color(0), label="serial (GPU then CPU overflow)")
    ax.bar(x, e.overlap_ms, w, color=viz.color(1), label="overlapped (CPU under GPU slabs)")
    ax.bar(x + w, e.ideal_ms, w, color=viz.color(2), label="ideal = max(GPU alone, CPU alone)")
    ax.set_xticks(x, [f"{b:g} A\n{p:.1f} % ovfl" for b, p in zip(e.bin_edge_A, e.overflow_pct)])
    ax.set_yscale("log"); ax.set_ylabel("compute phase (ms, log scale)"); ax.set_xlabel("bin edge (more overflow to the right)")
    ax.set_title("SmallBin-Overlap: hiding the CPU overflow pass behind GPU work"); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    viz.finish(fig, f"{FIG}/05_overlap.png", "L = 40 A, 8 slabs. Overlap can hide at most the shorter of the two pieces.")
    gp = os.path.join(STATS, "05d_overlap_gantt.csv")
    if os.path.exists(gp):
        g = pd.read_csv(gp); fig, ax = plt.subplots(figsize=(10, 2.8))
        for row, (eng, col) in enumerate([("GPU_kernel", viz.color(0)), ("CPU_overflow", viz.color(1))]):
            s = g[g.engine == eng]
            ax.broken_barh([(a, b - a) for a, b in zip(s.t_start_ms, s.t_end_ms)], (row * 1.2, 0.9), facecolors=col, edgecolors=viz.SURFACE, linewidth=1)
        ax.set_yticks([0.45, 1.65], ["GPU kernel (slab i)", "CPU overflow (slab i)"]); ax.invert_yaxis(); ax.set_xlabel("time (ms)"); ax.grid(axis="y", visible=False)
        ax.set_title("One overlapped run: 8 slabs, CPU work while the GPU executes")
        viz.finish(fig, f"{FIG}/05_overlap_gantt.png", "Bin edge 3.75 A, L = 40 A.")


def fig_divergence():
    p = os.path.join(STATS, "07_divergence.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(d))[::-1]
    a1.barh(y, d.all_in_pct, color=viz.color(2), label="all lanes inside", edgecolor=viz.SURFACE, linewidth=1.2)
    a1.barh(y, d.mixed_pct, left=d.all_in_pct, color=viz.color(1), label="MIXED (divergent)", edgecolor=viz.SURFACE, linewidth=1.2)
    a1.barh(y, d.all_out_pct, left=d.all_in_pct + d.mixed_pct, color=viz.GRID, label="all lanes outside", edgecolor=viz.SURFACE, linewidth=1.2)
    a1.set_yticks(y, [f"{t}\nwarp {w}" for t, w in zip(d.tile, d.warp_footprint)]); a1.set_xlim(0, 100); a1.set_xlabel("share of warp-level distance tests (%)")
    for yy, m in zip(y, d.mixed_pct):
        a1.text(101, yy, f"{m:.1f} %", va="center", fontsize=8, color=viz.INK2)
    a1.set_title("How often a warp disagrees"); a1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=8); a1.grid(axis="y", visible=False)
    x = np.arange(len(d)); w = 0.27
    a2.bar(x - w, d.branchy_ms, w, color=viz.color(0), label="branchy"); a2.bar(x, d.branchless_ms, w, color=viz.color(1), label="branchless")
    a2.bar(x + w, d.notest_ms, w, color=viz.color(2), label="no test (bound)")
    a2.set_xticks(x, d.tile, rotation=20); a2.set_ylabel("kernel time (ms)"); a2.set_title("What it costs"); a2.legend(fontsize=8); a2.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    viz.finish(fig, f"{FIG}/07_divergence.png", "Same 256 lattice points per block, arranged in different tile shapes. A warp is 32 consecutive threads (x fastest).")


if __name__ == "__main__":
    for f in (fig_scaling, fig_binsweep, fig_overlap, fig_divergence):
        try:
            f()
        except FileNotFoundError as e:
            print("skipped", f.__name__, "-", e)
