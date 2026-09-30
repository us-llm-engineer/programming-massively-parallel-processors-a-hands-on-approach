#!/usr/bin/env python3
"""Figures for 09_gpu_starvation from stats/09_*.csv  ->  figures/09_*.png   (run from the batch directory)."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import viz  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
STATS, FIG = os.path.join(ROOT, "stats"), viz.out_dir(ROOT)
sm = pd.read_csv(os.path.join(STATS, "09_summary.csv"))
gantt = pd.read_csv(os.path.join(STATS, "09_gantt.csv"))

A_VARS = ["A1 pageable + sync copies", "A2 pinned + sync copies", "A3 pinned + 2 streams", "A4 pinned + 4 streams",
          "A5 pinned + 4 streams, copy-ahead"]
B_VARS = ["B1 prepare then compute (serial)", "B2 1 producer thread (overlap)", "B3 2 producer threads (overlap)"]


def fig_busy():
    a = sm[sm.scenario == "A"]
    ks = sorted(a.param.unique())
    fig, ax = plt.subplots(figsize=(10, 5.2))
    w = 0.16
    for i, v in enumerate(A_VARS):
        vals = [a[(a.variant == v) & (a.param == k)].gpu_busy_pct.iloc[0] for k in ks]
        bars = ax.bar(np.arange(len(ks)) + (i - 2) * w, vals, w * 0.88, color=viz.color(i), label=v)
        if i in (0, 4):
            viz.bar_labels(ax, bars)
    ax.set_xticks(range(len(ks)), [f"K = {int(k)}" for k in ks])
    ax.set_ylim(0, 122); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("GPU busy (% of wall time)")
    ax.set_xlabel("compute intensity: FMAs per element  (low = transfer-bound, high = compute-bound)")
    ax.set_title("How much of the time is the GPU actually computing?")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="x", visible=False)
    viz.finish(fig, f"{FIG}/09_gpu_busy.png", "256 MB streamed through the GPU. GPU busy = share of wall time a kernel is executing. Median of 5 rounds.")


def fig_gantt():
    variants = ["A2 pinned + sync copies", "A4 pinned + 4 streams", "A5 pinned + 4 streams, copy-ahead"]
    g = gantt[(gantt.scenario == "A") & (gantt.param == 1024)]
    lanes = [("H2D", viz.color(0)), ("KERNEL", viz.color(1)), ("D2H", viz.color(2))]
    fig, axes = plt.subplots(len(variants), 1, figsize=(10, 6.4), sharex=True)
    for ax, v in zip(axes, variants):
        gv = g[g.variant == v]
        for row, (lane, col) in enumerate(lanes):
            s = gv[gv.engine == lane]
            ax.broken_barh([(t0, t1 - t0) for t0, t1 in zip(s.t_start_ms, s.t_end_ms)], (row * 1.2, 0.9), facecolors=col,
                           edgecolors=viz.SURFACE, linewidth=0.6)
        ax.set_yticks([0.45, 1.65, 2.85], [l for l, _ in lanes]); ax.invert_yaxis()
        ax.set_title(v, fontsize=10, loc="left", pad=3)
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("time since the first copy (ms)")
    fig.suptitle("Where the GPU waits: copy and compute timelines (K = 1024 FMAs per element)", x=0.012, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    viz.finish(fig, f"{FIG}/09_pipeline_gantt.png", None)


def fig_host():
    b = sm[sm.scenario == "B"]
    ps = sorted(b.param.unique())
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, v in enumerate(B_VARS):
        s = b[b.variant == v].sort_values("param")
        ax.plot(s.param, s.gpu_busy_pct, marker="o", color=viz.color(i), label=v, markeredgecolor=viz.SURFACE, markeredgewidth=1.5)
    ax.set_xticks(ps); ax.set_ylim(0, 100)
    ax.set_xlabel("extra CPU preparation time per chunk (ms), added to the real fill of an 8 MB chunk")
    ax.set_ylabel("GPU busy (% of wall time)")
    ax.set_title("When the CPU is slow at preparing data")
    ax.legend(loc="upper right")
    viz.finish(fig, f"{FIG}/09_host_prep.png", "Overlapping preparation with GPU work keeps the GPU busier until preparation becomes the slowest step.")


def fig_launch():
    c = sm[sm.scenario == "C"]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    bars = ax.barh(c.variant, c.extra, color=[viz.color(i) for i in range(len(c))], height=0.55)
    ax.invert_yaxis(); ax.set_xlabel("microseconds per tiny kernel (smaller is better)"); ax.grid(axis="y", visible=False)
    for b, v in zip(bars, c.extra):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{v:.1f} us", va="center", fontsize=9, color=viz.INK2)
    ax.set_title("Tiny kernels: the launch costs more than the work")
    viz.finish(fig, f"{FIG}/09_launch_overhead.png", "2000 kernels of 4096 elements x 16 FMAs each.")


def fig_overlap():
    a0 = sm[sm.scenario == "A0"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    vals = a0.overlap_ms_or_pct.clip(lower=0)
    ax.barh(a0.variant, 100, color=viz.GRID, height=0.55)
    bars = ax.barh(a0.variant, vals, color=[viz.color(i) for i in range(len(a0))], height=0.55)
    ax.invert_yaxis(); ax.set_xlim(0, 110); ax.set_xlabel("overlap of the shorter operation (%)"); ax.grid(visible=False)
    for b, v in zip(bars, a0.overlap_ms_or_pct):
        ax.text(102, b.get_y() + b.get_height() / 2, f"{v:.0f} %", va="center", fontsize=9, color=viz.INK2)
    ax.set_title("Does the hardware overlap a copy with compute?")
    viz.finish(fig, f"{FIG}/09_copy_compute_overlap.png", "100 % = fully hidden, 0 % = ran one after the other (64 MB copies, two streams).")


def fig_nvml():
    p = os.path.join(STATS, "09_nvml_timeline.csv")
    if not os.path.exists(p):
        return
    tl, mk = pd.read_csv(p), pd.read_csv(os.path.join(STATS, "09_nvml_marks.csv"))
    t = tl.t_ms / 1000
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, col, label, c, vmax in zip(axes, ["sm_clock_mhz", "power_w", "gpu_util_pct"], ["SM clock (MHz)", "power (W)", "NVML utilisation (%)"],
                                       [viz.color(0), viz.color(1), viz.color(2)], [2000, 40, 100]):
        for i in range(len(mk) - 1):
            if i % 2 == 0:
                ax.axvspan(mk.t_ms[i] / 1000, mk.t_ms[i + 1] / 1000, color=viz.GRID, alpha=0.6, lw=0)
        v = tl[col].where(tl[col] >= 0)
        ax.plot(t, v, color=c, lw=1.6); ax.set_ylabel(label); ax.set_ylim(0, vmax)
    for i in range(len(mk) - 1):
        axes[0].text((mk.t_ms[i] + mk.t_ms[i + 1]) / 2000, 2080, mk.label[i].split(" ")[0], ha="center", fontsize=9, color=viz.INK1)
    axes[-1].set_xlabel("time (s); A1 starved, A4 pipelined, B1 serial prep, B2 overlapped prep, C1 launch+sync, C3 fused")
    fig.suptitle("What the GPU itself reports while starved versus fed", x=0.012, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    viz.finish(fig, f"{FIG}/09_nvml_timeline.png", None)


if __name__ == "__main__":
    fig_busy(); fig_gantt(); fig_host(); fig_launch(); fig_overlap(); fig_nvml()
