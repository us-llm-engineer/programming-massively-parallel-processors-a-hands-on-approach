#!/usr/bin/env python3
"""Figures for 06_constant_cache_thrash: timing curves from stats/06_constant_cache.csv and hit rates from the Nsight Compute CSV."""
import csv
import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import viz  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
STATS, FIG = os.path.join(ROOT, "stats"), viz.out_dir(ROOT)


def fig_windows():
    d = pd.read_csv(os.path.join(STATS, "06_constant_cache.csv")); e1 = d[d.experiment == "E1"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (v, lab) in enumerate([("constant_different", "constant memory"), ("global_shared_different", "global memory + shared tiles")]):
        s = e1[e1.variant == v]
        ax.plot(s.param, s.ratio, marker="o", color=viz.color(i), label=lab, markeredgecolor=viz.SURFACE, markeredgewidth=1.5)
    ax.axhline(1.0, color=viz.INK2, lw=0.8, ls=":")
    ax.set_xscale("log", base=2); ax.set_xlabel("atoms read per block (W)"); ax.set_ylabel("time: different window per block / same window")
    ax.set_title("Constant cache: blocks reading different atoms vs the same atoms"); ax.legend(loc="upper left")
    viz.finish(fig, f"{FIG}/06_window_ratio.png", "Identical arithmetic; only the atoms each block reads differ. At W = 4000 both cases read the whole array, so the ratio returns to 1.")


def fig_lanes_layout():
    d = pd.read_csv(os.path.join(STATS, "06_constant_cache.csv"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    for i, (v, lab) in enumerate([("constant", "constant memory"), ("global", "global memory")]):
        s = d[(d.experiment == "E2") & (d.variant == v)]
        a1.plot(s.param, s.ratio, marker="o", color=viz.color(i), label=lab, markeredgecolor=viz.SURFACE, markeredgewidth=1.5)
    a1.set_xscale("log", base=2); a1.set_yscale("log"); a1.set_xlabel("distinct addresses read by the 32 lanes of a warp"); a1.set_ylabel("time relative to one address")
    a1.set_title("Warp address uniformity"); a1.legend(loc="upper left")
    s = d[d.experiment == "E3"]
    bars = a2.bar(["3 arrays", "12-byte struct", "16-byte struct"], s.ratio, color=[viz.color(0), viz.color(2), viz.color(3)], width=0.55)
    viz.bar_labels(a2, bars, "{:.2f}", 0.01); a2.set_ylim(0, 1.25); a2.set_ylabel("time relative to 3 arrays"); a2.set_title("Layout in constant memory"); a2.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    viz.finish(fig, f"{FIG}/06_lanes_layout.png", "Left: constant memory serves different addresses of one warp one after another. Right: the struct layout is fastest here.")


def read_ncu(path):
    lines = open(path).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    rows = [r for r in csv.DictReader(io.StringIO("\n".join(lines[start:]))) if r.get("ID", "").isdigit()]
    return rows


def fig_hitrates():
    p = os.path.join(STATS, "06_ncu.csv")
    if not os.path.exists(p):
        return
    rows = read_ncu(p)
    labels = ["const W=512\nsame window", "const W=512\ndifferent", "const W=64\nsame window", "const W=64\ndifferent"]
    pick = [0, 1, 4, 5]
    f = lambda r, k: float(r[k].replace(",", ""))
    idc = [f(rows[i], "idc__request_hit_rate.pct") for i in pick]
    miss = [f(rows[i], "gcc__cache_requests_type_constant_lookup_miss.sum") for i in pick]
    ms = [f(rows[i], "gpu__time_duration.sum") / 1e6 for i in pick]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    b = a1.bar(labels, idc, color=[viz.color(0), viz.color(1), viz.color(0), viz.color(1)], width=0.6); viz.bar_labels(a1, b, "{:.1f} %", 1)
    a1.set_ylim(0, 112); a1.set_ylabel("constant cache (L1) hit rate (%)"); a1.set_title("Hit rate (Nsight Compute)"); a1.grid(axis="x", visible=False); a1.tick_params(axis="x", labelsize=8)
    b = a2.bar(labels, ms, color=[viz.color(0), viz.color(1), viz.color(0), viz.color(1)], width=0.6); viz.bar_labels(a2, b, "{:.2f}", 0.03)
    a2.set_ylabel("kernel time (ms)"); a2.set_title("Time for the same four kernels"); a2.grid(axis="x", visible=False); a2.tick_params(axis="x", labelsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    viz.finish(fig, f"{FIG}/06_hit_rates.png", "Same arithmetic in every kernel. Blue: all blocks read the same window; orange: each block reads a different one.")


if __name__ == "__main__":
    for f in (fig_windows, fig_lanes_layout, fig_hitrates):
        try:
            f()
        except FileNotFoundError as e:
            print("skipped", f.__name__, "-", e)
