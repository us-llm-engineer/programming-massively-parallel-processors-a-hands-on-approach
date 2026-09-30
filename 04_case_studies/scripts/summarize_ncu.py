#!/usr/bin/env python3
"""Summarise an Nsight Compute raw CSV (from scripts/profile_kernels.sh) as a table.

usage: summarize_ncu.py <ncu.csv> [--labels a,b,c ...] [--group-by-kernel]
  --labels          names for the launches, in launch order (one per row)
  --group-by-kernel average all launches of the same kernel (for chunked launches) instead of one row per launch
Numeric columns are printed as-is; gpu__time_duration (ns) is shown in ms.
"""
import argparse, csv, io, collections

ap = argparse.ArgumentParser()
ap.add_argument("csv_path"); ap.add_argument("--labels", default=""); ap.add_argument("--group-by-kernel", action="store_true")
ap.add_argument("--cols", default="", help="comma-separated exact metric columns to show (raw pages carry .avg/.min/.max/.sum roll-ups)")
a = ap.parse_args()
lines = open(a.csv_path).read().splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
rows = [r for r in csv.DictReader(io.StringIO("\n".join(lines[start:]))) if r.get("ID", "").isdigit()]
skip = {"ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream", "Block Size", "Grid Size", "Device", "CC", "Section Name", "Metric Name", "Metric Unit"}
KEEP = ("gpu__", "smsp__", "sm__", "l1tex__", "lts__", "idc__", "gcc__", "dram__")     # profiler metrics only (the raw page also has device attributes)
metrics = [c for c in rows[0].keys() if c not in skip and c.startswith(KEEP)]
if a.cols:
    metrics = [c for c in a.cols.split(",") if c in rows[0]]
num = lambda s: float(s.replace(",", "")) if s.replace(",", "").replace(".", "", 1).replace("e", "", 1).replace("-", "", 1).isdigit() else 0.0
labels = a.labels.split(",") if a.labels else []
short = lambda m: m.replace("gpu__time_duration.sum", "time ms").replace("smsp__", "").replace("sm__", "").replace("l1tex__", "").replace("lts__", "").replace("idc__", "idc.").replace("gcc__cache_requests_type_", "l1.5.")[:26]
if a.group_by_kernel:
    groups = collections.OrderedDict()
    for r in rows: groups.setdefault(r["Kernel Name"].split("(")[0], []).append(r)
    items = [(k, [sum(num(r[m]) for r in v) / len(v) if "pct" in m or "hit_rate" in m else sum(num(r[m]) for r in v) for m in metrics], len(v)) for k, v in groups.items()]
else:
    items = [((labels[i] if i < len(labels) else r["Kernel Name"].split("(")[0]), [num(r[m]) for m in metrics], 1) for i, r in enumerate(rows)]
print(f"{'case':34s} " + " ".join(f"{short(m):>26s}" for m in metrics))
for name, vals, n in items:
    print(f"{name[:34]:34s} " + " ".join(f"{(v / 1e6 if 'time' in m else v):26.2f}" for m, v in zip(metrics, vals)))
