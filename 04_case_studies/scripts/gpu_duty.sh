#!/usr/bin/env bash
# Measure how busy the GPU is while a program runs (NVML via nvidia-smi, 100 ms samples).
# Usage: scripts/gpu_duty.sh bin/05a_cutoff_atom_centric [more programs...]
# Reports, per program: run time, mean GPU utilisation, share of time with util >= 50 % and <= 5 % (idle), mean SM clock and power.
# Note: nvidia-smi utilisation = share of each sample window in which at least one kernel was running.
tmp=$(mktemp)
printf "%-32s %8s %10s %10s %10s %10s %10s\n" program "run s" "mean util" ">=50% time" "<=5% idle" "SM MHz" "power W"
for prog in "$@"; do
  nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv,noheader,nounits -lms 100 > "$tmp" 2>/dev/null &
  smi=$!
  sleep 0.5
  start=$(date +%s%N)
  "$prog" > /dev/null 2>&1
  end=$(date +%s%N)
  sleep 0.5
  kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
  awk -F', ' -v name="$(basename "$prog")" -v dur_ns=$((end - start)) '
    { n++; u += $1; if ($1 >= 50) hi++; if ($1 <= 5) lo++; c += $2; p += $3 }
    END { if (n) printf "%-32s %8.1f %9.1f%% %9.1f%% %9.1f%% %10.0f %10.1f\n", name, dur_ns / 1e9, u / n, 100 * hi / n, 100 * lo / n, c / n, p / n }' "$tmp"
done
rm -f "$tmp"
