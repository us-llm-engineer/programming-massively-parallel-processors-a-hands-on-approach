#!/usr/bin/env bash
# Profile selected kernels with Nsight Compute and write one CSV row per kernel launch.
# Usage: [PROFILE_ONLY=1] [METRICS=a,b,c] scripts/profile_kernels.sh <binary> <kernel-name-regex> <output.csv>
#   e.g. PROFILE_ONLY=1 scripts/profile_kernels.sh bin/08a_mri_fhd_ladder 'regex:fhd_v' stats/08a_ncu.csv
# Needs GPU performance counters enabled for the user (Windows: NVIDIA Control Panel > Developer > Manage GPU Performance Counters).
bin="$1"; kernels="$2"; out="$3"
DEFAULT="gpu__time_duration.sum,smsp__inst_executed.sum,sm__inst_executed_pipe_xu.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,sm__throughput.avg.pct_of_peak_sustained_elapsed"
/usr/local/cuda/bin/ncu --csv --page raw --kernel-name "$kernels" --metrics "${METRICS:-$DEFAULT}" "$bin" 2>/dev/null \
  | grep -v '^==' > "$out"
echo "wrote $out ($(($(wc -l < "$out") - 2)) kernel launches)"
