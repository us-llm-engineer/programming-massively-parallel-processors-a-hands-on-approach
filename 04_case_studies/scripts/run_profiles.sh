#!/usr/bin/env bash
# Collect the Nsight Compute measurements used in the write-ups: constant-cache hit rates (06) and instruction/pipe usage of the MRI ladder (08a).
# Needs GPU performance counters enabled (Windows: NVIDIA Control Panel > Developer > Manage GPU Performance Counters). Run from the batch directory.
set -e
cd "$(dirname "$0")/.."
PROFILE_ONLY=1 METRICS=gpu__time_duration.sum,idc__request_hit_rate,gcc__cache_requests_type_constant_lookup_hit.sum,gcc__cache_requests_type_constant_lookup_miss.sum,sm__idc_divergent_instruction_replays.sum,l1tex__t_sector_hit_rate.pct \
  scripts/profile_kernels.sh bin/06_constant_cache_thrash 'regex:window_kernel|lane_kernel|layout_kernel' stats/06_ncu.csv
python3 scripts/summarize_ncu.py stats/06_ncu.csv \
  --cols "gpu__time_duration.sum,idc__request_hit_rate.pct,gcc__cache_requests_type_constant_lookup_hit.sum,gcc__cache_requests_type_constant_lookup_miss.sum,sm__idc_divergent_instruction_replays.sum,l1tex__t_sector_hit_rate.pct" \
  --labels "E1 const W512 same,E1 const W512 different,E1 global+shared W512 same,E1 global+shared W512 different,E1 const W64 same,E1 const W64 different,E2 const 1 addr/warp,E2 const 32 addr/warp,E3 3 arrays,E3 struct12,E3 struct16" \
  > output/06_ncu_summary.txt
PROFILE_ONLY=1 scripts/profile_kernels.sh bin/08a_mri_fhd_ladder 'regex:fhd_v' stats/08a_ncu.csv
python3 scripts/summarize_ncu.py stats/08a_ncu.csv --group-by-kernel \
  --cols "gpu__time_duration.sum,smsp__inst_executed.sum,sm__inst_executed_pipe_xu.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct" \
  > output/08a_ncu_summary.txt
echo "profiles written: output/06_ncu_summary.txt output/08a_ncu_summary.txt"
