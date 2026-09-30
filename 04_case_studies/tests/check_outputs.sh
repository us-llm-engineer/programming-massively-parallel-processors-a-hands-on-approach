#!/usr/bin/env bash
# Scan the result files in output/: every program result must end with "ALL VARIANTS PASS" and contain no FAIL marker.
cd "$(dirname "$0")/.."; bad=0; n=0
for f in output/[0-9]*.txt; do
  case "$f" in *ncu_summary*|*02_gpu_monitor*) continue;; esac          # profiler summary and the monitor report have no PASS line
  n=$((n+1))
  if ! grep -q "ALL VARIANTS PASS" "$f"; then echo "NOT PASSING: $f"; bad=1; fi
  if grep -q "FAIL" "$f"; then echo "FAIL marker in: $f"; bad=1; fi
done
[ "$bad" = 0 ] && echo "output check: all $n result files report ALL VARIANTS PASS" || exit 1
