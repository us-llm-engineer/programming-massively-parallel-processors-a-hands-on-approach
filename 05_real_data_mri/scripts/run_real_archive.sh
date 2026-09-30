#!/usr/bin/env bash
# Reconstruct every slice of every COMPLETE scan found in an M4Raw archive (possibly a truncated download), with several worker
# processes sharing the GPU so it stays busy while other workers extract, convert and score.
# Usage: [PAUSE=0.75] scripts/run_real_archive.sh <archive.zip> <workers> <out-dir> [max-scans]   (PAUSE = duty-cycle limit, see 02_real_archive.cu)
#   out-dir gets: png/<scan>.png (contact sheet of every reconstructed slice), png/showcase/, metrics.csv (one row per slice x method), scans.csv (per scan: status, wall seconds), gpu_timeline.csv (nvidia-smi, 1 Hz)
set -u
zip="$1"; W="${2:-3}"; out="$3"; max="${4:-100000}"
mkdir -p "$out/scans" "$out/tmp" bin; [ -x bin/02_real_archive ] || make bin/02_real_archive || exit 1
python3 scripts/zip_stream.py list "$zip" | head -n "$max" > "$out/list.txt"
echo "scan,status,wall_s,start_epoch" > "$out/scans.csv"
one() {                                                         # one scan, run by xargs
  name="$1"; zip="$2"; out="$3"; base=$(basename "$name" .h5); t="$out/tmp/$base"; mkdir -p "$t"; t0=$(date +%s%N)
  if python3 scripts/zip_stream.py get "$zip" "$name" "$t/scan.h5" && python3 scripts/prepare_m4raw.py "$t/scan.h5" "$t" > /dev/null \
     && ./bin/02_real_archive "$t/scan.bin" "$base" "$out/scans/$base.csv" "${PAUSE:-0}" "$t" > /dev/null \
     && python3 scripts/make_sheet.py "$t" "$base" "$out/png"; then st=PASS; else st=FAIL; fi
  rm -rf "$t"; t1=$(date +%s%N); echo "$base,$st,$(( (t1 - t0) / 1000000 ))e-3,$(( t0 / 1000000000 ))" >> "$out/scans.csv"
}
export -f one; export PAUSE
nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,power.draw,temperature.gpu,memory.used --format=csv -l 1 > "$out/gpu_timeline.csv" & smi=$!
s0=$(date +%s%N)
xargs -a "$out/list.txt" -P "$W" -I{} bash -c 'one "$@"' _ {} "$zip" "$out"
echo "wall_s,$(( ($(date +%s%N) - s0) / 1000000 ))e-3,workers,$W" > "$out/wall.csv"
kill $smi 2>/dev/null
head -1 "$(ls "$out"/scans/*.csv | head -1)" > "$out/metrics.csv"; for f in "$out"/scans/*.csv; do tail -n +2 "$f" >> "$out/metrics.csv"; done
echo "scans: $(grep -c PASS "$out/scans.csv") PASS of $(wc -l < "$out/list.txt"); slices: $(( $(wc -l < "$out/metrics.csv") / 2 )); wall $(cat "$out/wall.csv")"
