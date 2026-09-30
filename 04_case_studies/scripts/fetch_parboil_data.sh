#!/usr/bin/env bash
# Download the two Parboil mri-q input files used by 08d_mri_q_parboil and verify them.
# The files are NOT redistributed with this repository (their licence is not stated on the pages consulted); this script fetches
# them from a public mirror of the Parboil benchmark suite and checks size and SHA-256.
# Usage: scripts/fetch_parboil_data.sh [target-dir]        (default: data/parboil_mri_q)
set -euo pipefail
dest="${1:-data/parboil_mri_q}"
base="https://raw.githubusercontent.com/gtcasl/gpuocelot/master/tests/parboil/benchmarks/mri-q/input"
declare -A size=( ["small/32_32_32_dataset.bin"]=454664 ["large/64_64_64_dataset.bin"]=3186696 )
declare -A sha=( ["small/32_32_32_dataset.bin"]=c3d4e1a79b1c51570d36d44c366b36b67500008dbe78aa01578383f2eb26d961
                 ["large/64_64_64_dataset.bin"]=98b06021df3ee9bfb211b9809f65914f45f959e301baf8fe1b38305a0d8af58e )
for f in "${!size[@]}"; do
  mkdir -p "$dest/$(dirname "$f")"
  [ -f "$dest/$f" ] && [ "$(stat -c %s "$dest/$f")" = "${size[$f]}" ] || curl -fL --retry 3 -o "$dest/$f" "$base/$f"
  got=$(stat -c %s "$dest/$f"); h=$(sha256sum "$dest/$f" | cut -d' ' -f1)
  if [ "$got" != "${size[$f]}" ] || [ "$h" != "${sha[$f]}" ]; then echo "MISMATCH for $f (size $got, sha256 $h)" >&2; exit 1; fi
  echo "ok  $f  ${size[$f]} bytes  sha256 ${h:0:16}..."
done
