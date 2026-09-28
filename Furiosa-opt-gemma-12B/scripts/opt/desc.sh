#!/bin/bash
# desc.sh <worktree> <ops::fn>  -- compile one kernel with --dump-summary and list every DMA command
# with its engine count, destination (cluster, pe, dmn) and first engine's loop order / packet size
# (via dma_desc.py). This is the only dump that shows how a DMA is split across the 8 engines.
set -u
. "$(dirname "$0")/env.sh"
exec 9>"$OPT_WORK/.build.lock"; flock 9
WT="$1"; K="$2"; TAG=$(basename "$WT"); OUT="$OPT_WORK/sum_${TAG}_${K##*::}"
export FURIOSA_OPT_OUT_DIR="$OPT_WORK/kern_$TAG"
rm -rf "$OUT"; cd "$WT" || exit 1
cargo furiosa-opt compile --exact "$K" --dump-summary "$OUT" > "$OUT.log" 2>&1 || { echo "COMPILE FAIL"; grep -m5 -A4 "^error" "$OUT.log"; exit 1; }
python3 "$(dirname "$0")/dma_desc.py" "$OUT"
echo "(summary dir: $OUT)"
