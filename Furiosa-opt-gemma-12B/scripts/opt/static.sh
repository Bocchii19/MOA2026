#!/bin/bash
# static.sh <worktree> [ops::fn ...]  -- static makespan / DMA / Main from --dump-schedule
set -u
. "$(dirname "$0")/env.sh"
# One cargo at a time in the shared target dir: build.sh drops fingerprints, which must not race
# another build (it once deleted a fingerprint dir mid-write).
exec 9>"$OPT_WORK/.build.lock"; flock 9
WT="$1"; shift
KS=("$@"); [ ${#KS[@]} -eq 0 ] && KS=(ops::sliding_project_qkv ops::sliding_attention_output ops::decoder_feedforward)
TAG=$(basename "$WT")
cd "$WT" || exit 1
export FURIOSA_OPT_OUT_DIR="$OPT_WORK/kern_$TAG"
for K in "${KS[@]}"; do
  OUT="$OPT_WORK/sched_${TAG}_${K##*::}.json"
  if ! cargo furiosa-opt compile --exact "$K" --dump-schedule "$OUT" > "$OPT_WORK/static_$TAG.log" 2>&1; then
    echo "$K: COMPILE FAIL"; grep -m3 -A3 "^error" "$OPT_WORK/static_$TAG.log"; continue
  fi
  python3 "$(dirname "$0")/sched.py" --summary "$OUT" "$K"
done
