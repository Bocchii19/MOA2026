#!/bin/bash
# measure.sh <worktree> <tag> [--spans]  -- build with the measurement harness, run on Arena,
# print median / min / p25 and the score each implies. Logs land in $OPT_WORK/log_<tag>.txt.
set -u
. "$(dirname "$0")/env.sh"
WT="$1"; TAG="$2"; SPANS="${3:-}"
python3 "$(dirname "$0")/harness_overlay.py" "$WT" --runs 9 $SPANS >/dev/null || exit 1
"$(dirname "$0")/build.sh" "$WT" "$OPT_WORK/bin_$TAG" || exit 1
D="$OPT_WORK/stage_$TAG"; mkdir -p "$D"
cp -f "$OPT_WORK/bin_$TAG" "$D/test_runtime"
cp -f "$WT/ref/fixtures.safetensors" "$D/fixtures.safetensors"
ENV_PREFIX='FURIOSA_OPT_PROFILE="${FURIOSA_OPT_PROFILE:-info}"'
[ "$SPANS" = "--spans" ] && ENV_PREFIX='GEMMA4_DUMP_SPANS=1 FURIOSA_OPT_PROFILE="${FURIOSA_OPT_PROFILE:-trace}"'
sed 's/\r$//' "$WT/scripts/rngd/remote_entrypoint.sh" \
  | sed "s|FURIOSA_OPT_PROFILE=\"\${FURIOSA_OPT_PROFILE:-info}\"|$ENV_PREFIX|" > "$D/remote_entrypoint.sh"
chmod +x "$D/remote_entrypoint.sh"
# Arena allows 2 active jobs per user: retry while the quota is full.
while :; do
  OUT=$(furiosa-arena submit "$D/remote_entrypoint.sh" "$D/test_runtime" "$D/fixtures.safetensors" \
        --name "$TAG" --entrypoint remote_entrypoint.sh --timeout 70 2>&1)
  echo "$OUT" | grep -q "quota exceeded" || break; sleep 30
done
JOB=$(echo "$OUT" | grep -oP 'submitted job \K[0-9]+')
[ -z "$JOB" ] && { echo "$OUT"; exit 1; }
echo "job $JOB ($TAG)"
while ! furiosa-arena list 2>/dev/null | grep -E "^ *$JOB " | grep -qE "SUCCEEDED|FAILED"; do sleep 15; done
furiosa-arena logs "$JOB" > "$OPT_WORK/log_$TAG.txt" 2>&1
grep -E "FAIL|panick" "$OPT_WORK/log_$TAG.txt" | head -3
python3 - "$OPT_WORK/log_$TAG.txt" "$TAG" <<'PY'
import sys, re
t = open(sys.argv[1]).read()
rows = re.findall(r"median cycles=(\d+) min=(\d+) p25=(\d+)", t)
if len(rows) != 3: print("parse failed", len(rows)); sys.exit(1)
for k, lbl in enumerate(("median", "min", "p25")):
    v = [int(r[k]) for r in rows]; p = v[0] * v[1] * v[2]
    print(f"{sys.argv[2]:18} {lbl:6} K1={v[0]:7} K2={v[1]:6} K3={v[2]:7}  score={(3.754e17 / p) ** (1 / 3):.4f}")
PY
