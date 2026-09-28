#!/bin/bash
# sweep.sh <wt> <test> <runs> <tag>  -- one Arena job running <runs> seeded runs of one kernel; prints the
# numeric verdict per run (max|d|, PASS/FAIL) and the cycle median. For correctness robustness checks.
set -u
. "$(dirname "$0")/env.sh"
WT="$1"; TEST="$2"; RUNS="$3"; TAG="$4"; D="$OPT_WORK/sweep_$TAG"; rm -rf "$D"; mkdir -p "$D"
python3 "$(dirname "$0")/harness_overlay.py" "$WT" --runs "$RUNS" >/dev/null || exit 1
"$(dirname "$0")/build.sh" "$WT" "$D/bin" || exit 1
cp -f "$WT/ref/fixtures.safetensors" "$D/fixtures.safetensors"
cat > "$D/entry.sh" <<EOS
#!/bin/sh
cd "\$(dirname "\$0")" || exit 1
chmod +x ./bin 2>/dev/null
GEMMA4_ONLY="$TEST" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info ./bin; echo "@@rc=\$?@@"
EOS
chmod +x "$D/entry.sh"
while :; do
  OUT=$(furiosa-arena submit "$D/entry.sh" "$D/bin" "$D/fixtures.safetensors" --name "sweep-$TAG" --entrypoint entry.sh --timeout 70 2>&1)
  echo "$OUT" | grep -q "quota exceeded" || break; sleep 30
done
JOB=$(echo "$OUT" | grep -oP 'submitted job \K[0-9]+'); [ -z "$JOB" ] && { echo "$OUT"; exit 1; }
echo "job $JOB (sweep-$TAG)"
while ! furiosa-arena list 2>/dev/null | grep -E "^ *$JOB " | grep -qE "SUCCEEDED|FAILED"; do sleep 15; done
furiosa-arena logs "$JOB" > "$D/log.txt" 2>&1
echo "runs: $(grep -c "\] max|" "$D/log.txt")  PASS: $(grep -c -- "-> PASS" "$D/log.txt")  FAIL: $(grep -c "FAIL" "$D/log.txt")  $(grep -o "@@rc=[0-9]*@@" "$D/log.txt")"
grep -o "max|Δ|= *[0-9.]*" "$D/log.txt" | awk '{print $NF}' | sort -n | tail -3 | tr '\n' ' '; echo "<- three largest max|d|"
grep "median cycles" "$D/log.txt" | head -2
