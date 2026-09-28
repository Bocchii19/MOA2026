#!/bin/bash
# ab.sh <wtA> <wtB> <tag> [rounds]  -- A and B alternate inside ONE Arena job (A B A B ...), so
# (env RUNS=n seeded runs per invocation, default 7; ONLY=<test name> runs just that kernel, which
#  allows many more runs within the 70 s job limit, e.g. ONLY=sliding_attention_output RUNS=21)
# machine drift between jobs cannot masquerade as an effect. Each invocation runs 7 seeded runs per
# kernel; the report is the median over all of an arm's runs, per kernel, and the B-A delta.
set -u
. "$(dirname "$0")/env.sh"
WTA="$1"; WTB="$2"; TAG="$3"; ROUNDS="${4:-2}"
D="$OPT_WORK/ab_$TAG"; rm -rf "$D"; mkdir -p "$D"
for arm in A B; do
  WT=$([ $arm = A ] && echo "$WTA" || echo "$WTB")
  python3 "$(dirname "$0")/harness_overlay.py" "$WT" --runs "${RUNS:-7}" >/dev/null || exit 1
  "$(dirname "$0")/build.sh" "$WT" "$D/bin_$arm" || exit 1
done
[ "$(md5sum < "$D/bin_A")" = "$(md5sum < "$D/bin_B")" ] && echo "WARNING: arms are byte-identical"
cp -f "$WTA/ref/fixtures.safetensors" "$D/fixtures.safetensors"
cat > "$D/entry.sh" <<EOF
#!/bin/sh
cd "\$(dirname "\$0")" || exit 1
chmod +x ./bin_A ./bin_B 2>/dev/null
for r in \$(seq 1 $ROUNDS); do
  for arm in A B; do
    echo "@@AB_BEGIN \$r \$arm@@"
    GEMMA4_ONLY="${ONLY:-}" GEMMA4_PROFILE_SETTLE_MS=200 FURIOSA_OPT_PROFILE=info ./bin_\$arm
    echo "@@AB_END \$r \$arm rc=\$?@@"
  done
done
EOF
chmod +x "$D/entry.sh"
# Arena allows 2 active jobs per user: retry while the quota is full.
while :; do
  OUT=$(furiosa-arena submit "$D/entry.sh" "$D/bin_A" "$D/bin_B" "$D/fixtures.safetensors" \
        --name "ab-$TAG" --entrypoint entry.sh --timeout 70 2>&1)
  echo "$OUT" | grep -q "quota exceeded" || break; sleep 30
done
JOB=$(echo "$OUT" | grep -oP 'submitted job \K[0-9]+'); [ -z "$JOB" ] && { echo "$OUT"; exit 1; }
echo "job $JOB (ab-$TAG)"
while ! furiosa-arena list 2>/dev/null | grep -E "^ *$JOB " | grep -qE "SUCCEEDED|FAILED"; do sleep 15; done
furiosa-arena logs "$JOB" > "$D/log.txt" 2>&1
python3 - "$D/log.txt" "$TAG" <<'PY'
import sys, re, statistics as st
lines = open(sys.argv[1]).read().splitlines()
arm = None; k = None; runs = {"A": {}, "B": {}}; fails = {"A": 0, "B": 0}; seen = {"A": 0, "B": 0}
for ln in lines:
    m = re.match(r"@@AB_BEGIN (\d+) ([AB])@@", ln)
    if m: arm = m.group(2); seen[arm] += 1; continue
    if ln.startswith("@@AB_END"): arm = None; continue
    if arm is None: continue
    m = re.match(r"==>\s*(\S+)", ln)
    if m: k = m.group(1); continue
    if "-> FAIL" in ln or "] FAIL" in ln: fails[arm] += 1  # "-> FAIL" and "FAIL -- non-finite ..."
    m = re.search(r"\(of \d+ runs: \[([\d, ]+)\]\)", ln)
    if m and k: runs[arm].setdefault(k, []).extend(int(x) for x in m.group(1).split(","))
order = ["sliding_project_qkv", "sliding_attention_output", "decoder_feedforward"]
print(f"invocations A={seen['A']} B={seen['B']}  numeric FAIL lines A={fails['A']} B={fails['B']}")
med = {}
for a in "AB":
    med[a] = [st.median(runs[a][x]) if runs[a].get(x) else 0 for x in order]
    p = med[a][0] * med[a][1] * med[a][2]
    s = (3.754e17 / p) ** (1 / 3) if p else 0
    print(f"{sys.argv[2]:14} {a}  K1={med[a][0]:9.0f} K2={med[a][1]:8.0f} K3={med[a][2]:9.0f}  score={s:.4f}  (n={[len(runs[a].get(x,[])) for x in order]})")
d = [med['B'][i] - med['A'][i] for i in range(3)]
print(f"{sys.argv[2]:14} B-A  K1={d[0]:+9.0f} K2={d[1]:+8.0f} K3={d[2]:+9.0f}")
PY
