#!/bin/bash
# ab_queue.sh <result-file> <candidate>...  -- run ab.sh control-vs-candidate for each, in order.
# Wait for any ab.sh already running first (by PID, never by matching command lines).
. "$(dirname "$0")/env.sh"
OUTF="$1"; shift
for c in "$@"; do
  echo "=== $c ===" >> "$OUTF"
  "$(dirname "$0")/ab.sh" "$OPT_WORK/wt_ctrl" "$OPT_WORK/wt_$c" "$c" 2 2>&1 \
    | grep -E 'built|WARNING|B-A|  A  K1|  B  K1|FAIL|invocations' >> "$OUTF"
done
echo "=== queue done ===" >> "$OUTF"
