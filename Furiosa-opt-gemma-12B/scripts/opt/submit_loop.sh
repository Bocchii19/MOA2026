#!/bin/bash
# submit_loop.sh <results_tsv> [interval_s]  -- keep submitting the current opt-next tree whenever the grader
# is idle, from a worktree pinned to opt-next's HEAD (re-pinned when opt-next moves). The last submission
# is therefore always the newest verified tree. Stop by creating $OPT_WORK/submit_loop.stop.
set -u
. "$(dirname "$0")/env.sh"
R="$1"; IV="${2:-60}"; WT="$OPT_WORK/wt_subBest"
[ -d "$WT" ] || git -C "$OPT_REPO" worktree add -q --detach "$WT" opt-next
while [ ! -f "$OPT_WORK/submit_loop.stop" ]; do
  head=$(git -C "$OPT_REPO" rev-parse --short opt-next)
  [ "$(git -C "$WT" rev-parse --short HEAD)" = "$head" ] || git -C "$WT" checkout -q --detach "$head"
  git -C "$WT" checkout -q -- src 2>/dev/null
  if moa-submitter status 2>&1 | tail -3 | grep -qE "building|queued|running|pending|evaluating"; then sleep "$IV"; continue; fi
  out=$(cd "$WT" && moa-submitter submit 2>&1); id=$(echo "$out" | grep -oP 'Submission: +\K\w+')
  [ -z "$id" ] && { sleep "$IV"; continue; }
  while :; do s=$(moa-submitter status "$id" 2>&1); st=$(echo "$s" | grep -oP 'Status: +\K\S+')
    case "$st" in building|queued|running|pending|evaluating|"") sleep 30 ;; *) break ;; esac; done
  k1=$(echo "$s" | grep -oP 'sliding_project_qkv +\K\d+'); k2=$(echo "$s" | grep -oP 'sliding_attention_output +\K\d+')
  k3=$(echo "$s" | grep -oP 'decoder_feedforward +\K\d+'); sc=$(echo "$s" | grep -oP 'Score: +\K\S+')
  echo -e "loop@$head\t$id\t$st\t${k1:--}\t${k2:--}\t${k3:--}\t${sc:--}" >> "$R"
done
