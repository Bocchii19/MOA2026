#!/bin/bash
# submit_queue.sh <queue_file> <results_tsv>  -- official submissions, one at a time (the grader allows one
# in progress). Each queue line is "<label> <dir>"; lines are consumed in order and the file is re-read
# after every grading, so lines can be appended while it runs. Results: label id status K1 K2 K3 score.
set -u
Q="$1"; R="$2"; touch "$R"
status_of() { moa-submitter status "$1" 2>&1; }
wait_done() {
  while :; do s=$(status_of "$1"); st=$(echo "$s" | grep -oP 'Status: +\K\S+')
    case "$st" in building|queued|running|pending|evaluating|"") sleep 30 ;; *) break ;; esac; done
  k1=$(echo "$s" | grep -oP 'sliding_project_qkv +\K\d+'); k2=$(echo "$s" | grep -oP 'sliding_attention_output +\K\d+')
  k3=$(echo "$s" | grep -oP 'decoder_feedforward +\K\d+'); sc=$(echo "$s" | grep -oP 'Score: +\K\S+')
  echo "$st ${k1:--} ${k2:--} ${k3:--} ${sc:--}"
}
# an already-running submission (by label "prior:<id>" lines) is just awaited and recorded
n=0
while :; do
  line=$(sed -n "$((n+1))p" "$Q"); [ -z "$line" ] && break; n=$((n+1))
  label=${line%% *}; dir=${line#* }
  if [[ "$label" == prior:* ]]; then id=${label#prior:}; label=$dir
  else
    out=$(cd "$dir" && moa-submitter submit 2>&1); id=$(echo "$out" | grep -oP 'Submission: +\K\w+')
    if [ -z "$id" ]; then echo -e "$label\t-\tsubmit-error\t$(echo "$out" | tail -1)" >> "$R"; continue; fi
  fi
  echo -e "$label\t$id\t$(wait_done "$id" | tr ' ' '\t')" >> "$R"
done
echo "=== queue done ($n lines) ===" >> "$R"
