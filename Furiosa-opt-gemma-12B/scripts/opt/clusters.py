#!/usr/bin/env python3
"""clusters.py <span_log> <test> [runs]  -- per-cluster windows and cross-cluster waits from a span trace.

For each run: each cluster's window (first span to last span, which is what the harness takes the max
of), when its last real work ends, and every `Cluster` span (a cross-cluster sync wait) as
(start offset, duration). The kernel's reported time is the larger window; a long final `Cluster`
span on one cluster is that cluster waiting for the other (launch skew), not work.
Example: clusters.py $OPT_WORK/log_k2B_p2_tr.txt sliding_project_qkv 5"""
import sys

log, test = sys.argv[1], sys.argv[2]
nruns = int(sys.argv[3]) if len(sys.argv) > 3 else 9
lines = open(log).read().splitlines()
for r in range(1, nruns + 1):
    tag = f"SPAN {test}/run{r} "
    spans = {}
    for ln in lines:
        if not ln.startswith(tag):
            continue
        f = dict(x.split("=", 1) for x in ln.split()[2:])
        spans.setdefault(int(f["cl"]), []).append((int(f["beg"]), int(f["end"]), f.get("name", "")))
    if not spans:
        break
    out = []
    for c in sorted(spans):
        s = spans[c]
        beg = min(b for b, _, _ in s)
        end = max(e for _, e, _ in s)
        work = [e for _, e, n in s if n not in ("Task", "Cluster")]
        last = max(work) - beg if work else 0
        waits = [(b - beg, e - b) for b, e, n in s if n == "Cluster"]
        out.append(f"cl{c}: window {end - beg:7d} work-end {last:7d} waits {waits}")
    print(f"run{r}: " + " | ".join(out))
