#!/usr/bin/env python3
"""Paired A/B benchmark of two prebuilt test binaries on real RNGD.

    python3 scripts/bench_pair.py CTRL /path/ctrl-bin E35 /path/e35-bin -n 8

Submits strictly alternating A, B, A, B ... one job at a time and reports both the
pooled statistics and the paired differences.

Why strictly sequential and strictly alternating: the repository's public test
harness runs each kernel once per job with no warm-up. The official grader instead
uses seven independently seeded runs and per-kernel medians (verified 2026-09-22).
The public measurement carries cluster launch skew (K2 swings ~24%) on top of
whatever drift the shared controller's machine is under. Pooled medians across
separately-timed batches confound the change with that drift; alternating adjacent jobs
puts the control and the candidate in the same stretch of machine time, reducing
that confound. Two jobs are never run
concurrently, because the controller may place them on different devices and that
re-introduces exactly the confound pairing is meant to remove.

Read the pair-win count together with the median paired difference and sample size.
A small sample can suggest a useful follow-up without establishing a performance
gain. These estimated scores are not equivalent to official grades.
"""

import argparse
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CRATE = Path(__file__).resolve().parent.parent
FIXTURE = CRATE / "ref" / "fixtures.safetensors"
ENTRYPOINT = CRATE / "scripts" / "rngd" / "remote_entrypoint.sh"
BASELINE_PRODUCT = 3.754e17
KERNELS = ["sliding_project_qkv", "sliding_attention_output", "decoder_feedforward"]
TERMINAL = {"succeeded", "failed", "completed", "cancelled", "canceled"}


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=CRATE)


def stage(binary):
    d = Path(tempfile.mkdtemp(prefix="arena-stage-"))
    shutil.copy(ENTRYPOINT, d / "remote_entrypoint.sh")
    shutil.copy(binary, d / "test_runtime")
    shutil.copy(FIXTURE, d / "fixtures.safetensors")
    for f in d.iterdir():
        f.chmod(0o755)
    return d


def one_job(staged, tag, timeout=70, interval=5):
    """Submit, wait out the 2-active-job quota, poll to terminal, parse cycles."""
    while True:
        p = run(["furiosa-arena", "submit",
                 str(staged / "remote_entrypoint.sh"), str(staged / "test_runtime"),
                 str(staged / "fixtures.safetensors"),
                 "--name", tag, "--entrypoint", "remote_entrypoint.sh",
                 "--timeout", str(timeout)])
        out = p.stdout + p.stderr
        m = re.search(r"submitted job (\d+)", out)
        if m:
            job = int(m.group(1))
            break
        if "quota exceeded" in out:
            time.sleep(5 + random.random() * 5)
            continue
        return {"tag": tag, "job": -1, "state": "submit_failed", "detail": out.strip()[-200:]}

    deadline = time.time() + timeout + 400
    state = "timeout"
    while time.time() < deadline:
        s = run(["furiosa-arena", "status", str(job)])
        blob = s.stdout + s.stderr
        mm = re.search(r'"status"\s*:\s*"?([A-Za-z]+)', blob)
        state = (mm.group(1) if mm else "").lower()
        if state in TERMINAL:
            break
        time.sleep(interval)

    if state not in ("succeeded", "completed"):
        return {"tag": tag, "job": job, "state": state}

    log = run(["furiosa-arena", "logs", str(job)])
    text = log.stdout + log.stderr
    found, current = {}, None
    for line in text.splitlines():
        h = re.match(r"==>\s*(\S+)", line)
        if h:
            current = h.group(1)
        c = re.search(r"cycles=(\d+)", line)
        if c and current:
            found[current] = int(c.group(1))
    row = {"tag": tag, "job": job, "state": state,
           "passed": "all 3 tests passed" in text}
    if len(found) == 3:
        k1, k2, k3 = (found[k] for k in KERNELS)
        row.update(k1=k1, k2=k2, k3=k3,
                   score=(BASELINE_PRODUCT / (k1 * k2 * k3)) ** (1 / 3))
    return row


def summarise(name, rows):
    good = [r for r in rows if "score" in r and r.get("passed")]
    if not good:
        return None
    out = {"n": len(good)}
    for k in ("k1", "k2", "k3", "score"):
        v = sorted(r[k] for r in good)
        out[k] = (min(v), statistics.median(v))
    print(f"\n=== {name}  (n={len(good)}) ===")
    print(f"{'':<8}{'min':>11}{'median':>11}")
    for k, label in (("k1", "K1"), ("k2", "K2"), ("k3", "K3")):
        print(f"{label:<8}{out[k][0]:>11,.0f}{out[k][1]:>11,.0f}")
    print(f"{'score':<8}{out['score'][0]:>11.4f}{out['score'][1]:>11.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_a")
    ap.add_argument("bin_a")
    ap.add_argument("tag_b")
    ap.add_argument("bin_b")
    ap.add_argument("-n", type=int, default=8, help="pairs (2n jobs total)")
    args = ap.parse_args()

    sa, sb = stage(args.bin_a), stage(args.bin_b)
    print(f"A = {args.tag_a}  ({args.bin_a})")
    print(f"B = {args.tag_b}  ({args.bin_b})")
    print(f"{args.n} alternating pairs, one job at a time\n", flush=True)

    rows_a, rows_b, pairs = [], [], []
    for i in range(args.n):
        ra = one_job(sa, args.tag_a)
        rb = one_job(sb, args.tag_b)
        rows_a.append(ra)
        rows_b.append(rb)
        fmt = lambda r: (f"K1={r['k1']:>7,} K2={r['k2']:>6,} K3={r['k3']:>8,} "
                         f"score={r['score']:.4f}" + ("" if r.get("passed") else "  **FAIL**")
                         ) if "score" in r else f"{r['state'].upper()}"
        print(f"pair {i + 1:>2}  A {fmt(ra)}")
        print(f"         B {fmt(rb)}", flush=True)
        if "score" in ra and "score" in rb and ra.get("passed") and rb.get("passed"):
            pairs.append((ra, rb))

    summarise(args.tag_a, rows_a)
    summarise(args.tag_b, rows_b)

    if not pairs:
        print("\nno complete pairs")
        return 1
    print(f"\n=== paired differences (B - A), {len(pairs)} complete pairs ===")
    print(f"{'':<8}{'median':>11}{'mean':>11}{'B wins':>9}")
    for k, label in (("k1", "K1"), ("k2", "K2"), ("k3", "K3")):
        d = [b[k] - a[k] for a, b in pairs]
        wins = sum(1 for x in d if x < 0)
        print(f"{label:<8}{statistics.median(d):>+11,.0f}{statistics.mean(d):>+11,.0f}"
              f"{wins:>6}/{len(d)}")
    d = [b["score"] - a["score"] for a, b in pairs]
    wins = sum(1 for x in d if x > 0)
    print(f"{'score':<8}{statistics.median(d):>+11.4f}{statistics.mean(d):>+11.4f}"
          f"{wins:>6}/{len(d)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
