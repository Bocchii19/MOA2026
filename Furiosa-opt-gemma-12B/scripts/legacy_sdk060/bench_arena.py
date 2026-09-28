#!/usr/bin/env python3
"""Automated Arena benchmark loop for the three graded Stage 1 kernels.

    python3 scripts/bench_arena.py <tag> -n 6 [--no-build] [--csv FILE]

Builds `tests/test_kernels.rs` for the NPU, submits it to the Arena controller `n`
times, polls every job to completion, and reports per-kernel cycle counts plus the
estimated Stage 1 score for each run.

Why repeated runs: each graded kernel is launched exactly once per job with no
warm-up, so cluster launch skew lands directly in the measured window. Observed
spread is roughly +/-6% on K1, +/-24% on K2 and +/-3% on K3, which is far larger
than most candidate effects. A single run decides nothing; the median over a
handful of runs decides a little; paired alternating runs against a control decide
the most, because they also cancel machine drift.

Results append to OPTIMIZATION_RESULTS.csv so experiments accumulate across
sessions rather than being re-measured.
"""

import argparse
import concurrent.futures as futures
import csv
import os
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
DEFAULT_CSV = CRATE / "OPTIMIZATION_RESULTS.csv"

# From scripts/estimate_score.py: the product of the three unpublished baseline
# cycle counts, recovered from leaderboard rows (they agree to four digits).
BASELINE_PRODUCT = 3.754e17

KERNELS = ["sliding_project_qkv", "sliding_attention_output", "decoder_feedforward"]
TERMINAL = {"succeeded", "failed", "completed", "cancelled", "canceled"}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build():
    print("==> building test_kernels for the NPU", flush=True)
    proc = run(
        ["cargo", "furiosa-opt", "test", "--release", "--test", "test_kernels",
         "--no-run", "--message-format=json"],
        cwd=CRATE,
    )
    if proc.returncode != 0:
        sys.exit(f"build failed:\n{proc.stderr[-4000:]}")
    binary = None
    for line in proc.stdout.splitlines():
        if '"kind":["test"]' in line and '"name":"test_kernels"' in line:
            m = re.search(r'"executable":"([^"]*)"', line)
            if m:
                binary = m.group(1)
    if not binary:
        sys.exit("could not find the built test_kernels binary in cargo's output")
    return Path(binary)


def newest_binary():
    deps = CRATE / "target" / "release" / "deps"
    cands = [p for p in deps.glob("test_kernels-*")
             if p.is_file() and os.access(p, os.X_OK) and p.suffix != ".d"]
    if not cands:
        sys.exit("no previously built test_kernels binary; drop --no-build")
    return max(cands, key=lambda p: p.stat().st_mtime)


def stage(binary):
    """The remote entrypoint runs `./test_runtime` next to `./fixtures.safetensors`, and the
    controller keeps each uploaded file's basename -- so the three files must be staged under
    exactly those names before they are submitted."""
    d = Path(tempfile.mkdtemp(prefix="arena-stage-"))
    shutil.copy(ENTRYPOINT, d / "remote_entrypoint.sh")
    shutil.copy(binary, d / "test_runtime")
    shutil.copy(FIXTURE, d / "fixtures.safetensors")
    for f in d.iterdir():
        f.chmod(0o755)
    return d


def submit(staged, tag, timeout):
    """Submit one job, waiting out the controller's 2-active-job quota; return its id."""
    while True:
        proc = run(
            ["furiosa-arena", "submit",
             str(staged / "remote_entrypoint.sh"), str(staged / "test_runtime"),
             str(staged / "fixtures.safetensors"),
             "--name", tag, "--entrypoint", "remote_entrypoint.sh",
             "--timeout", str(timeout)],
            cwd=CRATE,
        )
        out = proc.stdout + proc.stderr
        m = re.search(r"submitted job (\d+)", out)
        if m:
            return int(m.group(1))
        if "quota exceeded" in out:
            time.sleep(5 + random.random() * 5)
            continue
        print(f"  submit failed: {out.strip()[-300:]}", flush=True)
        return None


def poll(job, interval, deadline):
    """Block until `job` reaches a terminal state; return that state."""
    while time.time() < deadline:
        proc = run(["furiosa-arena", "status", str(job)], cwd=CRATE)
        blob = proc.stdout + proc.stderr
        m = re.search(r'"status"\s*:\s*"?([A-Za-z]+)', blob)
        state = (m.group(1) if m else "").lower()
        if state in TERMINAL:
            return state
        time.sleep(interval)
    return "timeout"


def cycles(job):
    """Parse the three `cycles=N` lines out of a finished job's log."""
    proc = run(["furiosa-arena", "logs", str(job)], cwd=CRATE)
    log = proc.stdout + proc.stderr
    found, current = {}, None
    for line in log.splitlines():
        head = re.match(r"==>\s*(\S+)", line)
        if head:
            current = head.group(1)
        c = re.search(r"cycles=(\d+)", line)
        if c and current:
            found[current] = int(c.group(1))
    passed = "all 3 tests passed" in log
    return found, passed, log


def score(k1, k2, k3):
    return (BASELINE_PRODUCT / (k1 * k2 * k3)) ** (1 / 3)


def one_run(staged, tag, interval, timeout):
    job = submit(staged, tag, timeout)
    if job is None:
        return {"job": -1, "state": "submit_failed"}
    state = poll(job, interval, time.time() + timeout + 300)
    if state not in ("succeeded", "completed"):
        return {"job": job, "state": state}
    found, passed, _ = cycles(job)
    row = {"job": job, "state": state, "passed": passed}
    if len(found) == 3:
        k1, k2, k3 = (found[k] for k in KERNELS)
        row.update(k1=k1, k2=k2, k3=k3, score=score(k1, k2, k3))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="job name, e.g. E39-downsplit")
    ap.add_argument("-n", type=int, default=6, help="number of runs")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--parallel", type=int, default=2,
                    help="jobs in flight at once; the controller allows at most 2 active")
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=70,
                    help="per-job device timeout; the controller caps this at 70")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--note", default="")
    ap.add_argument("--binary", default="",
                    help="submit this prebuilt test binary instead of building//finding one")
    args = ap.parse_args()

    binary = Path(args.binary) if args.binary else (newest_binary() if args.no_build else build())
    staged = stage(binary)
    print(f"==> binary {binary.name} -> {staged}", flush=True)
    print(f"==> submitting {args.n} x '{args.tag}'", flush=True)

    rows = []
    with futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        pending = [pool.submit(one_run, staged, args.tag, args.interval, args.timeout)
                   for _ in range(args.n)]
        for fut in futures.as_completed(pending):
            row = fut.result()
            rows.append(row)
            if "score" in row:
                print(f"  job {row['job']}: K1={row['k1']:>7,} K2={row['k2']:>7,} "
                      f"K3={row['k3']:>8,}  score={row['score']:.4f}"
                      f"{'' if row['passed'] else '  **NUMERICS FAIL**'}", flush=True)
            else:
                print(f"  job {row['job']}: {row['state'].upper()}", flush=True)

    good = [r for r in rows if "score" in r and r.get("passed")]
    if not good:
        print("\nno usable runs")
        return 1

    def stat(key):
        vals = sorted(r[key] for r in good)
        return min(vals), statistics.median(vals)

    print(f"\n=== {args.tag}  ({len(good)}/{args.n} usable) ===")
    print(f"{'':<8}{'min':>10}{'median':>10}")
    for key, label in (("k1", "K1"), ("k2", "K2"), ("k3", "K3")):
        lo, med = stat(key)
        print(f"{label:<8}{lo:>10,.0f}{med:>10,.0f}")
    lo, med = stat("score")
    print(f"{'score':<8}{lo:>10.4f}{med:>10.4f}")
    print(f"samples: {', '.join(f'{r['score']:.4f}' for r in sorted(good, key=lambda r: r['score']))}")

    path = Path(args.csv)
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["tag", "job", "commit", "k1", "k2", "k3", "score",
                        "passed", "timestamp", "note"])
        commit = run(["git", "rev-parse", "--short", "HEAD"], cwd=CRATE).stdout.strip()
        for r in sorted(rows, key=lambda r: r["job"]):
            w.writerow([args.tag, r["job"], commit, r.get("k1", ""), r.get("k2", ""),
                        r.get("k3", ""), f"{r['score']:.4f}" if "score" in r else "",
                        r.get("passed", ""), time.strftime("%Y-%m-%d %H:%M:%S"), args.note])
    print(f"appended {len(rows)} rows to {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
