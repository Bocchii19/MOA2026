#!/usr/bin/env python3
"""Alternate prebuilt A/B public-test binaries within ONE Arena allocation.

Example: bench_batch.py CONTROL control.bin CANDIDATE candidate.bin -n 3 --output DIR
Resume observation (never resubmits): bench_batch.py --resume DIR

Each invocation is a new process using the SAME fixture. All runs, including the
first and any failures, are retained. This measures launch variability on one
Arena allocation; it is NOT the official seven-independent-input evaluation.
"""

import argparse
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CRATE = Path(__file__).resolve().parent.parent
KERNELS = ("sliding_project_qkv", "sliding_attention_output", "decoder_feedforward")
BASELINE_PRODUCT = 3.754e17
TERMINAL = {"succeeded", "completed", "failed", "cancelled", "canceled", "timed_out", "timeout"}
BEGIN = re.compile(r"^@@GEMMA4_BATCH_BEGIN (\d+) (\d+) ([AB])@@$")
END = re.compile(r"^@@GEMMA4_BATCH_END (\d+) (\d+)@@$")


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def estimate(cycles):
    return (BASELINE_PRODUCT / (cycles[KERNELS[0]] * cycles[KERNELS[1]] * cycles[KERNELS[2]])) ** (1 / 3)


def parse_run(lines):
    """Strictly validate one public-harness invocation, allowing extra trace lines."""
    cycles, headers, passes, errors = {}, [], {}, []
    current = None
    for line in lines:
        header = re.fullmatch(r"==>\s*(\S+)\s*", line)
        if header:
            current = header[1]
            headers.append(current)
        cycle = re.fullmatch(r"\s*cycles=(\d+)\s*", line)
        if cycle:
            if current not in KERNELS or current in cycles or int(cycle[1]) <= 0:
                errors.append(f"unexpected/duplicate/nonpositive cycle line: {line}")
            else:
                cycles[current] = int(cycle[1])
        passed = re.match(r"^\[([^]]+)\].*-> PASS\s*$", line)
        if passed:
            label = passed[1].strip()
            passes[label] = passes.get(label, 0) + 1
        if re.search(r"\bFAIL\b|\bpanicked\b|\bnon-finite\b", line):
            errors.append(line)
    expected_passes = {"sliding_project_qkv q": 1, "sliding_project_qkv k": 1,
                       "sliding_project_qkv v": 1, KERNELS[1]: 1, KERNELS[2]: 1}
    if headers != list(KERNELS):
        errors.append(f"expected exactly three ordered kernel headers, got {headers}")
    if set(cycles) != set(KERNELS):
        errors.append("missing kernel cycle counts")
    if passes != expected_passes:
        errors.append(f"incorrect PASS records: {passes}")
    if sum(line.strip() == "all 3 tests passed" for line in lines) != 1:
        errors.append("missing or duplicate all-three-tests PASS marker")
    return cycles, errors


def parse_log(blob, order):
    rows, errors, active, body = [], [], None, []
    for line in blob.splitlines():
        begin, end = BEGIN.fullmatch(line), END.fullmatch(line)
        if begin:
            if active is not None:
                errors.append(f"run {active['index']} has no end marker")
                cycles, failures = parse_run(body)
                failures.append("missing end marker before the next invocation")
                rows.append({**active, "exit_code": None, "cycles": cycles, "passed": False,
                             "errors": failures, "estimated_score": None})
            active = {"index": int(begin[1]), "pair": int(begin[2]), "variant": begin[3]}
            body = []
        elif end:
            if active is None or active["index"] != int(end[1]):
                errors.append(f"unexpected end marker: {line}")
                continue
            cycles, failures = parse_run(body)
            code = int(end[2])
            if code:
                failures.append(f"process exit code {code}")
            row = {**active, "exit_code": code, "cycles": cycles,
                   "passed": not failures, "errors": failures,
                   "estimated_score": None if failures else estimate(cycles)}
            rows.append(row)
            active = None
        elif active is not None:
            body.append(line)
    if active is not None:
        cycles, failures = parse_run(body)
        failures.append("missing end marker; invocation incomplete")
        rows.append({**active, "exit_code": None, "cycles": cycles, "passed": False,
                     "errors": failures, "estimated_score": None})
    observed = [{k: row[k] for k in ("index", "pair", "variant")} for row in rows]
    if observed != order:
        errors.append("observed invocation sequence differs from the complete requested run order")
    if any(not row["passed"] for row in rows):
        errors.append("at least one invocation failed validation; no performance summary is valid")
    return rows, errors


def summarise(rows):
    summary = {"baseline_product_estimate": BASELINE_PRODUCT, "variants": {}, "paired": {}}
    for variant in ("A", "B"):
        selected = [row for row in rows if row["variant"] == variant]
        medians = {kernel: statistics.median(row["cycles"][kernel] for row in selected)
                   for kernel in KERNELS}
        summary["variants"][variant] = {
            "n": len(selected), "median_cycles": medians,
            "estimated_score_from_kernel_medians": estimate(medians),
            "median_of_individual_estimated_scores": statistics.median(row["estimated_score"] for row in selected),
        }
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair"], {})[row["variant"]] = row
    for kernel in KERNELS:
        differences = [pair["B"]["cycles"][kernel] - pair["A"]["cycles"][kernel]
                       for pair in pairs.values()]
        summary["paired"][kernel] = {"b_minus_a_cycles": differences,
                                       "median_b_minus_a_cycles": statistics.median(differences),
                                       "b_wins": sum(delta < 0 for delta in differences), "n": len(pairs)}
    return summary


def command(args):
    return subprocess.run(args, cwd=CRATE, capture_output=True, text=True, timeout=45)


def prepare(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("output directory must be empty; use --resume for an existing job")
    staged = output / "stage"
    staged.mkdir()
    files = {}
    for key, source, name in (("A", args.bin_a, "test_a"), ("B", args.bin_b, "test_b"),
                              ("fixture", args.fixture, "fixtures.safetensors")):
        source = Path(source).resolve(strict=True)
        target = staged / name
        shutil.copyfile(source, target)
        target.chmod(0o644 if key == "fixture" else 0o755)
        files[key] = {"source": str(source), "staged": str(target), "sha256": digest(target),
                      "bytes": target.stat().st_size}
    order = []
    for pair in range(1, args.n + 1):
        sequence = "BA" if args.order == "ba" or (args.order == "alternate" and pair % 2 == 0) else "AB"
        for variant in sequence:
            order.append({"index": len(order) + 1, "pair": pair, "variant": variant})
    shell = ['#!/bin/sh', 'set -u', 'cd "$(dirname "$0")" || exit 1',
             'cp ./test_a ./test_a.exec || exit 126', 'cp ./test_b ./test_b.exec || exit 126',
             'chmod +x ./test_a.exec ./test_b.exec || exit 126',
             f'export TUC_PROFILE_LEVEL={args.profile}', 'failed=0']
    for row in order:
        index, pair, variant = row["index"], row["pair"], row["variant"]
        shell.extend([f"echo '@@GEMMA4_BATCH_BEGIN {index} {pair} {variant}@@'",
                      f'./test_{variant.lower()}.exec 2>&1', 'code=$?',
                      f'echo "@@GEMMA4_BATCH_END {index} $code@@"',
                      'if [ "$code" -ne 0 ]; then failed=1; fi'])
    shell.extend(['rm -f ./test_a.exec ./test_b.exec', 'exit "$failed"'])
    entry = staged / "batch_entrypoint.sh"
    entry.write_text("\n".join(shell) + "\n")
    entry.chmod(0o755)
    metadata = {"created_at": now(), "state": "prepared", "job_id": None,
                "tags": {"A": args.tag_a, "B": args.tag_b}, "files": files,
                "entrypoint_sha256": digest(entry), "run_order": order,
                "timeout_seconds": args.timeout, "profile": args.profile,
                "method": "One Arena allocation; sequential new processes; identical public fixture; no excluded runs. "
                          "Not equivalent to official seven independently seeded inputs.",
                "device": "All invocations inherit the same Arena allocation environment; no device override."}
    save(output / "metadata.json", metadata)
    return output, metadata


def observe(output, metadata, interval):
    job = metadata["job_id"]
    while True:
        status = command(["furiosa-arena", "status", str(job)])
        (output / "status.log").write_text(status.stdout + status.stderr)
        if status.returncode:
            raise RuntimeError("status observation failed; resume this same job")
        state_doc = json.loads(status.stdout)
        state = state_doc["status"].lower()
        metadata.update(state=state, last_observed_at=now(), arena_status=state_doc)
        metadata["observation_state"] = "observing"
        metadata.pop("observation_error", None)
        save(output / "metadata.json", metadata)
        print(f"job {job}: {state}", flush=True)
        if state in TERMINAL:
            break
        time.sleep(interval)
    logs = command(["furiosa-arena", "logs", str(job)])
    (output / "logs-command.stderr").write_text(logs.stderr)
    if logs.returncode:
        (output / "partial.log").write_text(logs.stdout)
        raise RuntimeError("log retrieval failed; resume this same job")
    (output / "raw.log").write_text(logs.stdout)
    rows, errors = parse_log(logs.stdout, metadata["run_order"])
    if state not in {"succeeded", "completed"} or state_doc.get("exit_code", 0) not in (0, None):
        errors.append(f"Arena job did not succeed: {state}, exit={state_doc.get('exit_code')}")
    save(output / "samples.json", rows)
    result = {"job_id": job, "valid": not errors, "errors": errors,
              "method": metadata["method"], "tags": metadata["tags"],
              "summary": None if errors else summarise(rows)}
    save(output / "summary.json", result)
    metadata.update(observation_state="complete", evidence_finished_at=now(), valid=not errors)
    save(output / "metadata.json", metadata)
    print(json.dumps(result, indent=2), flush=True)
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tag_a", "bin_a", "tag_b", "bin_b"):
        parser.add_argument(name, nargs="?")
    parser.add_argument("-n", type=int, default=3, help="pairs in one job (default: 3)")
    parser.add_argument("--order", choices=("ab", "ba", "alternate"), default="ab",
                        help="pair order; alternate means AB, BA, AB, ...")
    parser.add_argument("--fixture", default=str(CRATE / "ref" / "fixtures.safetensors"))
    parser.add_argument("--profile", choices=("info", "trace"), default="info",
                        help="keep trace diagnostics separate from ordinary timing measurements")
    parser.add_argument("--output", help="empty evidence directory")
    parser.add_argument("--resume", help="observe existing evidence directory; never submit again")
    parser.add_argument("--timeout", type=int, default=70,
                        help="remote execution timeout, excluding queue time (default: 70 seconds)")
    parser.add_argument("--interval", type=float, default=15, help="status poll interval (1–60 seconds)")
    parser.add_argument("--prepare-only", action="store_true", help="stage and hash files without submitting")
    args = parser.parse_args()
    if args.n <= 0 or args.timeout <= 0 or not 1 <= args.interval <= 60:
        parser.error("pairs/timeout must be positive and interval between 1 and 60")
    output, metadata = None, None
    try:
        if args.resume:
            output = Path(args.resume).resolve(strict=True)
            metadata = json.loads((output / "metadata.json").read_text())
            if not metadata.get("job_id"):
                raise ValueError("no recorded job ID; inspect submit.log/Arena state before any new submission")
        else:
            if not all((args.tag_a, args.bin_a, args.tag_b, args.bin_b, args.output)):
                parser.error("provide TAG_A BIN_A TAG_B BIN_B --output DIR, or --resume DIR")
            output, metadata = prepare(args)
            if args.prepare_only:
                print(f"prepared only: {output}")
                return 0
            staged = output / "stage"
            metadata.update(state="submitting", submission_started_at=now())
            save(output / "metadata.json", metadata)
            submitted = command(["furiosa-arena", "submit", str(staged / "batch_entrypoint.sh"),
                                 str(staged / "test_a"), str(staged / "test_b"),
                                 str(staged / "fixtures.safetensors"), "--name", "gemma4-paired-batch",
                                 "--entrypoint", "batch_entrypoint.sh", "--timeout", str(args.timeout),
                                 "--env", f"TUC_PROFILE_LEVEL={args.profile}"])
            receipt = submitted.stdout + submitted.stderr
            (output / "submit.log").write_text(receipt)
            found = re.search(r"submitted job (\d+)", receipt, re.IGNORECASE)
            if not found:
                raise RuntimeError("submission returned no job ID; inspect submit.log/Arena state; not retrying")
            metadata.update(job_id=int(found[1]), state="submitted", submitted_at=now())
            save(output / "metadata.json", metadata)
            (output / "job-id.txt").write_text(str(metadata["job_id"]) + "\n")
            print(f"submitted job {metadata['job_id']}; evidence: {output}", flush=True)
        return observe(output, metadata, args.interval)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        if output is not None and metadata is not None:
            metadata.update(observation_error=str(error) or type(error).__name__, observation_stopped_at=now())
            if metadata.get("job_id"):
                metadata["observation_state"] = "waiting_for_observation_resume"
            save(output / "metadata.json", metadata)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        if metadata and metadata.get("job_id"):
            print(f"Job {metadata['job_id']} was not cancelled or restarted; resume with --resume {output}", file=sys.stderr)
            return 2
        return 1


if __name__ == "__main__":
    sys.exit(main())
