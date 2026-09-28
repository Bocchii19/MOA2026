#!/usr/bin/env python3
"""Summarize named RNGD spans from a diagnostic harness log.

Usage: python3 scripts/analyze_trace.py run.spans.txt [other.log ...] [--json]

Times are elapsed intervals, including any queue/dependency waits within a span.
Subtracting the union of Cluster spans is a diagnostic, not a replacement for
the official cycle count or a measurement of useful work. Engine spans can
overlap Cluster spans and can themselves contain waits.
"""

import argparse
import json
import re
from pathlib import Path


SPAN = re.compile(r"^\s*SPAN\s+(.+?)\s+begin=\s*(\d+)\s+end=\s*(\d+)\s+dur=\s*(\d+)\s*$")
HEADER = re.compile(r"^\s*==>\s+(\S+)")
CYCLES = re.compile(r"^\s*cycles=(\d+)\s*$")


def union(intervals):
    """Return disjoint half-open intervals, merging overlap and adjacency."""
    merged = []
    for begin, end in sorted(intervals):
        if end < begin:
            raise ValueError(f"reversed interval: {begin}..{end}")
        if begin == end:
            continue
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((begin, end))
    return merged


def duration(intervals):
    return sum(end - begin for begin, end in union(intervals))


def intersection_duration(left, right):
    left, right = union(left), union(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        total += max(0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total


def subtract(intervals, others):
    """Total length of `intervals` not covered by any interval in `others`."""
    keep = union(intervals)
    blocked = union(others)
    total = 0
    for begin, end in keep:
        cursor = begin
        for b, e in blocked:
            if e <= cursor or b >= end:
                continue
            if b > cursor:
                total += b - cursor
            cursor = max(cursor, e)
            if cursor >= end:
                break
        if cursor < end:
            total += end - cursor
    return total


def exclusive_spans(spans, name):
    """Per-span exclusive cost: how much the union of `name` spans shrinks without each one.

    This is the per-transfer number an A/B comparison needs. A whole-kernel delta cannot say
    which transfer moved; this can, because a transfer that fully overlaps another costs 0.
    """
    sel = [(b, e) for n, b, e in spans if n == name]
    whole = duration(sel)
    out = []
    for index, (begin, end) in enumerate(sel):
        rest = sel[:index] + sel[index + 1:]
        out.append({
            "index": index,
            "begin": begin,
            "end": end,
            "duration": end - begin,
            "exclusive": whole - duration(rest),
        })
    return out


def parse(text):
    """Keep repeated invocations separate; reject malformed span records."""
    runs, current = [], None
    for lineno, line in enumerate(text.splitlines(), 1):
        header = HEADER.match(line)
        if header:
            current = {"kernel": header[1], "reported_cycles": None, "spans": []}
            runs.append(current)
        match = SPAN.match(line)
        if line.lstrip().startswith("SPAN ") and match is None:
            raise ValueError(f"line {lineno}: malformed SPAN record")
        if match:
            if current is None:
                raise ValueError(f"line {lineno}: span without kernel header")
            name, begin, end, elapsed = match.groups()
            begin, end, elapsed = int(begin), int(end), int(elapsed)
            if end < begin or elapsed != end - begin:
                raise ValueError(f"line {lineno}: inconsistent span duration")
            current["spans"].append((name, begin, end))
        cycles = CYCLES.match(line)
        if cycles and current is not None:
            current["reported_cycles"] = int(cycles[1])
    return [run for run in runs if run["spans"]]


def summarize(run):
    spans = run["spans"]
    hull = max(end for _, _, end in spans) - min(begin for _, begin, _ in spans)
    waits = [(begin, end) for name, begin, end in spans if name == "Cluster"]
    dma = [(begin, end) for name, begin, end in spans if name == "DMA"]
    engines = [(begin, end) for name, begin, end in spans
               if name == "DMA" or name.startswith("Renegade::")]
    wait_union = duration(waits)
    tu = [(begin, end) for name, begin, end in spans if name.startswith("Renegade::")]
    # The two numbers the 2026-09-23 round was missing. `exposed_tu` is Tensor Unit time with the
    # DMA queue idle: on K3 it is ~0 (the queue is busy 83% of the kernel, so TU work is free, which
    # is what E34 measured), but on K2 the weight stream ends early and the remainder sits on the
    # critical path. The "TU work is free" rule is K3's, and does not transfer.
    exposed_tu = subtract(tu, dma)
    all_engines = dma + tu
    return {
        "kernel": run["kernel"],
        "reported_cycles": run["reported_cycles"],
        "span_hull_cycles": hull,
        "cluster_union_cycles": wait_union,
        "cluster_sum_cycles": sum(end - begin for begin, end in waits),
        "hull_minus_cluster_union_cycles": hull - wait_union,
        "dma_union_cycles": duration(dma),
        "cluster_engine_overlap_cycles": intersection_duration(waits, engines),
        "exposed_tu_cycles": exposed_tu,
        "idle_cycles": hull - duration(all_engines) - subtract(waits, all_engines),
        "dma_transfers": exclusive_spans(spans, "DMA"),
        "span_count": len(spans),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--transfers", action="store_true",
                        help="print the per-transfer exclusive cost table")
    args = parser.parse_args()
    rows = []
    for path in args.logs:
        try:
            runs = parse(path.read_text())
            if not runs:
                raise ValueError("no named spans; use the diagnostic trace harness")
            rows.extend({"file": str(path), **summarize(run)} for run in runs)
        except (OSError, ValueError) as exc:
            parser.error(f"{path}: {exc}")
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(f"{row['file']} :: {row['kernel']}")
            print(f"  hull={row['span_hull_cycles']:,}  "
                  f"cluster_union={row['cluster_union_cycles']:,}  "
                  f"hull_minus_wait={row['hull_minus_cluster_union_cycles']:,}  "
                  f"dma_union={row['dma_union_cycles']:,}  "
                  f"wait_engine_overlap={row['cluster_engine_overlap_cycles']:,}")
            print(f"  exposed_tu={row['exposed_tu_cycles']:,}  "
                  f"idle={row['idle_cycles']:,}")
            if args.transfers:
                print("   #      begin        end   duration  exclusive")
                for t in sorted(row["dma_transfers"], key=lambda t: -t["exclusive"]):
                    print(f"  {t['index']:2d} {t['begin']:>10,} {t['end']:>10,} "
                          f"{t['duration']:>10,} {t['exclusive']:>10,}")
        print("Diagnostic intervals only: hull-minus-wait is not useful-work time or an official score.")


if __name__ == "__main__":
    main()
