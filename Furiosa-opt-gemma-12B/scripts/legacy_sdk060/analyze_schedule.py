#!/usr/bin/env python3
"""Summarize a furiosa-opt schedule dump: makespan, per-context busy time, top nodes."""
import json, sys, re
from collections import defaultdict

def load(path):
    return json.load(open(path))

def src_of(desc):
    m = re.search(r'--> (\S+)', desc or '')
    return m.group(1) if m else '?'

def expr_of(desc):
    parts = (desc or '').split('\n')
    return parts[-1].strip()[:70] if len(parts) > 1 else ''

def merged_busy(intervals):
    """Union length of [b,e) intervals."""
    if not intervals: return 0
    iv = sorted(intervals)
    total, cb, ce = 0, iv[0][0], iv[0][1]
    for b, e in iv[1:]:
        if b > ce:
            total += ce - cb; cb, ce = b, e
        else:
            ce = max(ce, e)
    return total + (ce - cb)

def report(path, topn=15):
    d = load(path)
    ins = d['instructions']
    makespan = max(i['lifetime']['end'] for i in ins)
    print(f"\n{'='*78}\n{path}\n  makespan = {makespan:,} cycles   ({len(ins)} instructions)\n{'='*78}")

    by_ctx = defaultdict(list)
    for i in ins:
        for c in i['contexts']:
            by_ctx[c].append((i['lifetime']['begin'], i['lifetime']['end']))
    print(f"\n  {'context':<22}{'busy':>12}{'%span':>8}{'nodes':>8}")
    for c, iv in sorted(by_ctx.items(), key=lambda kv: -merged_busy(kv[1])):
        b = merged_busy(iv)
        print(f"  {c:<22}{b:>12,}{100*b/makespan:>7.1f}%{len(iv):>8}")

    # per-source-line cost, summed duration (overlap-blind) and merged
    by_src = defaultdict(list)
    for i in ins:
        by_src[(src_of(i['description']), expr_of(i['description']))].append(
            (i['lifetime']['begin'], i['lifetime']['end']))
    print(f"\n  top source lines by summed duration:")
    print(f"  {'sum':>10}{'merged':>10}{'n':>5}  location / expr")
    rows = sorted(by_src.items(), key=lambda kv: -sum(e-b for b,e in kv[1]))
    for (src, expr), iv in rows[:topn]:
        s = sum(e-b for b, e in iv)
        if s == 0: continue
        print(f"  {s:>10,}{merged_busy(iv):>10,}{len(iv):>5}  {src}  {expr}")

    print(f"\n  longest single nodes:")
    print(f"  {'dur':>10}  {'begin':>9}  {'tpe':<6}{'ctx':<20}{'util':>6}  location")
    for i in sorted(ins, key=lambda i: -(i['lifetime']['end']-i['lifetime']['begin']))[:topn]:
        dur = i['lifetime']['end']-i['lifetime']['begin']
        if dur == 0: continue
        u = i.get('util', {}).get('total_util', 0.0)
        print(f"  {dur:>10,}  {i['lifetime']['begin']:>9,}  {i['tpe']:<6}{','.join(i['contexts'])[:19]:<20}{u:>6.2f}  {src_of(i['description'])}")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        report(p)
