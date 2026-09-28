#!/usr/bin/env python3
"""exclusive.py <spans-log> <test/runN> [cluster]  -- for every DMA span, the cycles during which it
is the ONLY DMA in flight. Those cycles are what the DMA union (and so, in a DMA-bound kernel, the
window) is made of; a span that is always overlapped by another costs nothing on its own."""
import sys, re
path, test = sys.argv[1], sys.argv[2]; cl = sys.argv[3] if len(sys.argv) > 3 else "0"
d = []
for ln in open(path):
    m = re.match(r"SPAN (\S+) cl=(\d+) beg=(\d+) end=(\d+) dur=(\d+) name=(.*)", ln.strip())
    if m and m.group(1) == test and m.group(2) == cl and m.group(6) == "DMA":
        d.append((int(m.group(3)), int(m.group(4))))
d.sort()
end = max(e for b, e in d)
cnt = bytearray(end + 1)
for b, e in d:
    for t in range(b, e): cnt[t] += 1
tot_ex = 0
print(f"{'begin':>7} {'end':>7} {'dur':>6} {'exclusive':>9}")
for b, e in d:
    ex = sum(1 for t in range(b, e) if cnt[t] == 1)
    tot_ex += ex
    print(f"{b:7} {e:7} {e-b:6} {ex:9}")
union = sum(1 for t in range(end) if cnt[t] > 0)
print(f"union {union}  sum-of-exclusive {tot_ex}  overlapped {union - tot_ex}")
