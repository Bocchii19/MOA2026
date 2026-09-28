#!/usr/bin/env python3
"""sched.py [--summary] schedule.json [label] -- read a --dump-schedule file.

Without --summary: every DMA instruction plus Main/Sub passes of >= 300 static cycles, in time order.
With --summary: one line of totals (makespan, DMA, Main, Sub)."""
import json, sys, collections
args = sys.argv[1:]
summary = args and args[0] == "--summary"
if summary: args = args[1:]
j = json.load(open(args[0])); label = args[1] if len(args) > 1 else args[0]
ins = j["instructions"]
by = collections.Counter()
for i in ins: by[i["tpe"]] += i["lifetime"]["end"] - i["lifetime"]["begin"]
mk = max(i["lifetime"]["end"] for i in ins)
dma = sum(v for k, v in by.items() if k.startswith("Dma"))
ndma = sum(1 for i in ins if i["tpe"].startswith("Dma"))
if summary:
    print(f"{label:34} makespan={mk:7}  dma={dma:7} ({ndma} cmds)  main={by['Main']:6}  sub={by['Sub']:6}")
    sys.exit(0)
sizes = {t["index"]: t["size"] for t in j["tensors"]}
for i in sorted(ins, key=lambda i: i["lifetime"]["begin"]):
    lf = i["lifetime"]; d = lf["end"] - lf["begin"]; tp = i["tpe"]
    if not (tp.startswith("Dma") or (tp in ("Main", "Sub") and d >= 300)): continue
    b = sum(sizes.get(str(x), 0) for x in i.get("output_tensors", []))
    desc = (i.get("description") or "").replace("\n", " ").split("-->")[-1].strip()[:70]
    print(f"{i['index']:>5} {tp:10} {lf['begin']:>7} {lf['end']:>7} {d:>7} {b:>10}  {desc}")
print(f"makespan {mk}  dma {dma} ({ndma} cmds)  main {by['Main']}  sub {by['Sub']}")
