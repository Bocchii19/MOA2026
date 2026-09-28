#!/usr/bin/env python3
"""dma_desc.py <dump-summary-dir> [--full]  -- every DMA command in resourcelir.desc.json: how many
engines (src/dst descriptor pairs) it is split into, the engines' (cluster, pe, dmn), and the
loop order with strides of the first engine. Loop entries read `Axis=count:stride`, outermost first;
the last entry is the packet (bytes per request)."""
import json, sys, re
d = sys.argv[1]; full = "--full" in sys.argv
j = json.load(open(d + "/text_form/resourcelir.desc.json"))
for key, op in j["operators"].items():
    desc = op.get("description", "") if isinstance(op, dict) else ""
    if not desc.startswith("DmaCommand"): continue
    head = desc.splitlines()[0]
    srcs = re.findall(r"^src(\d+): \[(.*?)\], (\w+)\((.*?)\)$", desc, re.M)
    dsts = re.findall(r"^dst(\d+): \[(.*?)\], (\w+)\((.*?)\)$", desc, re.M)
    locs = []
    for _, _, tier, loc in dsts:
        m = dict(re.findall(r"(\w+)=(\w+)", loc)); locs.append(f"c{m.get('cluster','-')}p{m.get('pe','-')}d{m.get('dmn','-')}")
    print(f"{key:>6} {head:34} engines={len(dsts):2}  {' '.join(sorted(set(locs)))}")
    if dsts:
        print(f"         src0 [{srcs[0][1]}] {srcs[0][2]}")
        print(f"         dst0 [{dsts[0][1]}] {dsts[0][2]}")
    if full: print(desc)
