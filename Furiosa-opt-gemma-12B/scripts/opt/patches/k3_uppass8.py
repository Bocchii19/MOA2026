#!/usr/bin/env python3
"""k3_uppass8.py <worktree> -- Up's quad-seed scale passes as 8+8+8+6 rows (four passes) instead of
five 6-row passes. The VRF holds 8.5 rows of block scales, and Gate already runs 8-row passes."""
import sys, re
wt = sys.argv[1]
p = wt + "/src/device/shared/mlp.rs"; t = open(p).read()
i = t.index("fn up_gate_scale_pass4(")
j = t.index("\n}\n", i) + 3
body = t[i:j]
gen = body.replace("fn up_gate_scale_pass4(", "fn up_gate_scale_pass4_g<const G: usize>(")
# turn the literal 6-row window into a macro instead: simpler and type-level safe
t = t[:j] + "\n" + """macro_rules! up_gate_scale_pass4_fn {
    ($name:ident, $g:literal) => {
""" + body.replace("fn up_gate_scale_pass4(", "fn $name(").replace("L % 30 = 6", "L % 30 = $g").replace(", 6, m![", ", $g, m![") + """    };
}
up_gate_scale_pass4_fn!(up_gate_scale_pass4_8, 8);
""" + t[j:]
old = """    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 0);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 6);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 12);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 18);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 24);"""
new = """    up_gate_scale_pass4_8(device, &blocks0, &scale, &mut result, 0);
    up_gate_scale_pass4_8(device, &blocks0, &scale, &mut result, 8);
    up_gate_scale_pass4_8(device, &blocks0, &scale, &mut result, 16);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 24);"""
assert old in t; t = t.replace(old, new, 1)
open(p, "w").write(t)
print("Up scale passes 8+8+8+6")
