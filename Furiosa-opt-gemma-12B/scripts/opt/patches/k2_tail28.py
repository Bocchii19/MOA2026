#!/usr/bin/env python3
"""k2_tail28.py <worktree> (env TILES=28,2)  -- K2 weight row tiles; the last tile's contraction is
the first thing in the tail, so a smaller last tile shortens the tail directly."""
import sys, os
wt = sys.argv[1]
rows = [int(x) for x in os.environ.get("TILES", "28,2").split(",")]
assert sum(rows) == 30 and all(r % 2 == 0 for r in rows)
p = wt + "/src/device/sliding/projection.rs"; t = open(p).read()
old = """output_tile_fns!(load_output_tile24, project_output_tile24, 24, 0);
output_tile_fns!(load_output_tile6, project_output_tile6, 6, 24);"""
off = 0; inst = []
for i, r in enumerate(rows):
    inst.append(f"output_tile_fns!(load_output_t{i}, project_output_t{i}, {r}, {off});"); off += r
assert old in t; t = t.replace(old, "\n".join(inst), 1)
old = """    let w0 = load_output_tile24(device, weight);
    let w1 = load_output_tile6(device, weight);
    let mut result: DmTensor<f32, Chip, OutputCluster, OutputRows, m![H % 30]> = DmTensor::new();
    project_output_tile24(device, &x_trf, w0, &mut result);
    project_output_tile6(device, &x_trf, w1, &mut result);"""
new = "\n".join(f"    let w{i} = load_output_t{i}(device, weight);" for i in range(len(rows)))
new += "\n    let mut result: DmTensor<f32, Chip, OutputCluster, OutputRows, m![H % 30]> = DmTensor::new();\n"
new += "\n".join(f"    project_output_t{i}(device, &x_trf, w{i}, &mut result);" for i in range(len(rows)))
assert old in t; t = t.replace(old, new, 1)
open(p, "w").write(t)
print("K2 tiles", rows)
