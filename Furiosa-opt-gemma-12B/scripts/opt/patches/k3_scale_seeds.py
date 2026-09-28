#!/usr/bin/env python3
"""k3_scale_seeds.py <worktree>  (env GATE=q4|plain  UP=q4|plain)
Which block-scale seed Up and Gate use. ver16: Up q4 (quad seed + ring-4 InterTranspose), Gate plain."""
import sys, os
wt = sys.argv[1]; gate = os.environ.get("GATE", "plain"); up = os.environ.get("UP", "q4")
p = wt + "/src/device/shared/mlp.rs"; t = open(p).read()
Q4 = """    let scale = load_up_gate_scale4(device, scale);
    let mut result: UpResult = DmTensor::new();
    let mut blocks0: UpBlocks = DmTensor::new();
    up_gate_tile30(device, x, w0, &mut blocks0, 0);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 0);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 6);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 12);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 18);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 24);"""
PLAIN = """    let scale = load_up_gate_scale(device, scale);
    let mut result: UpResult = DmTensor::new();
    let mut blocks0: UpBlocks = DmTensor::new();
    up_gate_tile30(device, x, w0, &mut blocks0, 0);
    up_gate_scale_pass8(device, &blocks0, &scale, &mut result, 0);
    up_gate_scale_pass8(device, &blocks0, &scale, &mut result, 8);
    up_gate_scale_pass8(device, &blocks0, &scale, &mut result, 16);
    up_gate_scale_pass(device, &blocks0, &scale, &mut result, 24);"""
i_first = t.index("fn project_up_gate_small_first("); i_last = t.index("fn project_up_gate_small_last(")
first, rest = t[:i_last], t[i_last:]
if up == "plain": assert Q4 in first; first = first.replace(Q4, PLAIN, 1)
if gate == "q4": assert PLAIN in rest; rest = rest.replace(PLAIN, Q4, 1)
open(p, "w").write(first + rest)
print(f"up={up} gate={gate}")
