#!/usr/bin/env python3
"""k2_probe_operands.py <worktree> -- PROBE (numerically wrong): the norm-weight and residual loads
are dropped and weight_scale stands in for both, pricing two tail DMAs in a paired A/B."""
import sys
wt = sys.argv[1]
p = wt + "/src/device/sliding/projection.rs"; t = open(p).read()
old = """    let weight_scale: AoOperand = weight_scale.to_dm(&mut device.tdma);
    let norm_weight: AoOperand = norm_weight.to_dm(&mut device.tdma);
    let residual: AoOperand = residual.to_dm(&mut device.tdma);"""
new = """    let _ = (norm_weight, residual);
    let weight_scale: AoOperand = weight_scale.to_dm(&mut device.tdma);"""
assert old in t; t = t.replace(old, new, 1)
old = """    let norm_weight_vrf = ao_operand_vrf(device, &norm_weight);
    let residual_vrf = ao_operand_vrf(device, &residual);"""
new = """    let norm_weight_vrf = ao_operand_vrf(device, &weight_scale);
    let residual_vrf = ao_operand_vrf(device, &weight_scale);"""
assert old in t; t = t.replace(old, new, 1)
open(p, "w").write(t)
print("K2 probe: two operand loads removed")
