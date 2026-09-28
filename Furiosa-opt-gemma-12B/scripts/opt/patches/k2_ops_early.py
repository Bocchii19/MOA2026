#!/usr/bin/env python3
"""k2_ops_early.py <worktree> -- build the three epilogue operand VRFs before the cross-cluster
gather (source order only), to see whether the scheduler then issues their loads before it."""
import sys
wt = sys.argv[1]
p = wt + "/src/device/sliding/projection.rs"; t = open(p).read()
old_g = """    // One DM-to-DM gather from the 64 row groups of both clusters into cluster 0's epilogue slices.
    let y: DmTensor<f32, Chip, Cluster, AoEpilogueSlices, m![H % 120]> = result.to_dm(&mut device.tdma);
"""
old_v = """    let weight_scale_vrf = ao_operand_vrf(device, &weight_scale);
    let norm_weight_vrf = ao_operand_vrf(device, &norm_weight);
    let residual_vrf = ao_operand_vrf(device, &residual);
"""
assert old_g in t and old_v in t
t = t.replace(old_v, "", 1)
t = t.replace(old_g, old_v + old_g, 1)
open(p, "w").write(t)
print("operand VRFs before the gather (source order)")
