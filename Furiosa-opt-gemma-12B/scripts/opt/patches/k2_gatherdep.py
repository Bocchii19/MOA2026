#!/usr/bin/env python3
"""k2_gatherdep.py <worktree>  (env ALL=1 also pulls weight_scale)

K2's epilogue waited ~2k after the gather for operand loads the scheduler lists after it (a paired
probe removing two of the three tail operand loads measured K2 -827). The gather now writes into a
buffer first filled by a pass that reads the operands (its values are all overwritten by the
gather), so the operand loads and VRFs are ordered before the gather instead of after it."""
import sys, os
wt = sys.argv[1]; ALL = os.environ.get("ALL", "0") == "1"
p = wt + "/src/device/sliding/projection.rs"; t = open(p).read()
old = """    // One DM-to-DM gather from the 64 row groups of both clusters into cluster 0's epilogue slices.
    let y: DmTensor<f32, Chip, Cluster, AoEpilogueSlices, m![H % 120]> = result.to_dm(&mut device.tdma);
"""
vrfs = "    let residual_vrf = ao_operand_vrf(device, &residual);\n"
extra = ""
if ALL:
    vrfs += "    let weight_scale_vrf = ao_operand_vrf(device, &weight_scale);\n"
    extra = "        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_scale_vrf)\n"
new = ("    // The gather overwrites every element of a buffer first filled from the operands, so their\n"
       "    // loads and VRFs are ordered before the gather rather than queued behind it.\n" + vrfs +
"""    let mut y: DmTensor<f32, Chip, Cluster, AoEpilogueSlices, m![H % 120]> = device
        .main
        .begin(norm_weight.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0f32)
        .vector_fp_binary(FpBinaryOp::AddF, &residual_vrf)
""" + extra + """        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();
    result.view().to_dm_view(&mut device.tdma, y.view_mut());
""")
assert old in t; t = t.replace(old, new, 1)
# the originals further down are now duplicates
t = t.replace("    let weight_scale_vrf = ao_operand_vrf(device, &weight_scale);\n    let norm_weight_vrf = ao_operand_vrf(device, &norm_weight);\n    let residual_vrf = ao_operand_vrf(device, &residual);\n",
              ("" if ALL else "    let weight_scale_vrf = ao_operand_vrf(device, &weight_scale);\n") +
              "    let norm_weight_vrf = ao_operand_vrf(device, &norm_weight);\n", 1)
open(p, "w").write(t)
print("K2 gather ordered after operand loads", "(all three)" if ALL else "(norm, residual)")
