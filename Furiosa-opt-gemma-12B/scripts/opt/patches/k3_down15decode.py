#!/usr/bin/env python3
"""k3_down15decode.py <worktree>

Down weight as one 15-row load decoded by one pass, so Down stores the FP4 table once instead of
twice (the fused group3 pass stored its own). Block scales keep their 4+8+3 split, so group 0 still
starts after four scale rows -- merging those was what sank the earlier 15-row attempt."""
import sys
wt = sys.argv[1]
p = wt + "/src/device/shared/mlp.rs"; t = open(p).read()
t = t.replace("down_tile_fns!(load_down_tile3, decode_down_tile3_unused, 3);",
              "down_tile_fns!(load_down_tile15, decode_down_tile15, 15);", 1)
t = t.replace("down_stream_group_fn!(down_stream_group4, 12, 4);",
              "down_stream_group_fn!(down_stream_group4, 12, 4);\n"
              "down_stream_group_fn!(down_stream_group4_15, 15, 4);\n"
              "down_stream_group_fn!(down_stream_group3_15, 15, 3);", 1)
# s8 variant over a 15-row fp8 tile
old_s8 = "fn down_stream_group4_s8(\n    device: &mut Device,\n    x: &DownActivation,\n    fp8: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 12, L % 7680]>,"
assert old_s8 in t
t = t.replace(old_s8, old_s8.replace("= 12, L % 7680]>,", "= 15, L % 7680]>,"), 1)
t = t.replace(".begin(fp8.view().tile::<m![H % 60 / 4 = 12], 4, m![H % 60 / 4 = 4 # 12, L % 7680]>(tile_offset))",
              ".begin(fp8.view().tile::<m![H % 60 / 4 = 15], 4, m![H % 60 / 4 = 4 # 15, L % 7680]>(tile_offset))", 1)
old = """    let w0 = load_down_tile12(device, down_weight_packed, 0);"""
new = """    let w0 = load_down_tile15(device, down_weight_packed, 0);"""
assert old in t; t = t.replace(old, new, 1)
old = """    let w1 = load_down_tile3(device, down_weight_packed, 12);
    let w0 = decode_down_tile12(device, w0);
    let mut result: DownResult = DmTensor::new();
    down_stream_group4(device, &x, &w0, 0, &scale0, &mut result, 0);
    down_stream_group4_s8(device, &x, &w0, 4, &scale12, 0, &mut result, 4);
    down_stream_group4_s8(device, &x, &w0, 8, &scale12, 4, &mut result, 8);
    down_fused_group3(device, &x, &w1, &scale3, &mut result, 12);"""
new = """    let w0 = decode_down_tile15(device, w0);
    let mut result: DownResult = DmTensor::new();
    down_stream_group4_15(device, &x, &w0, 0, &scale0, &mut result, 0);
    down_stream_group4_s8(device, &x, &w0, 4, &scale12, 0, &mut result, 4);
    down_stream_group4_s8(device, &x, &w0, 8, &scale12, 4, &mut result, 8);
    down_stream_group3_15(device, &x, &w0, 12, &scale3, &mut result, 12);"""
assert old in t; t = t.replace(old, new, 1)
open(p, "w").write(t)
print("Down: one 15-row load, one decode (one table store), scales 4+8+3")
