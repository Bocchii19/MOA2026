#!/usr/bin/env python3
"""k3_downfused.py <worktree>

Down weight in four row tiles (4+4+4+3), each contracted with the FP4 -> FP8 lookup fused into the
pass (as `down_fused_group3` already does), instead of one 12-row decode pass to an FP8 buffer
followed by three contractions. Group g can start when its own tile lands, and the 7.5k static
decode pass disappears; the price is two more table stores and two more weight commands."""
import sys
wt = sys.argv[1]
p = wt + "/src/device/shared/mlp.rs"; t = open(p).read()
t = t.replace("down_tile_fns!(load_down_tile12, decode_down_tile12, 12);",
              "down_tile_fns!(load_down_tile12, decode_down_tile12, 12);\n"
              "down_tile_fns!(load_down_tile4, decode_down_tile4_unused, 4);", 1)
fused = '''
/// A Down row group contracted straight from its packed FP4 tile (lookup fused into the pass),
/// with its block scales read as a `$rows`-row window of an `$srows`-row scale load.
macro_rules! down_fused_fn {
    ($name:ident, $rows:literal, $srows:literal) => {
        fn $name(
            device: &mut Device,
            x: &DownActivation,
            packed: &DmTensor<f4e2m1, Chip, DownCluster, DownRows, m![H % 60 / 4 = $rows, L % 7680]>,
            scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = $srows, L / 16 % 480]>,
            scale_offset: usize,
            result: &mut DownResult,
            result_offset: usize,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = $rows, L / 16 % 480]> = device
                .sub
                .begin(scale.view().tile::<m![H % 60 / 4 = $srows], $rows, m![H % 60 / 4 = $rows # $srows, L / 16 % 480]>(scale_offset))
                .fetch::<m![H % 60 / 4 = $rows, L / 16 / 240 % 2], m![L / 16 % 240]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 / 4 = $rows, L / 16 / 8 % 60], m![L / 16 % 8]>()
                .to_vrf();
            device.main
                .begin(packed.view())
                .fetch::<m![H % 60 / 4 = $rows], m![L % 7680]>()
                .fetch_table_lookup::<f8e4m3>()
                .collect::<m![H % 60 / 4 = $rows, L / 64 % 120, L / 32 % 2], m![L % 32]>()
                .contract_outer::<m![H % 60 / 4 = $rows, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
                .contract_packet::<m![L / 16 % 4]>()
                .contract_time::<m![H % 60 / 4 = $rows, L / 64 % 120]>()
                .contract_lane::<m![H % 60 / 4 = $rows, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![L / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_intra_slice_reduce::<L, m![H % 60 / 4 = $rows], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_fp_div(DOWN_ACTIVATION_SCALE)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 2]>()
                .commit_view(result.view_mut().tile::<m![H % 60 / 4], $rows, m![H % 60 / 4 = $rows #{!} 15, 1 # 2]>(result_offset));
        }
    };
}
down_fused_fn!(down_fused4_s4, 4, 4);
down_fused_fn!(down_fused4_s8, 4, 8);
'''
t = t.replace("macro_rules! down_scale_fn {", fused.strip() + "\n\nmacro_rules! down_scale_fn {", 1)
old = """    let w0 = load_down_tile12(device, down_weight_packed, 0);"""
new = """    let w0 = load_down_tile4(device, down_weight_packed, 0);
    let w1 = load_down_tile4(device, down_weight_packed, 4);
    let w2 = load_down_tile4(device, down_weight_packed, 8);"""
assert old in t; t = t.replace(old, new, 1)
old = """    let w1 = load_down_tile3(device, down_weight_packed, 12);
    let w0 = decode_down_tile12(device, w0);
    let mut result: DownResult = DmTensor::new();
    down_stream_group4(device, &x, &w0, 0, &scale0, &mut result, 0);
    down_stream_group4_s8(device, &x, &w0, 4, &scale12, 0, &mut result, 4);
    down_stream_group4_s8(device, &x, &w0, 8, &scale12, 4, &mut result, 8);
    down_fused_group3(device, &x, &w1, &scale3, &mut result, 12);"""
new = """    let w3 = load_down_tile3(device, down_weight_packed, 12);
    let mut result: DownResult = DmTensor::new();
    down_fused4_s4(device, &x, &w0, &scale0, 0, &mut result, 0);
    down_fused4_s8(device, &x, &w1, &scale12, 0, &mut result, 4);
    down_fused4_s8(device, &x, &w2, &scale12, 4, &mut result, 8);
    down_fused_group3(device, &x, &w3, &scale3, &mut result, 12);"""
assert old in t; t = t.replace(old, new, 1)
open(p, "w").write(t)
print("Down: 4+4+4+3 tiles, lookup fused into every group")
