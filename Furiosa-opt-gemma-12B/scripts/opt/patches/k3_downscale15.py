#!/usr/bin/env python3
"""k3_downscale15.py <worktree> -- Down block scales in one 15-row DMA instead of 4+8+3."""
import sys
wt = sys.argv[1]
p = wt + "/src/device/shared/mlp.rs"; t = open(p).read()
t = t.replace("down_scale_fn!(load_down_scale8, 8);", "down_scale_fn!(load_down_scale8, 8);\ndown_scale_fn!(load_down_scale15, 15);", 1)
reps = [
 ("""    scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 8, L / 16 % 480]>,
    scale_offset: usize,""", """    scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 15, L / 16 % 480]>,
    scale_offset: usize,"""),
 (""".begin(scale.view().tile::<m![H % 60 / 4 = 8], 4, m![H % 60 / 4 = 4 # 8, L / 16 % 480]>(scale_offset))""",
  """.begin(scale.view().tile::<m![H % 60 / 4 = 15], 4, m![H % 60 / 4 = 4 # 15, L / 16 % 480]>(scale_offset))"""),
 ("""            scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = $group, L / 16 % 480]>,
            result: &mut DownResult,
            result_offset: usize,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = $group, L / 16 % 480]> = device
                .sub
                .begin(scale.view())""", """            scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 15, L / 16 % 480]>,
            result: &mut DownResult,
            result_offset: usize,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = $group, L / 16 % 480]> = device
                .sub
                .begin(scale.view().tile::<m![H % 60 / 4 = 15], $group, m![H % 60 / 4 = $group # 15, L / 16 % 480]>(0))"""),
 ("""    scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 3, L / 16 % 480]>,
    result: &mut DownResult,
    result_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = 3, L / 16 % 480]> = device
        .sub
        .begin(scale.view())""", """    scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 15, L / 16 % 480]>,
    result: &mut DownResult,
    result_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = 3, L / 16 % 480]> = device
        .sub
        .begin(scale.view().tile::<m![H % 60 / 4 = 15], 3, m![H % 60 / 4 = 3 # 15, L / 16 % 480]>(12))"""),
 ("""    let scale0 = load_down_scale4(device, down_weight_scale, 0);
    let scale12 = load_down_scale8(device, down_weight_scale, 4);
    let scale3 = load_down_scale3(device, down_weight_scale, 12);""",
  """    let scale = load_down_scale15(device, down_weight_scale, 0);"""),
 ("""    down_stream_group4(device, &x, &w0, 0, &scale0, &mut result, 0);
    down_stream_group4_s8(device, &x, &w0, 4, &scale12, 0, &mut result, 4);
    down_stream_group4_s8(device, &x, &w0, 8, &scale12, 4, &mut result, 8);
    down_fused_group3(device, &x, &w1, &scale3, &mut result, 12);""",
  """    down_stream_group4(device, &x, &w0, 0, &scale, &mut result, 0);
    down_stream_group4_s8(device, &x, &w0, 4, &scale, 4, &mut result, 4);
    down_stream_group4_s8(device, &x, &w0, 8, &scale, 8, &mut result, 8);
    down_fused_group3(device, &x, &w1, &scale, &mut result, 12);"""),
]
for a, b in reps:
    assert a in t, a[:70]
    t = t.replace(a, b, 1)
open(p, "w").write(t)
print("Down scale: one 15-row load")
