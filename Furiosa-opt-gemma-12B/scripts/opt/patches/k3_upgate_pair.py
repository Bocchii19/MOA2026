#!/usr/bin/env python3
"""k3_upgate_pair.py <worktree>

Up and Gate contracted in ONE pass (`begin_interleaved` over both packed weights), so the FP4
table is stored once for the pair instead of once each. On hardware the second table store sits
between the Up and Gate weight streams and the Gate stream starts right after it ends; with one
store the two streams can run back to back. Scale passes are unchanged except that they read
their half of the shared block buffer."""
import sys, re
wt = sys.argv[1]
p = wt + "/src/device/shared/mlp.rs"; t = open(p).read()

# 1. the pair contraction and a two-matrix block buffer
pair = '''
/// Block-16 partial dots for Up (index 0) and Gate (index 1) out of one contraction pass.
type UpBlocks2 = DmTensor<f32, Chip, UpCluster, UpRows, m![Dummy2, L % 30, H / 16 % 240]>;

/// Up and Gate in one pass: both packed weights stream interleaved, so the FP4 -> FP8 table is
/// stored once for the pair. x hi/lo is still a stream-adapter time broadcast.
fn up_gate_tile30_pair(
    device: &mut Device,
    x: &UpActivation,
    pair: &DmTensor<f4e2m1, Chip, UpCluster, UpRows, m![Dummy2, L % 30 = 30, H]>,
) -> UpBlocks2 {
    let blocks: DmTensor<f32, Chip, UpCluster, UpRows, m![Dummy2, L % 30 = 30, H / 64 % 60, H / 16 % 4]> = device
        .main
        .begin(pair.view())
        .fetch::<m![Dummy2, L % 30 = 30], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![Dummy2, L % 30 = 30, H / 64 % 60, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Dummy2, L % 30 = 30, H / 64 % 60, Gs], m![H % 64], _, _, _>(x)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![Dummy2, L % 30 = 30, H / 64 % 60]>()
        .contract_lane::<m![Dummy2, L % 30 = 30, H / 64 % 60], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
        .commit_trim::<m![H / 16 % 4]>()
        .commit();
    unsafe { blocks.reshape() }
}
'''
anchor = "/// Block scales applied and summed over K for rows `r0..r0+6`."
assert anchor in t
t = t.replace(anchor, pair.strip() + "\n\n" + anchor, 1)

# 2. scale passes that read half `$k` of the shared buffer
def kvariant(src, name_from, name_to, k):
    s = src.replace(f"fn {name_from}(", f"fn {name_to}(")
    s = s.replace("blocks: &UpBlocks,", "blocks: &UpBlocks2,")
    s = re.sub(r"\.begin\(blocks\.view\(\)\.tile::<m!\[L % 30\], (\$g|6), m!\[L % 30 = (\$g|6) # 30, H / 16 % 240\]>\(r0\)\)",
               lambda m: (f".begin(blocks.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, L % 30, H / 16 % 240]>({k})"
                          f".tile::<m![L % 30], {m.group(1)}, m![Dummy2 = 1 # 2, L % 30 = {m.group(2)} # 30, H / 16 % 240]>(r0))"), s)
    assert "UpBlocks2" in s and f">({k})" in s, name_to
    return s
i = t.index("fn up_gate_scale_pass4("); j = t.index("\n}\n", i) + 3
up4 = kvariant(t[i:j], "up_gate_scale_pass4", "up_gate_scale_pass4_pair", 0)
t = t[:j] + "\n" + up4 + t[j:]
i = t.index("macro_rules! up_gate_scale_pass_fn {"); j = t.index("up_gate_scale_pass_fn!(up_gate_scale_pass8, 8);", i) + len("up_gate_scale_pass_fn!(up_gate_scale_pass8, 8);")
mac = t[i:j]
mac_k = kvariant(mac, "$name", "$name", 1).replace("macro_rules! up_gate_scale_pass_fn {", "macro_rules! up_gate_scale_pass_pair_fn {")
mac_k = mac_k.replace("up_gate_scale_pass_fn!(up_gate_scale_pass, 6);", "up_gate_scale_pass_pair_fn!(up_gate_scale_pass_pair, 6);")
mac_k = mac_k.replace("up_gate_scale_pass_fn!(up_gate_scale_pass8, 8);", "up_gate_scale_pass_pair_fn!(up_gate_scale_pass8_pair, 8);")
t = t[:j] + "\n" + mac_k + t[j:]

# 3. the driver
old = """    let up = project_up_gate_small_first(device, &x_trf, up_weight_packed, up_weight_scale);
    let gate = project_up_gate_small_last(device, &x_trf, gate_weight_packed, gate_weight_scale);
    (up, gate)"""
new = """    // Both packed weights land in the two halves of one DM tensor (two DMA commands, one buffer),
    // so a single contraction pass -- and a single table store -- covers the pair.
    let mut pair: DmTensor<f4e2m1, Chip, UpCluster, UpRows, m![Dummy2, L % 30 = 30, H]> = DmTensor::new();
    up_weight_packed.view().to_dm_view(&mut device.tdma, pair.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, L % 30 = 30, H]>(0));
    gate_weight_packed.view().to_dm_view(&mut device.tdma, pair.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, L % 30 = 30, H]>(1));
    let up_scale = load_up_gate_scale4(device, up_weight_scale);
    let gate_scale = load_up_gate_scale(device, gate_weight_scale);
    let blocks = up_gate_tile30_pair(device, &x_trf, &pair);
    let mut up: UpResult = DmTensor::new();
    up_gate_scale_pass4_pair(device, &blocks, &up_scale, &mut up, 0);
    up_gate_scale_pass4_pair(device, &blocks, &up_scale, &mut up, 6);
    up_gate_scale_pass4_pair(device, &blocks, &up_scale, &mut up, 12);
    up_gate_scale_pass4_pair(device, &blocks, &up_scale, &mut up, 18);
    up_gate_scale_pass4_pair(device, &blocks, &up_scale, &mut up, 24);
    let mut gate: UpResult = DmTensor::new();
    up_gate_scale_pass8_pair(device, &blocks, &gate_scale, &mut gate, 0);
    up_gate_scale_pass8_pair(device, &blocks, &gate_scale, &mut gate, 8);
    up_gate_scale_pass8_pair(device, &blocks, &gate_scale, &mut gate, 16);
    up_gate_scale_pass_pair(device, &blocks, &gate_scale, &mut gate, 24);
    (up, gate)"""
assert old in t; t = t.replace(old, new, 1)
open(p, "w").write(t)
print("Up+Gate: one contraction pass, one table store")
