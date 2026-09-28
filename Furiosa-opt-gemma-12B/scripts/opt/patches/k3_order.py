#!/usr/bin/env python3
"""k3_order.py <worktree>  (env GATE=1 also orders the Down weight ahead of the Gate contraction)

On hardware each weight stream waits for the PREVIOUS contraction to be dispatched: Up weight lands
53.4k -> Up table store 55.0-60.4k -> Up contraction 60.4k -> Gate weight DMA 60.6k (and the same
Gate -> Down), because the static schedule lists the Gate weight DMA (34.4k) after the Up
contraction (30.5k). Here x reaches each contraction through an exact `+ 0` pass whose zero is
derived from that matrix's block-scale load. On hardware the scale has landed long before the
contraction can run (Up scale lands 55.0k), so this costs nothing there, but in the static
schedule it moves the contraction after the next weight DMA, which is then issued first."""
import sys, os
wt = sys.argv[1]; GATE = os.environ.get("GATE", "1") == "1"
p = wt + "/src/device/shared/mlp.rs"; t = open(p).read()
helpers = '''
/// Zero per slice, available only once `scale` (a block-scale load) has landed: an ordering handle.
fn zero_after_scale4(device: &mut Device, scale: &UpScale4) -> VrfTensor<f32, Chip, UpCluster, UpRows, m![1 # 8]> {
    device
        .main
        .begin(
            scale
                .view()
                .tile::<m![Dq4], 1, m![Dq4 = 1 # 4, L % 30, H / 16]>(0)
                .tile::<m![L % 30], 1, m![Dq4 = 1 # 4, L % 30 = 1 # 30, H / 16]>(0),
        )
        .fetch::<m![H / 16 / 8 % 30], m![H / 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 16 / 8 % 30], m![H / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 16 / 4 % 60], m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0f32)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub)
}

fn zero_after_scale(device: &mut Device, scale: &UpScale) -> VrfTensor<f32, Chip, UpCluster, UpRows, m![1 # 8]> {
    device
        .main
        .begin(scale.view().tile::<m![L % 30], 1, m![L % 30 = 1 # 30, H / 16]>(0))
        .fetch::<m![H / 16 / 8 % 30], m![H / 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 16 / 8 % 30], m![H / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 16 / 4 % 60], m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0f32)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// `x + 0` (exact) followed by the TRF load, so a contraction reading it waits on `zero`.
fn up_x_trf_after(
    device: &mut Device,
    x: &DmTensor<f8e4m3, Chip, UpCluster, UpRows, m![Gs, H]>,
    zero: &VrfTensor<f32, Chip, UpCluster, UpRows, m![1 # 8]>,
) -> UpActivation {
    let x: DmTensor<f8e4m3, Chip, UpCluster, UpRows, m![Gs, H]> = device
        .main
        .begin(x.view())
        .fetch::<m![Gs, H / 8], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, H / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, zero)
        .vector_widen_concat::<m![Gs, H / 8], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    device
        .sub
        .begin(x.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf()
}
'''
anchor = "fn project_up_and_gate("
assert anchor in t
t = t.replace(anchor, helpers.strip() + "\n\n" + anchor, 1)
old = """    let x_trf: UpActivation = device
        .sub
        .begin(x.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let up = project_up_gate_small_first(device, &x_trf, up_weight_packed, up_weight_scale);
    let gate = project_up_gate_small_last(device, &x_trf, gate_weight_packed, gate_weight_scale);"""
new = """    // Each contraction reads x through a `+ 0` whose zero comes from its own block-scale load:
    // free on hardware (the scale lands before the contraction could start), but it lists the next
    // weight DMA ahead of the contraction, so that stream is no longer held behind its dispatch.
    let up_scale = load_up_gate_scale4(device, up_weight_scale);
    let gate_scale = load_up_gate_scale(device, gate_weight_scale);
    let z_up = zero_after_scale4(device, &up_scale);
    let x_trf_up = up_x_trf_after(device, &x, &z_up);
""" + ("""    let z_gate = zero_after_scale(device, &gate_scale);
    let x_trf_gate = up_x_trf_after(device, &x, &z_gate);
""" if GATE else """    let x_trf_gate = up_x_trf_after(device, &x, &z_up);
""") + """    let up = project_up_gate_small_first(device, &x_trf_up, up_weight_packed, up_scale);
    let gate = project_up_gate_small_last(device, &x_trf_gate, gate_weight_packed, gate_scale);"""
assert old in t; t = t.replace(old, new, 1)
t = t.replace("""    weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> UpResult {
    let w0 = load_up_gate_tile30(device, weight, 0);
    let scale = load_up_gate_scale4(device, scale);""", """    weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    scale: UpScale4,
) -> UpResult {
    let w0 = load_up_gate_tile30(device, weight, 0);""", 1)
t = t.replace("""    weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> UpResult {
    let w0 = load_up_gate_tile30(device, weight, 0);
    let scale = load_up_gate_scale(device, scale);""", """    weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    scale: UpScale,
) -> UpResult {
    let w0 = load_up_gate_tile30(device, weight, 0);""", 1)
open(p, "w").write(t)
print("K3 weight-DMA ordering handles", "(up + gate)" if GATE else "(up only)")
