use super::rope::HeadCluster as Cluster;
use super::rope::KvHeadsAcrossSlices as Heads;
use crate::axes::{Ds, Gs};
use crate::{Chip, EPS};
use furiosa_opt_std::prelude::*;
const DS_F32: f32 = Ds::SIZE as f32;

// The norm weight is the *stream* of the final pass and the projection its VRF operand (same
// arithmetic: x / rms * w == w * x / rms in f32 up to the operation order), so the weight load is
// consumed by a late Main pass and the scheduler keeps it out of the slot between weight loads.

/// Normalize each head on its own slice; the reduction axis remains Ds = 256.
/// q with its per-channel scale folded in (see `normalize_value_scaled`): mean square of s * q,
/// then w * q * s / rms -> bf16. Drops the separate scale pass and its bf16 rounding.
pub(crate) fn normalize_query_scaled(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]>,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]> {
    let scale: VrfTensor<f32, Chip, Cluster, Heads, m![Gs, Ds]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![Gs], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![Gs], m![1 # 8]>()
        .collect::<m![Gs], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![Gs], m![1 # 8]>()
        .collect::<m![Gs], m![1 # 8]>()
        .to_vrf();
    let x_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Gs, Ds]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let weight: DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]> = weight.to_dm(&mut device.tdma);
    device.main
        .begin(weight.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// Normalize each head on its own slice; the reduction axis remains Ds = 256.
/// k with its per-channel scale folded in (final pass gated on the gathered q, see r18).
pub(crate) fn normalize_key_scaled_gated(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
    gate: &DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let zero_q: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(gate.view().tile::<m![Gs], 1, m![Gs = 1 # 2, Ds]>(0))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let zero_q: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .sub
        .begin(zero_q.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let scale: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let x_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let weight: DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> = weight.to_dm(&mut device.tdma);
    device.main
        .begin(weight.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_fp_binary(FpBinaryOp::AddF, &zero_q)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// k with its per-channel scale folded in.
pub(crate) fn normalize_key_scaled(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let scale: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let x_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let weight: DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> = weight.to_dm(&mut device.tdma);
    device.main
        .begin(weight.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// Normalize each head on its own slice; the reduction axis remains Ds = 256.
pub(crate) fn normalize_value(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    device.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// V with its per-channel scale folded in: three passes (mean-square of `v * s` in f32, sqrt,
/// `bf16(v * s / rms)`) instead of scale -> bf16, mean-square, sqrt, normalize. The bf16 rounding
/// of `v * s` between the passes is dropped: at most one bf16 ulp of the output.
pub(crate) fn normalize_value_scaled(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let scale: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    device.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
