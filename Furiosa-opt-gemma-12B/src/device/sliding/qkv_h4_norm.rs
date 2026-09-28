use super::rope_h4::HeadCluster as Cluster;
use super::rope_h4::KvHeadsAcrossSlices as Heads;
use crate::axes::{Ds, Gs};
use crate::{Chip, EPS};
use furiosa_opt_std::prelude::*;

const DS_F32: f32 = Ds::SIZE as f32;

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

/// k with its per-channel scale folded in. The scale is never loaded into VRF straight from its
/// DMA: it is streamed against the gathered K (`x_vrf`), `s * x` goes straight to VRF from Main and
/// the sqrt pass writes the rms VRF (5 passes instead of 6). Without the direct scale -> VRF pass the
/// static schedule issues the 4-byte rope_offset load during the K weight stream (its latency hides
/// in the K tail instead of opening the K -> Q gap) and loads the k rms weight ahead of the Q weight
/// instead of between Q and V. Hardware (paired A/B, 7/7 rounds): K1 -400..-700.
pub(crate) fn normalize_key_scaled(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let x_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
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
        .to_vrf(&mut device.sub);
    // s * x straight into VRF (Main -> Sub VRF), the operand of the final pass.
    let sx: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .main
        .begin(scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
    let weight: DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> = weight.to_dm(&mut device.tdma);
    device.main
        .begin(weight.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sx)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// V with its scale folded in and the sqrt pass writing the rms VRF directly (Main -> Sub VRF).
pub(crate) fn normalize_value_scaled_sv(
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
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
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
        .to_vrf(&mut device.sub);
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

/// V with its scale folded in and no sqrt pass: the mean-square pass writes `h = ms / 2 + EPS / 2`
/// straight into VRF (Main -> Sub VRF), and the final pass rebuilds the rms inline from it:
/// stash `s * x`, `h + h` (operand/operand mode) = ms + EPS, sqrt, then `stash / rms` on the FpDiv
/// ALU (FpFpu is taken by the sqrt). The halving is exact (power of two), so the value matches
/// `normalize_value_scaled_sv` up to the divider's rounding. One Main pass (the sqrt) fewer on the
/// K1 tail.
pub(crate) fn normalize_value_scaled_iv(
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
    let half_ms: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
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
        .vector_fp_div(2.0f32 * DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, 0.5f32 * EPS)
        .vector_final()
        .to_vrf(&mut device.sub);
    device.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary_with_mode(FpBinaryOp::AddF, BinaryArgMode::Mode11, &half_ms)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, Stash)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
