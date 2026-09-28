use super::rope::HeadCluster as Cluster;
use super::rope::KvHeadsAcrossSlices as Heads;
use crate::axes::{Ds, Gs};
use crate::{Chip, EPS};
use furiosa_opt_std::prelude::*;
const DS_F32: f32 = Ds::SIZE as f32;

fn load_norm_weight(
    device: &mut Device,
    w: &HbmTensor<bf16, Chip, m![Ds]>,
) -> VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> {
    let w: DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> = w.to_dm(&mut device.tdma);
    device.sub
        .begin(w.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

/// Normalize each head on its own slice; the reduction axis remains Ds = 256.
pub(crate) fn normalize_query(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]>,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]> {
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
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
    let weight = load_norm_weight(device, weight);
    device.main
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// Normalize each head on its own slice; the reduction axis remains Ds = 256.
pub(crate) fn normalize_key(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
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
    let weight = load_norm_weight(device, weight);
    device.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight)
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
