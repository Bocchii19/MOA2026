//! Fast attention-output projection: both clusters, FP8 weights fed straight into the
//! contraction, activations carried as an FP8 hi/lo pair with f32 accumulation.
//!
//! Layout: 2 clusters split the 3840 output rows (1920 each); inside a cluster the 256 slices
//! are 16 row groups (120 rows) x 16 chunks of the 4096-wide reduction axis (256 wide, 256 B
//! contiguous per weight row piece). Partials of the 16 chunks are summed by the inter-slice
//! reducer.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, Qs};

axes![Hl = 2];

/// 2 clusters = output rows / 1920.
pub(crate) type OCl = m![H / 1920];
/// 256 slices = 16 row groups x 16 reduction chunks.
pub(crate) type OSl = m![H / 120 % 16, Qs / 256];
type OSlOut = m![H / 120 % 16, 1 # 16];

/// activation x scaled by 2^6 before the hi/lo split so the lo part stays in the f8 normal range.
const ACT_SCALE: f32 = 64.0;

/// x (bf16) -> DM f8 hi/lo pair `[Hl, Qs % 256]` (both halves committed straight into the pair tensor).
fn encode_hi_lo(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, OCl, OSl, m![Qs % 256]>,
) -> DmTensor<f8e4m3, Chip, OCl, OSl, m![Hl, Qs % 256]> {
    let mut pair: DmTensor<f8e4m3, Chip, OCl, OSl, m![Hl, Qs % 256]> = DmTensor::new();
    device
        .main
        .begin(x.view())
        .fetch::<m![Qs % 256 / 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs % 256 / 8], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs % 256 / 4], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ACT_SCALE)
        .vector_widen_concat::<m![Qs % 256 / 8], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(pair.view_mut().tile::<m![Hl], 1, m![Hl = 1 #{!} 2, Qs % 256]>(0));

    let hi_vrf: VrfTensor<f32, Chip, OCl, OSl, m![Qs % 256]> = device
        .sub
        .begin(pair.view().tile::<m![Hl], 1, m![Hl = 1 # 2, Qs % 256]>(0))
        .fetch::<m![1], m![Qs % 256]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs % 256 / 8], m![Qs % 8]>()
        .to_vrf();

    device
        .main
        .begin(x.view())
        .fetch::<m![Qs % 256 / 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs % 256 / 8], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs % 256 / 4], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ACT_SCALE)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![Qs % 256 / 8], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(pair.view_mut().tile::<m![Hl], 1, m![Hl = 1 #{!} 2, Qs % 256]>(1));
    pair
}

/// y[H] (bf16, unscaled by row scale, still multiplied by ACT_SCALE) = x[Qs] . W[H, Qs]
pub(crate) fn project_output_raw(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> DmTensor<bf16, Chip, OCl, OSlOut, m![H % 120]> {
    let weight_dm: DmTensor<f8e4m3, Chip, OCl, OSl, m![H % 120, Qs % 256]> = weight.to_dm(&mut device.tdma);

    let x_dm: DmTensor<bf16, Chip, OCl, OSl, m![Qs % 256]> = x.to_dm(&mut device.tdma);
    let pair = encode_hi_lo(device, &x_dm);

    let x_trf: TrfTensor<f8e4m3, Chip, OCl, OSl, m![1], m![Hl, Qs % 256]> = device
        .sub
        .begin(pair.view())
        .fetch::<m![Hl], m![Qs % 256]>()
        .collect::<m![Hl, Qs % 256 / 32], m![Qs % 32]>()
        .to_trf();

    device
        .main
        .begin(weight_dm.view())
        .fetch::<m![H % 120, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![H % 120, Qs / 32 % 8], m![Qs % 32]>()
        .contract_outer::<m![H % 120, Qs / 32 % 8, Hl], m![Qs % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120]>()
        .contract_lane::<m![H % 120], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<OSlOut, m![H % 120]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 / 4], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit()
}

type HiddenRows = m![1 # 8, H / 120];

/// Applies the per-row weight scale (and undoes ACT_SCALE) to y[H] read back from HBM.
pub(crate) fn apply_row_scale(
    device: &mut Device,
    y: &HbmTensor<bf16, Chip, m![H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, crate::device::layout::Cluster, HiddenRows, m![H % 120]> {
    let y: DmTensor<bf16, Chip, crate::device::layout::Cluster, HiddenRows, m![H % 120]> = y.to_dm(&mut device.tdma);
    let scale: DmTensor<bf16, Chip, crate::device::layout::Cluster, HiddenRows, m![H % 120]> =
        weight_scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, crate::device::layout::Cluster, HiddenRows, m![H % 120]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();

    device
        .main
        .begin(y.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 1.0 / ACT_SCALE)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

axes![Dm = 32];

/// post-attention RMSNorm (weight) + residual add, all in the `[H / 120]` layout the row scale
/// already produced, so nothing is re-laid-out between the scale, the norm and the residual.
pub(crate) fn norm_and_residual(
    device: &mut Device,
    t: &DmTensor<bf16, Chip, crate::device::layout::Cluster, HiddenRows, m![H % 120]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    use crate::device::layout::Cluster;
    const H_F32: f32 = H::SIZE as f32;

    let mean_square: DmTensor<f32, Chip, Cluster, HiddenRows, m![1 # 8]> = device
        .main
        .begin(t.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let reduced: DmTensor<f32, Chip, Cluster, m![1 # 8, Dm], m![1 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 8, Dm], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, crate::EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, m![1 # 8, Dm], m![1 # 8]> = device
        .main
        .begin(reduced.view())
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
    let rms: DmTensor<f32, Chip, Cluster, HiddenRows, m![1 # 8]> = unsafe { rms.reshape() };
    let rms_vrf: VrfTensor<f32, Chip, Cluster, HiddenRows, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let weight: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> = rms_weight.to_dm(&mut device.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, HiddenRows, m![H % 120]> = device
        .sub
        .begin(weight.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> = device
        .main
        .begin(t.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    let res: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> = residual.to_dm(&mut device.tdma);
    let res_vrf: VrfTensor<f32, Chip, Cluster, HiddenRows, m![H % 120]> = device
        .sub
        .begin(res.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();

    let out: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> = device
        .main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &res_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    out.view().to_hbm_view(&mut device.tdma, residual.view_mut());
}
