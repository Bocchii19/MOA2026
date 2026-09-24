
use furiosa_opt_std::prelude::*;

use crate::axes::*;
use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

/// The 3840 output rows of the O projection split across the two clusters, 1920 each.
type OCl = m![H / 1920];
/// Within a cluster: 16 row groups of 120 x 16 reduction chunks of 256 (the chunk index is
/// innermost so that the partial sums line up for one `vector_inter_slice_reduce`).
type OSl = m![H % 1920 / 120, Qs / 256];

/// Placement of the post-projection work: both clusters, 16 slices x 240 rows.
type PCl = m![1 # 2];
type PSl = m![1 # 16, H / 240];

/// Scale that lifts the bf16 activation into the f8e4m3 normal range before the hi/lo split,
/// divided out again after the contraction. `x` here is an attention output (a convex
/// combination of value vectors), so `64 * |x|` stays far below the f8e4m3 maximum of 448.
const HILO_SCALE: f32 = 64.0;

/// Fused `sliding_attention_output`: O projection, per-channel scale, post-attention RMSNorm
/// and residual add.
///
/// Three structural differences from a straightforward lowering:
///
/// 1. **No 256-way broadcast of the attention output.** The baseline replicates the 4096-wide
///    activation into every slice through the switch ring. Instead the reduction axis `Qs` is
///    split 16 ways across slices, so each slice DMAs only its own 256-wide chunk and the
///    partial dot products are combined with one `vector_inter_slice_reduce`.
/// 2. **The f8 weights are never dequantized.** `contract_outer` takes an `f8e4m3` operand
///    pair directly, so the O weights feed the tensor unit as they are stored. To keep the
///    activation accurate at f8 the kernel splits it into `hi + lo` lanes (the residual of the
///    first rounding), which the contraction folds in one pass over a two-element `Lane`.
/// 3. **Both clusters do different work.** The output rows are partitioned across the clusters
///    rather than computed twice.
pub(crate) fn attention_output(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    o_weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    // The O weight is the only large transfer; request it first so it streams under the
    // activation encoding and the scale load.
    let wd: DmTensor<f8e4m3, Chip, OCl, OSl, m![H % 120, Qs % 256]> = o_weight.to_dm(&mut ctx.tdma);

    // --- activation: this slice's 256-wide chunk of Qs, as 2^6-scaled f8 hi/lo lanes ---
    let xq: HbmTensorView<'_, bf16, Chip, m![Qs]> = unsafe { x.view().reshape() };
    let xd: DmTensor<bf16, Chip, OCl, OSl, m![Qs % 256]> = xq.to_dm(&mut ctx.tdma);
    let mut hl: DmTensor<f8e4m3, Chip, OCl, OSl, m![Dummy2, Qs % 256]> = DmTensor::new();
    ctx.main
        .begin(xd.view())
        .fetch::<m![Qs % 256 / 8], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs % 256 / 8], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs % 256 / 4], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), HILO_SCALE)
        .vector_widen_concat::<m![Qs % 256 / 8], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(hl.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Qs % 256]>(0));

    let hi_vrf: VrfTensor<f32, Chip, OCl, OSl, m![Qs % 256]> = ctx
        .sub
        .begin(hl.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Qs % 256]>(0))
        .fetch::<m![Qs % 256 / 8], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs % 256 / 8], m![Qs % 8]>()
        .to_vrf();

    ctx.main
        .begin(xd.view())
        .fetch::<m![Qs % 256 / 8], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs % 256 / 8], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs % 256 / 4], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), HILO_SCALE)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![Qs % 256 / 8], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(hl.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Qs % 256]>(1));

    let x_trf: TrfTensor<f8e4m3, Chip, OCl, OSl, m![Dummy2], m![Qs % 256]> = ctx
        .sub
        .begin(hl.view())
        .fetch::<m![Dummy2], m![Qs % 256]>()
        .collect::<m![Dummy2, Qs % 256 / 32], m![Qs % 32]>()
        .to_trf();

    // --- O projection: partial dot products per chunk, reduced across the 16 chunk slices ---
    let y: DmTensor<bf16, Chip, OCl, m![H % 1920 / 120, 1 # 16], m![H % 120]> = ctx
        .main
        .begin(wd.view())
        .fetch::<m![H % 120, Qs % 256 / 32], m![Qs % 32]>()
        .collect::<m![H % 120, Qs % 256 / 32], m![Qs % 32]>()
        .contract_outer::<m![H % 120, Qs % 256 / 64], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120]>()
        .contract_lane::<m![H % 120, Dummy2], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Dummy2, m![H % 120], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(HILO_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![H % 1920 / 120, 1 # 16], m![H % 120]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 / 4], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();

    // --- regroup onto 16 slices x 240 rows for the norm; a DM->DM copy across clusters is
    // rejected by the synchronization checker, so the two halves meet through HBM ---
    let mut scratch: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    y.view().to_hbm_view(&mut ctx.tdma, scratch.view_mut());
    let g: DmTensor<bf16, Chip, PCl, PSl, m![H % 240]> = scratch.to_dm(&mut ctx.tdma);
    let wn: DmTensor<bf16, Chip, PCl, PSl, m![H % 240]> = post_attn_rms_weight.to_dm(&mut ctx.tdma);
    let res: DmTensor<bf16, Chip, PCl, PSl, m![H % 240]> = residual_hbm.to_dm(&mut ctx.tdma);
    // The per-row O scale is applied here, on the norm placement, instead of inside the projection:
    // its load no longer sits in front of the O weight on the DMA queue.
    let sd: DmTensor<bf16, Chip, PCl, PSl, m![H % 240]> = o_weight_scale.to_dm(&mut ctx.tdma);
    let s_vrf: VrfTensor<f32, Chip, PCl, PSl, m![H % 240]> = ctx
        .sub
        .begin(sd.view())
        .fetch::<m![H % 240 / 8], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 240 / 8], m![H % 8]>()
        .to_vrf();

    // --- RMSNorm statistics over all 3840 rows ---
    let ms: DmTensor<f32, Chip, PCl, m![1 # 16, Gf], m![1 # 8]> = ctx
        .main
        .begin(g.view())
        .fetch::<m![H % 240 / 8], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 240 / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 240 / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &s_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![1 # 16, Gf], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, PCl, m![1 # 16, Gf], m![1 # 8]> = ctx
        .main
        .begin(ms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, PCl, PSl, m![1 # 8]> = unsafe { rms.reshape() };

    let rms_vrf: VrfTensor<f32, Chip, PCl, PSl, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let w_vrf: VrfTensor<f32, Chip, PCl, PSl, m![H % 240]> = ctx
        .sub
        .begin(wn.view())
        .fetch::<m![H % 240 / 8], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 240 / 8], m![H % 8]>()
        .to_vrf();
    let r_vrf: VrfTensor<f32, Chip, PCl, PSl, m![H % 240]> = ctx
        .sub
        .begin(res.view())
        .fetch::<m![H % 240 / 8], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 240 / 8], m![H % 8]>()
        .to_vrf();

    // --- residual + normalize(y) * weight ---
    let out: DmTensor<bf16, Chip, PCl, PSl, m![H % 240]> = ctx
        .main
        .begin(g.view())
        .fetch::<m![H % 240 / 8], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 240 / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 240 / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &s_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &w_vrf)
        .vector_widen_concat::<m![H % 240 / 8], m![H % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &r_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    out.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}
