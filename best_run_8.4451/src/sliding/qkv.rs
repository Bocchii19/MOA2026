//! Fused `sliding_project_qkv`: input RMSNorm, Q/K/V projection, per-head RMSNorm, RoPE and the
//! K/V ring-cache writes.
//!
//! 1. **The weight read is 256-byte aligned.** Every slice owns 8 whole rows (4 for K and V) and
//!    reads them as one contiguous run, so each row starts on a 256-byte boundary -- the unit the
//!    HBM path moves in. The earlier layout split the H reduction into four 960-byte chunks,
//!    which leaves three chunks in four straddling that boundary. Measured on the same
//!    15,728,640 bytes of f8 weight: 960-byte chunks stream at 467 B/cycle, while
//!    `sliding_attention_output`, whose reduction chunk lands on 256 bytes exactly, streams at
//!    660. The kernel is weight-DMA bound, so that 1.41x is the biggest lever in it.
//!
//! 2. **The activation reaches every slice over the switch, not the DMA engine.** Giving all 256
//!    slices the whole hidden vector with a replicating DMA would move 1,966,080 bytes a cluster
//!    on the engine the weights need. Instead `x` and the norm weight arrive as 120-element
//!    chunks (61,440 bytes a cluster), are multiplied together on that compact placement, and the
//!    product is all-gathered over the switch ring, which is otherwise idle here. Folding the
//!    norm weight in before the gather also keeps it out of the VRF, which holds 8,192 bytes a
//!    slice and cannot take an f32 copy of a 3,840-wide vector.
//!
//! 3. **The f8 weights feed the tensor unit as stored.** The unit contracts f8 or bf16, and
//!    widening f8 weights to bf16 needs `fetch_table_lookup`, which runs at 2.6 B/cycle against
//!    `fetch`'s 32. So the weight stream stays f8 and the activation is carried as an `hi + lo`
//!    pair of f8e4m3 lanes (the second holds the residual of the first rounding), which keeps it
//!    at roughly bf16 accuracy.
//!
//! 4. **The input RMSNorm's `1 / rms(x)` is never applied.** Q, K and V each go through their own
//!    per-head RMSNorm, which is invariant to a positive scale of its input:
//!    `norm(y / r) = y / sqrt(ms(y) + eps * r^2)`, and with `r^2 = ms(x) + eps` the second term is
//!    ~1e-6 of the first for any real activation. Only the elementwise norm weight is applied.
//!
//! 5. **Both clusters do different rows**: heads 0-7 / 8-15 of Q and 0-3 / 4-7 of K and V.
//!
//! 6. **Few DMA commands, few mid-kernel HBM stores.** Every DMA command costs ~1-2k cycles on
//!    RNGD whatever its size, and an HBM store makes cluster 0 wait for cluster 1. The per-head
//!    regroup is a switch pool on the TU (no DM->DM transfers) and the two RoPE rows go to HBM in
//!    one store.

use furiosa_opt_std::prelude::*;

use crate::axes::*;
use crate::device::layout::Replicated;
use crate::device::shared::hidden::{self, HiddenChunks};
use crate::{Chip, EPS};


/// Magnitude the hi/lo encoder maps the largest |value| to: under the f8e4m3 maximum of 448, so
/// no element saturates however peaked the activation is.
const HILO_TARGET: f32 = 256.0;
/// Keeps the encoder's divisor finite for an all-zero vector.
const MS_FLOOR: f32 = 1e-30;
const DS_F32: f32 = Ds::SIZE as f32;

/// Q rows: one cluster per half of the heads, 64 active slices of 32 whole rows.
type QCl = m![Qs / 2048];
type QSl = m![Qs % 2048 / 32, 1 # 4];
/// K and V rows: the same placement with 8 rows per active slice.
type KCl = m![Ps / 1024];
type KSl = m![Ps % 1024 / 8, 1 # 2];
/// Per-head work: one head per slice. Each PE holds the two query heads of one KV group
/// (`QHead`); the K and V heads of that group sit on the first of them (`KvHead`).
type PCl = m![Ns / 4];
type QHead = m![Ns % 4, Gs, 1 # 32];
type KvHead = m![Ns % 4, 1 # 64];

/// An all-gathered vector seen on the Q weight placement: same 256 slices, same bytes, just the
/// mapping the contraction's operands are in.
fn on_q_slices<'a, D: Scalar>(
    t: &'a DmTensor<D, Chip, QCl, Replicated, m![H]>,
) -> DmTensorView<'a, D, Chip, QCl, QSl, m![H]> {
    unsafe { t.view().reshape() }
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn project_qkv(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    // --- the weights and their per-row scales ---
    let wq: DmTensor<f8e4m3, Chip, QCl, QSl, m![Qs % 32, H]> = q_weight.to_dm(&mut ctx.tdma);
    let wk: DmTensor<f8e4m3, Chip, KCl, KSl, m![Ps % 8, H]> = k_weight.to_dm(&mut ctx.tdma);
    let wv: DmTensor<f8e4m3, Chip, KCl, KSl, m![Ps % 8, H]> = v_weight.to_dm(&mut ctx.tdma);
    let sq: DmTensor<bf16, Chip, QCl, QSl, m![Qs % 32]> = q_weight_scale.to_dm(&mut ctx.tdma);
    let sk: DmTensor<bf16, Chip, KCl, KSl, m![Ps % 8]> = k_weight_scale.to_dm(&mut ctx.tdma);
    let sv: DmTensor<bf16, Chip, KCl, KSl, m![Ps % 8]> = v_weight_scale.to_dm(&mut ctx.tdma);

    // --- x * input_rms_weight on the compact chunk placement, then all-gathered ---
    let x_chunks: DmTensor<bf16, Chip, QCl, HiddenChunks, m![H % 120]> = x.to_dm(&mut ctx.tdma);
    let w_chunks: DmTensor<bf16, Chip, QCl, HiddenChunks, m![H % 120]> = input_rms_weight.to_dm(&mut ctx.tdma);
    let w_vrf: VrfTensor<f32, Chip, QCl, HiddenChunks, m![H % 120]> = ctx
        .sub
        .begin(w_chunks.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 / 8], m![H % 120 % 8]>()
        .to_vrf();
    let xw_chunks: DmTensor<bf16, Chip, QCl, HiddenChunks, m![H % 120]> = ctx
        .main
        .begin(x_chunks.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120 / 4], m![H % 120 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &w_vrf)
        .vector_widen_concat::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 120 % 8 # 16]>()
        .commit_trim::<m![H % 120 % 8]>()
        .commit();
    let xw = hidden::replicate(ctx, &xw_chunks);

    // --- RoPE coefficients for this position (off the critical path) ---
    let rope_hbm = rope_rows(ctx, cos, sin, rope_offset);
    // One replicating load, shared by both per-head placements: `QHead` and `KvHead` are both
    // 256 slices holding the same replicated rows.
    let rope_dm: DmTensor<bf16, Chip, PCl, QHead, m![Dummy2, Ds]> = rope_hbm.to_dm(&mut ctx.tdma);
    let (q_a, q_b) = rope_coefficients::<QHead>(
        ctx,
        rope_dm.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(0),
        rope_dm.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(1),
        q_rms_weight,
    );
    // `QHead` and `KvHead` are both 256 slices holding the same replicated rows, so the KV
    // placement reads the same DM bytes under its own mapping. Views move on use and cannot be
    // cloned inside a device function, so each tile needs its own annotated reshape.
    let rope_kv_cos: DmTensorView<'_, bf16, Chip, PCl, KvHead, m![Dummy2, Ds]> =
        unsafe { rope_dm.view().reshape() };
    let rope_kv_sin: DmTensorView<'_, bf16, Chip, PCl, KvHead, m![Dummy2, Ds]> =
        unsafe { rope_dm.view().reshape() };
    let (k_a, k_b) = rope_coefficients::<KvHead>(
        ctx,
        rope_kv_cos.tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(0),
        rope_kv_sin.tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(1),
        k_rms_weight,
    );

    // --- the encoder scale: `max|x * w| / 256` over the whole 3,840-wide vector ---
    let ms: DmTensor<f32, Chip, QCl, QSl, m![1 # 8]> = ctx
        .main
        .begin(on_q_slices(&xw))
        .fetch::<m![H / 8], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let unscale: DmTensor<f32, Chip, QCl, QSl, m![1 # 8]> = ctx
        .main
        .begin(ms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, MS_FLOOR)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div(HILO_TARGET)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let unscale_vrf: VrfTensor<f32, Chip, QCl, QSl, m![1 # 8]> = ctx
        .sub
        .begin(unscale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    // --- the f8 activation lane: `x * w / unscale`, one pass over the whole vector ---
    let lanes: DmTensor<f8e4m3, Chip, QCl, QSl, m![H]> = ctx
        .main
        .begin(on_q_slices(&xw))
        .fetch::<m![H / 8], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &unscale_vrf)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    let q_trf: TrfTensor<f8e4m3, Chip, QCl, QSl, m![1], m![H]> = ctx
        .sub
        .begin(lanes.view())
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 32], m![H % 32]>()
        .to_trf();
    let kv_lanes: DmTensorView<'_, f8e4m3, Chip, KCl, KSl, m![H]> = unsafe { lanes.view().reshape() };
    let kv_trf: TrfTensor<f8e4m3, Chip, KCl, KSl, m![1], m![H]> = ctx
        .sub
        .begin(kv_lanes)
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 32], m![H % 32]>()
        .to_trf();
    let kv_unscale: DmTensorView<'_, f32, Chip, KCl, KSl, m![1 # 8]> = unsafe { unscale.view().reshape() };
    let kv_unscale_vrf: VrfTensor<f32, Chip, KCl, KSl, m![1 # 8]> = ctx
        .sub
        .begin(kv_unscale)
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    // --- per-row scales ---
    let sq_vrf: VrfTensor<f32, Chip, QCl, QSl, m![Qs % 32]> = ctx
        .sub
        .begin(sq.view())
        .fetch::<m![1], m![Qs % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 4], m![Qs % 8]>()
        .to_vrf();
    let sk_vrf: VrfTensor<f32, Chip, KCl, KSl, m![Ps % 8]> = ctx
        .sub
        .begin(sk.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .to_vrf();
    let sv_vrf: VrfTensor<f32, Chip, KCl, KSl, m![Ps % 8]> = ctx
        .sub
        .begin(sv.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .to_vrf();

    // --- the three projections: each slice owns whole rows, so nothing crosses slices ---
    let q_raw: DmTensor<bf16, Chip, QCl, QSl, m![Qs % 32]> = ctx
        .main
        .begin(wq.view())
        .fetch::<m![Qs % 32, H / 32], m![H % 32]>()
        .collect::<m![Qs % 32, H / 32], m![H % 32]>()
        .contract_outer::<m![Qs % 32, H / 64], m![H % 64], _, _, _>(&q_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 32]>()
        .contract_lane::<m![Qs % 32], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sq_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &unscale_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs % 32 / 4], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    let k_raw = project_kv(ctx, &wk, &kv_trf, &sk_vrf, &kv_unscale_vrf);
    let v_raw = project_kv(ctx, &wv, &kv_trf, &sv_vrf, &kv_unscale_vrf);

    // --- regroup per head over the switch: each head's rows sit on consecutive slices of its
    // group, so a ring-local pool puts the whole head on the group's first slice (TU, no DMA) ---
    let q_src: DmTensor<bf16, Chip, PCl, m![Ns % 4, Gs, Ds / 32, 1 # 4], m![Ds % 32]> =
        unsafe { q_raw.reshape() };
    let k_src: DmTensor<bf16, Chip, PCl, m![Ns % 4, Ds / 8, 1 # 2], m![Ds % 8]> =
        unsafe { k_raw.reshape() };
    let v_src: DmTensor<bf16, Chip, PCl, m![Ns % 4, Ds / 8, 1 # 2], m![Ds % 8]> =
        unsafe { v_raw.reshape() };
    let q: DmTensor<bf16, Chip, PCl, QHead, m![Ds]> = ctx
        .main
        .begin(q_src.view())
        .fetch::<m![1], m![Ds % 32]>()
        .switch::<QHead, m![Ds / 32]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 4 })
        .collect::<m![Ds / 32, Ds % 32 / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();
    let k: DmTensor<bf16, Chip, PCl, KvHead, m![Ds]> = ctx
        .main
        .begin(k_src.view())
        .fetch::<m![1], m![Ds % 8 # 16]>()
        .switch::<KvHead, m![Ds / 8]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 2 })
        .collect::<m![Ds / 8], m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let v: DmTensor<bf16, Chip, PCl, KvHead, m![Ds]> = ctx
        .main
        .begin(v_src.view())
        .fetch::<m![1], m![Ds % 8 # 16]>()
        .switch::<KvHead, m![Ds / 8]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 2 })
        .collect::<m![Ds / 8], m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let q = normalize_and_rotate(ctx, &q, &q_a, &q_b);
    q.view().to_hbm_view(&mut ctx.tdma, q_out.view_mut());

    let k = normalize_and_rotate(ctx, &k, &k_a, &k_b);
    k.dma_scatter::<m![1], _, _>(kv_offset, k_cache);

    let v_rms = head_rms(ctx, &v);
    let v: DmTensor<bf16, Chip, PCl, KvHead, m![Ds]> = ctx
        .main
        .begin(v.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &v_rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    v.dma_scatter::<m![1], _, _>(kv_offset, v_cache);
}

/// One of the two Ps x H projections (K or V) on the `KCl` x `KSl` placement.
fn project_kv(
    ctx: &mut Context,
    weight: &DmTensor<f8e4m3, Chip, KCl, KSl, m![Ps % 8, H]>,
    x_trf: &TrfTensor<f8e4m3, Chip, KCl, KSl, m![1], m![H]>,
    scale_vrf: &VrfTensor<f32, Chip, KCl, KSl, m![Ps % 8]>,
    unscale_vrf: &VrfTensor<f32, Chip, KCl, KSl, m![1 # 8]>,
) -> DmTensor<bf16, Chip, KCl, KSl, m![Ps % 8]> {
    ctx.main
        .begin(weight.view())
        .fetch::<m![Ps % 8, H / 32], m![H % 32]>()
        .collect::<m![Ps % 8, H / 32], m![H % 32]>()
        .contract_outer::<m![Ps % 8, H / 64], m![H % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 8]>()
        .contract_lane::<m![Ps % 8], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), unscale_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ps % 8 / 4], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit()
}

// --- per-head post-processing: one head of `Ds` elements per slice --------------------------

/// Per-head RMSNorm with weight, then RoPE: `(x * A + rotate_half(x) * B) / rms(x)`, where
/// `A = w * cos` and `B = rotate_half(w) * sin` come from [`rope_coefficients`].
fn normalize_and_rotate<Sl: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, PCl, Sl, m![Ds]>,
    a: &VrfTensor<f32, Chip, PCl, Sl, m![Ds]>,
    b: &VrfTensor<f32, Chip, PCl, Sl, m![Ds]>,
) -> DmTensor<bf16, Chip, PCl, Sl, m![Ds]> {
    let rms = head_rms(ctx, x);
    let rot = rotate_half(ctx, x.view());

    let t: DmTensor<f32, Chip, PCl, Sl, m![Ds]> = ctx
        .main
        .begin(rot.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), b)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let t_vrf: VrfTensor<f32, Chip, PCl, Sl, m![Ds]> = ctx
        .sub
        .begin(t.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), a)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_fp_binary(FpBinaryOp::AddF, &t_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// `sqrt(mean(x^2) + EPS)` of the slice's head, as a VRF scalar.
fn head_rms<Sl: M>(ctx: &mut Context, x: &DmTensor<bf16, Chip, PCl, Sl, m![Ds]>) -> VrfTensor<f32, Chip, PCl, Sl, m![1 # 8]> {
    let ms: DmTensor<f32, Chip, PCl, Sl, m![1 # 8]> = ctx
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
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, PCl, Sl, m![1 # 8]> = ctx
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
    ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// This position's rows of the RoPE tables, one 512 B row each in HBM.
///
/// A gather cannot replicate its destination, so the rows are gathered into a single slice, packed
/// into one two-row tensor on the TU and written back to HBM with ONE store; every per-head
/// placement then loads them with a plain replicating DMA. An HBM store in mid-kernel makes
/// cluster 0 wait for cluster 1, so one store instead of two removes a whole sync point (measured
/// 90.0k -> 84.8k with the switch regroup, Arena jobs 32740/32761).
fn rope_rows(
    ctx: &mut Context,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
) -> HbmTensor<bf16, Chip, m![Dummy2, Ds]> {
    let cos_row: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Ds]> = cos.dma_gather_scaled(rope_offset);
    let sin_row: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Ds]> = sin.dma_gather_scaled(rope_offset);
    let mut rows0: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Dummy2, Ds]> = DmTensor::new();
    ctx.main
        .begin(cos_row.view())
        .fetch::<m![1], m![Ds]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(rows0.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(0));
    ctx.main
        .begin(sin_row.view())
        .fetch::<m![1], m![Ds]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(rows0.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(1));
    rows0.to_hbm(&mut ctx.tdma)
}

/// `(w * cos, rotate_half(w) * sin)` as VRF operands for the slices of `Sl`. `sin` arrives with
/// its low half negated, so `x * A + rotate_half(x) * B` is RoPE of `x * w`.
fn rope_coefficients<Sl: M>(
    ctx: &mut Context,
    cos: DmTensorView<'_, bf16, Chip, PCl, Sl, m![Dummy2 = 1 # 2, Ds]>,
    sin: DmTensorView<'_, bf16, Chip, PCl, Sl, m![Dummy2 = 1 # 2, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> (VrfTensor<f32, Chip, PCl, Sl, m![Ds]>, VrfTensor<f32, Chip, PCl, Sl, m![Ds]>) {
    let w: DmTensor<bf16, Chip, PCl, Sl, m![Ds]> = rms_weight.to_dm(&mut ctx.tdma);
    let w_rot = rotate_half(ctx, w.view());

    let a = multiply_to_vrf(ctx, cos, w.view());
    let b = multiply_to_vrf(ctx, sin, w_rot.view());
    (a, b)
}

/// `x * y` elementwise over one head, as a VRF operand.
fn multiply_to_vrf<Sl: M, E: M>(
    ctx: &mut Context,
    x: DmTensorView<'_, bf16, Chip, PCl, Sl, E>,
    y: DmTensorView<'_, bf16, Chip, PCl, Sl, m![Ds]>,
) -> VrfTensor<f32, Chip, PCl, Sl, m![Ds]> {
    let y_vrf: VrfTensor<f32, Chip, PCl, Sl, m![Ds]> = ctx
        .sub
        .begin(y)
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let xy: DmTensor<f32, Chip, PCl, Sl, m![Ds]> = ctx
        .main
        .begin(x)
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &y_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    ctx.sub
        .begin(xy.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

/// `rotate_half` without the sign (the sign lives in `sin`): the two 128-wide halves swapped.
fn rotate_half<Sl: M>(ctx: &mut Context, x: DmTensorView<'_, bf16, Chip, PCl, Sl, m![Ds]>) -> DmTensor<bf16, Chip, PCl, Sl, m![Ds]> {
    let mut rot: DmTensor<bf16, Chip, PCl, Sl, m![Ds]> = DmTensor::new();
    ctx.main
        .begin(x.tile::<m![Ds], 128, m![Ds = 128 # 256]>(0))
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rot.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128));
    ctx.main
        .begin(x.tile::<m![Ds], 128, m![Ds = 128 # 256]>(128))
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rot.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0));
    rot
}

