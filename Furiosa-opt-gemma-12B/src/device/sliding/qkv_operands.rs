//! Q/K/V operand spans (plain merge, no fetch lifts). Three loads replace five, all with the
//! over-read pattern (`view().pad()` then `unsafe reshape`), every mapping uniform:
//! * `q_weight_scale` alone (unchanged: p-stride 2 chunks, so it cannot share a mapping with k / v);
//! * `k_weight_scale | v_weight_scale` (contiguous, both 256 elements per head, p-stride 1 chunk):
//!   one `[Two, Ps % 256]` load onto the head slices (cluster `Ps / 1024`, slice `Ps / 256 % 4`);
//! * `q_rms_weight | k_rms_weight` (contiguous, identical on every head slice): one `[Two, Ds]` load.
use furiosa_opt_std::prelude::*;

use super::rope::HeadCluster;
use super::rope::HeadCluster as Cluster;
use super::rope::HeadSlices as Heads;
use crate::{Chip, EPS};
use crate::axes::{Ds, Gs, Ns, Ps, Qs};

axes![Two = 2];

type QScaleDm = super::qkv_proj::QwsHeads;
type KvDm = DmTensor<bf16, Chip, m![Ps / 1024], m![Ps / 256 % 4, 1 # 64], m![Two, Ps % 256]>;
type RmsDm = DmTensor<bf16, Chip, HeadCluster, Heads, m![Two, Ds]>;

pub(crate) type OperandSpan = (QScaleDm, KvDm, RmsDm);

pub(crate) fn load_operand_span(
    device: &mut Device,
    q_scale: QScaleDm,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> OperandSpan {
    // k_weight_scale | v_weight_scale: 8,192 B contiguous.
    let kv: HbmTensorView<'_, bf16, Chip, m![Ps # 4096]> = k_weight_scale.view().pad();
    let kv: HbmTensorView<'_, bf16, Chip, m![Two, Ps]> = unsafe { kv.reshape() };
    let kv: KvDm = kv.to_dm(&mut device.tdma);
    // q_rms_weight | k_rms_weight: 1,024 B contiguous.
    let rms: HbmTensorView<'_, bf16, Chip, m![Ds # 512]> = q_rms_weight.view().pad();
    let rms: HbmTensorView<'_, bf16, Chip, m![Two, Ds]> = unsafe { rms.reshape() };
    let rms: RmsDm = rms.to_dm(&mut device.tdma);
    (q_scale, kv, rms)
}

type OneView<'l> = DmTensorView<'l, bf16, Chip, HeadCluster, Heads, m![Two = 1 # 2, Ds]>;

fn kv_view(span: &OperandSpan, which: usize) -> OneView<'_> {
    let v: DmTensorView<'_, bf16, Chip, HeadCluster, Heads, m![Two, Ds]> = unsafe { span.1.view().reshape() };
    v.tile::<m![Two], 1, m![Two = 1 # 2, Ds]>(which)
}

/// q_rms (which = 0) / k_rms (which = 1) on every head slice.
pub(crate) fn rms_view(span: &OperandSpan, which: usize) -> OneView<'_> {
    span.2.view().tile::<m![Two], 1, m![Two = 1 # 2, Ds]>(which)
}

pub(crate) fn q_scale_vrf(device: &mut Device, span: &OperandSpan) -> VrfTensor<f32, Chip, HeadCluster, Heads, m![Gs, Ds]> {
    // fresh_k1_3: q_ws arrives with both clusters' records in time; the fetch lifts the record onto the cluster.
    let v: VrfTensor<f32, Chip, m![Qs / 2048], m![Qs / 512 % 4, 1 # 64], m![Qs / 256 % 2, Qs / 64 % 4, Qs / 16 % 4, Qs / 8 % 2, Qs % 8]> = device
        .sub
        .begin(span.0.view())
        .fetch::<m![Qs / 2048, Qs / 256 % 2, Qs / 64 % 4, Qs / 16 % 4], m![Qs % 16]>()
        .fetch_cluster_lift::<m![Qs / 2048], m![Qs / 256 % 2, Qs / 64 % 4, Qs / 16 % 4]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 256 % 2, Qs / 64 % 4, Qs / 16 % 4, Qs / 8 % 2], m![Qs % 8]>()
        .to_vrf();
    unsafe { v.reshape() }
}

pub(crate) fn v_scale_vrf(device: &mut Device, span: &OperandSpan) -> VrfTensor<f32, Chip, HeadCluster, Heads, m![Ds]> {
    device
        .sub
        .begin(kv_view(span, 1))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

/// The k_scale stream of a Main pass; continue the chain after it.
macro_rules! k_scale_stream {
    ($device:expr, $span:expr) => {
        $device
            .main
            .begin($crate::device::sliding::qkv_operands::k_scale_view($span))
            .fetch::<m![Ds / 16], m![Ds % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Ds / 8], m![Ds % 8]>()
    };
}
pub(crate) use k_scale_stream;

pub(crate) fn k_scale_view(span: &OperandSpan) -> OneView<'_> {
    kv_view(span, 0)
}

const DS_F32: f32 = Ds::SIZE as f32;

/// Normalize each head on its own slice; the reduction axis remains Ds = 256.
/// q with its per-channel scale folded in (see `normalize_value_scaled`): mean square of s * q,
/// then w * q * s / rms -> bf16. Drops the separate scale pass and its bf16 rounding.
pub(crate) fn normalize_query_span(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]>,
    span: &OperandSpan,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]> {
    let scale = q_scale_vrf(device, span);
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
    device.main
        .begin(rms_view(span, 0))
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
/// instead of between Q and V.
pub(crate) fn normalize_key_span(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    span: &OperandSpan,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let x_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = k_scale_stream!(device, span)
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
    let sx: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = k_scale_stream!(device, span)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
    device.main
        .begin(rms_view(span, 1))
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

/// The un-gathered K contraction result viewed on the head axes (slice `Ns % 4, c, 0`, time `e`, packet
/// 16 channels), exactly as `gather_kv_blocks` reads it.
type KRaw<'l> = DmTensorView<'l, bf16, Chip, Cluster, m![Ns % 4, Ds / 16 % 4, 1 # 16], m![Ds / 64, Ds % 16]>;

/// k_scale on every head slice as f32 VRF (Sub), straight from the kv span.
pub(crate) fn k_scale_vrf(device: &mut Device, span: &OperandSpan) -> VrfTensor<f32, Chip, HeadCluster, Heads, m![Ds]> {
    device
        .sub
        .begin(kv_view(span, 0))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

/// V7 K1 HOL: the K head gather and the k_scale multiply in ONE Main pass: the un-gathered contraction
/// result is fetched, cast to f32, ring-gathered (Broadcast1 {4, 16}, the same channel order as
/// `gather_kv_blocks`) and multiplied by k_scale, giving `s * x` (f32) on the head slices. The product is
/// the same f32 multiply as the ms / sx passes of `normalize_key_span` (bit-identical). Because this pass
/// now waits for the kv span, the static list places the q|k rms load and the Q weight command ahead of
/// the K gather, so on hardware they are issued together with rope_offset / kv at the K contraction
/// dispatch instead of behind the K gather (HOL: rms 1,725 cold, Q issued after the gathered-K anchor).
pub(crate) fn key_sx_gathered(
    device: &mut Device,
    r: &super::qkv_proj::KsResult,
    span: &OperandSpan,
) -> DmTensor<f32, Chip, Cluster, Heads, m![Ds]> {
    let scale = k_scale_vrf(device, span);
    // k_rms on the head slices (Sub VRF): used only as `w - w = 0` so that this pass also waits for the
    // q|k rms load (the list then issues rms and the Q weight ahead of this pass). Value unchanged.
    let w: VrfTensor<f32, Chip, HeadCluster, Heads, m![Ds]> = device
        .sub
        .begin(rms_view(span, 1))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let raw: KRaw<'_> = unsafe { r.view().reshape() };
    device
        .main
        .begin(raw)
        .fetch::<m![Ds / 64], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .switch::<Heads, m![Ds / 64, Ds / 16 % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        // w - w = 0 (w finite), exp(0) = 1, sx / 1 = sx: the value is s * x, the pass depends on w.
        .vector_fp_binary_with_mode(FpBinaryOp::SubF, BinaryArgMode::Mode11, &w)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, Stash)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// `normalize_key_span` from the gathered `s * x` (f32): mean square of sx, sqrt -> VRF, sx -> VRF (Sub),
/// final `w * sx / rms` -> bf16. Same ALU sequence per element as `normalize_key_span`.
pub(crate) fn normalize_key_sx(
    device: &mut Device,
    sx: &DmTensor<f32, Chip, Cluster, Heads, m![Ds]>,
    span: &OperandSpan,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(sx.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
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
    let sx_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(sx.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    device.main
        .begin(rms_view(span, 1))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sx_vrf)
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
/// `reference normalisation up to the divider's rounding. One Main pass (the sqrt) fewer on the
/// tail of the Q/K/V projection.
pub(crate) fn normalize_value_span(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    span: &OperandSpan,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let scale = v_scale_vrf(device, span);
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

