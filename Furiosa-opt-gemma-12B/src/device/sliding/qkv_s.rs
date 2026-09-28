//! K1 Q / K / V projections in a strip geometry (the K2 G3 idea applied to K1's three weights).
//!
//! H = 3,840 = 15 strips of 256 columns. Per cluster, slice (a, c, s) = row group (a, c) x strip slot
//! s = H / 256 (15 live strips in 16 slots, slot 15 dead): Q row groups hold 128 rows (a = Qs / 512 % 4,
//! c = Qs / 16 % 4, rows 512 a + 64 e + 16 c + p), K / V row groups 64 rows (a = Ps / 256 % 4,
//! c = Ps / 16 % 4, rows 256 a + 64 e + 16 c + p), so a live slice holds 128 (64) runs of exactly 256 B
//! and each DMA engine's inner loops walk one whole 3,840 B weight row across the 15 strip slices
//! (stride 256, HBM bit 8 toggling) instead of 64 rows 3,840 B apart (whole-row layout: Q 604 / K 579 /
//! V 583 B/cycle vs ~630 for strip reads).
//!
//! The activation is normalized and split hi/lo on the same strip layout (each slice needs only its
//! 512 B strip of hi/lo in the TRF, one TRF load for all three contractions). Each contraction uses
//! 64 B packets (hi/lo a TRF-side time broadcast) and the Inter-Slice Reducer sums the 15 strips; the
//! dead slot is excluded by the compiler's valid-count tagging of the padded `H / 256 # 16` digit
//! (hardware-checked with the dead slot's weight and activation pre-filled with NaN). The result lands
//! on slot 0 of each row group and a ring-64 `Broadcast1 {4, 16}` gathers each head onto its PE's first
//! slice (`rope_h4::KvHeadsAcrossSlices`) in natural channel order.
use furiosa_opt_std::prelude::*;

use super::rope_h4::HeadCluster as Cluster;
use super::rope_h4::KvHeadsAcrossSlices;
use crate::Chip;
use crate::axes::{Ds, Gs, H, Ps, Qs};

type QCluster = m![Qs / 2048];

/// 16 row groups x 16 strip slots (slot 15 padding).
pub(crate) type QsSlices = m![Qs / 512 % 4, Qs / 16 % 4, H / 256 # 16];

pub(crate) type QsWeight = DmTensor<f8e4m3, Chip, QCluster, QsSlices, m![Qs / 64 % 8, Qs % 16, H % 256]>;

pub(crate) type QsActivation = TrfTensor<f8e4m3, Chip, QCluster, QsSlices, m![1], m![Gs, H % 256]>;

pub(crate) type QsHilo = DmTensor<f8e4m3, Chip, QCluster, QsSlices, m![Gs, H % 256]>;

const QKV_ACTIVATION_SCALE: f32 = 16.0;

pub(crate) fn load_q_strips(device: &mut Device, weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>) -> QsWeight {
    weight.to_dm(&mut device.tdma)
}

pub(crate) fn q_strip_trf(device: &mut Device, x: &QsHilo) -> QsActivation {
    device
        .sub
        .begin(x.view())
        .fetch::<m![Gs, H / 32 % 8], m![H % 32]>()
        .collect::<m![Gs, H / 32 % 8], m![H % 32]>()
        .to_trf()
}

/// Q: 64 B packets (two 32 B weight flits per packet, hi/lo `Gs` a TRF-side time broadcast), 1,024
/// outer steps per slice (vs 1,920 for the whole-row 32 B form), the 15 strips summed by the Inter-Slice
/// Reducer (the dead strip slot is excluded by the compiler's valid-count tagging; promoting a time digit
/// into the freed slot is rejected with valid counts). The result: 128 rows on slot 0 of each row
/// group, scaled back by 1/16 and rounded to bf16.
pub(crate) type QsResult = DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, Qs / 16 % 4, 1 # 16], m![Qs / 64 % 8, Qs % 16]>;

pub(crate) fn contract_q_strips(device: &mut Device, x: &QsActivation, weight: QsWeight) -> QsResult {
    device
        .main
        .begin(weight.view())
        .fetch::<m![Qs / 64 % 8, Qs % 16, H / 64 % 4, H / 32 % 2], m![H % 32]>()
        .collect::<m![Qs / 64 % 8, Qs % 16, H / 64 % 4, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Qs / 64 % 8, Qs % 16, H / 64 % 4, Gs], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 64 % 8, Qs % 16]>()
        .contract_lane::<m![Qs / 64 % 8, Qs % 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Qs / 512 % 4, Qs / 16 % 4, 1 # 16], m![Qs / 64 % 8, Qs % 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 64 % 8, Qs % 16 / 4], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit()
}

/// The 4 row groups of each PE (slot 0 of slices 64 a + 16 c) gathered onto the PE's first slice in
/// natural channel order: each source sends its 128 rows as 8 packets of 16 consecutive channels
/// (`Qs % 16`), the ring-64 `Broadcast1 {4, 16}` appends `c` innermost in time, so time x packet is
/// `[e, c, p]` = channel `64 e + 16 c + p`.
pub(crate) fn gather_q_strips(
    device: &mut Device,
    r: &QsResult,
) -> DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, 1 # 4, 1 # 16], m![Qs % 512]> {
    device
        .main
        .begin(r.view())
        .fetch::<m![Qs / 64 % 8], m![Qs % 16]>()
        .switch::<m![Qs / 512 % 4, 1 # 4, 1 # 16], m![Qs / 64 % 8, Qs / 16 % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
        .collect::<m![Qs / 64 % 8, Qs / 16 % 4], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit()
}

axes![QsSlot = 16];

/// The strip slots as a fully live broadcast digit (reduce / switch outputs).
type QsFull = m![Qs / 512 % 4, Qs / 16 % 4, QsSlot];

/// Quarter seeds (K2's J4 form): slice (a, c, s) receives the 128-B quarter c of strip s, so each engine
/// writes 60 pieces of 128 B instead of 120 of 256 B; one ring-64 `Broadcast1 {4, 16}` (one time step per
/// source) gives every row group its whole 512-B strip in natural channel order.
type XsQuarter = DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, H / 64 % 4, H / 256 # 16], m![H % 64]>;

/// The weight spread, f32 through the switch so it can add `zero_vrf` (= 0, exact): the dependency on
/// the zero VRF places it after the zero pass in the static schedule, so the weight VRF pass comes
/// after the K weight DMA's issue and the x DMA is issued right behind the zero pass at the seed's landing.
fn spread_quarters(
    device: &mut Device,
    q: &XsQuarter,
    zero_vrf: &VrfTensor<f32, Chip, QCluster, QsSlices, m![1 # 8]>,
) -> XsBf16 {
    device
        .main
        .begin(q.view())
        .fetch::<m![1], m![H % 64]>()
        .fetch_cast::<f32>()
        .switch::<QsSlices, m![H / 64 % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
        .collect::<m![H / 8 % 32], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 64], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, zero_vrf)
        .vector_widen_concat::<m![H / 8 % 32], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// The x spread, f32 through the switch so it can add `zero_vrf` (= 0): the data dependency places it
/// after the zero pass's VRF write in the static schedule, so the K weight DMA's issue is listed before
/// it (otherwise x's first consumer ties with the K DMA at x's static landing and the wait on x holds
/// the K issue).
fn spread_x_quarters(
    device: &mut Device,
    q: &XsQuarter,
    zero_vrf: &VrfTensor<f32, Chip, QCluster, QsSlices, m![1 # 8]>,
) -> XsBf16 {
    device
        .main
        .begin(q.view())
        .fetch::<m![1], m![H % 64]>()
        .fetch_cast::<f32>()
        .switch::<QsSlices, m![H / 64 % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
        .collect::<m![H / 8 % 32], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 64], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, zero_vrf)
        .vector_widen_concat::<m![H / 8 % 32], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// x (and the input RMS weight) on every row group, each slice holding its 256-column strip.
pub(crate) type XsBf16 = DmTensor<bf16, Chip, QCluster, QsSlices, m![H % 256]>;

/// Input RMSNorm on the strip layout: the mean square sums the 15 strips with the Inter-Slice Reducer
/// (dead slot excluded by valid-count tagging) and broadcasts it back over the 16 slots. As in
/// `qkv_h4::normalize_input_x8w`, the rms weight is loaded first and `sum(0 * w)` joins the mean square
/// so the scheduler lists the K weight load right after x.
pub(crate) fn normalize_input_s(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> XsBf16 {
    let weight_seed: XsQuarter = rms_weight.to_dm(&mut device.tdma);
    // `sum(0 * w)` on the weight SEED (Sub, 64 elements per slice, the seed's first consumer at its
    // landing): the x DMA is issued right behind it, the K weight DMA right after the zero VRF
    // (before x lands), and both spreads -- which add this zero -- come after the K issue.
    let zero_seed: DmTensor<f32, Chip, QCluster, m![Qs / 512 % 4, H / 64 % 4, H / 256 # 16], m![1 # 8]> = device
        .sub
        .begin(weight_seed.view())
        .fetch::<m![H / 16 % 4], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 16], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let zero: DmTensor<f32, Chip, QCluster, QsSlices, m![1 # 8]> = unsafe { zero_seed.reshape() };
    let zero_vrf: VrfTensor<f32, Chip, QCluster, QsSlices, m![1 # 8]> = device
        .sub
        .begin(zero.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let weight_dm = spread_quarters(device, &weight_seed, &zero_vrf);
    let x_seed: XsQuarter = x.to_dm(&mut device.tdma);
    let x = spread_x_quarters(device, &x_seed, &zero_vrf);
    let mean_square: DmTensor<f32, Chip, QCluster, QsFull, m![1 # 8]> = device
        .sub
        .begin(x.view())
        .fetch::<m![H / 16 % 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 32], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 64], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_fp_binary(FpBinaryOp::AddF, &zero_vrf)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<QsFull, m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, QCluster, QsFull, m![1 # 8]> = device
        .sub
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, crate::EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, QCluster, QsSlices, m![1 # 8]> = unsafe { rms.reshape() };
    let weight_vrf: VrfTensor<f32, Chip, QCluster, QsSlices, m![H % 256]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 32], m![H % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, QCluster, QsSlices, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 32], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 64], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 32], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// `[hi, lo]` per strip slice (`hi = f8(16 x)`, `lo = f8(16 x - hi)`), as `qkv_h4::split_qkv_activation8`.
fn split_s(device: &mut Device, x: &XsBf16) -> QsHilo {
    let mut out: QsHilo = DmTensor::new();
    device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 32], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 64], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), QKV_ACTIVATION_SCALE)
        .vector_widen_concat::<m![H / 8 % 32], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, H % 256]>(0));
    let hi_neg: XsF8 = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 32], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 64], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -QKV_ACTIVATION_SCALE)
        .vector_widen_concat::<m![H / 8 % 32], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    let hi_neg_vrf: VrfTensor<f32, Chip, QCluster, QsSlices, m![H % 256]> = device
        .sub
        .begin(hi_neg.view())
        .fetch::<m![H / 32 % 8], m![H % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 32], m![H % 8]>()
        .to_vrf();
    device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 32], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 64], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), QKV_ACTIVATION_SCALE)
        .vector_fp_binary(FpBinaryOp::AddF, &hi_neg_vrf)
        .vector_widen_concat::<m![H / 8 % 32], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, H % 256]>(1));
    out
}

type XsF8 = DmTensor<f8e4m3, Chip, QCluster, QsSlices, m![H % 256]>;

/// K / V in the strip geometry: per cluster 1,024 rows, slice (a = Ps / 256 % 4, c = Ps / 16 % 4, s),
/// a live slice holding 64 rows x 256 B = 16 KiB (rows 256 a + 64 e + 16 c + p).
type KvCluster = m![Ps / 1024];
pub(crate) type KsSlices = m![Ps / 256 % 4, Ps / 16 % 4, H / 256 # 16];
pub(crate) type KsWeight = DmTensor<f8e4m3, Chip, KvCluster, KsSlices, m![Ps / 64 % 4, Ps % 16, H % 256]>;
/// Relabelled onto the Q strip axes (same slices, same element extents) so the Q strip TRF serves K / V.
type KsWeightQ = DmTensor<f8e4m3, Chip, QCluster, QsSlices, m![Qs / 64 % 4, Qs % 16, H % 256]>;
type KsResultQ = DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, Qs / 16 % 4, 1 # 16], m![Qs / 64 % 4, Qs % 16]>;
pub(crate) type KsResult = DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, Ps / 16 % 4, 1 # 16], m![Ps / 64 % 4, Ps % 16]>;

pub(crate) fn contract_kv_strips(device: &mut Device, x: &QsActivation, weight: KsWeight) -> KsResult {
    let weight: KsWeightQ = unsafe { weight.reshape() };
    let out: KsResultQ = device
        .main
        .begin(weight.view())
        .fetch::<m![Qs / 64 % 4, Qs % 16, H / 64 % 4, H / 32 % 2], m![H % 32]>()
        .collect::<m![Qs / 64 % 4, Qs % 16, H / 64 % 4, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Qs / 64 % 4, Qs % 16, H / 64 % 4, Gs], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 64 % 4, Qs % 16]>()
        .contract_lane::<m![Qs / 64 % 4, Qs % 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Qs / 512 % 4, Qs / 16 % 4, 1 # 16], m![Qs / 64 % 4, Qs % 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 64 % 4, Qs % 16 / 4], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();
    unsafe { out.reshape() }
}

/// The 4 row groups of each head (slot 0 of slices 64 a + 16 c) gathered onto the PE's first slice in
/// natural channel order (time x packet = `[e, c, p]` = channel `64 e + 16 c + p`), plus the scale.
pub(crate) fn gather_kv_strips(
    device: &mut Device,
    r: &KsResult,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let gathered: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 4, 1 # 16], m![Ps % 256]> = device
        .main
        .begin(r.view())
        .fetch::<m![Ps / 64 % 4], m![Ps % 16]>()
        .switch::<m![Ps / 256 % 4, 1 # 4, 1 # 16], m![Ps / 64 % 4, Ps / 16 % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
        .collect::<m![Ps / 64 % 4, Ps / 16 % 4], m![Ps % 16]>()
        .commit_trim::<m![Ps % 16]>()
        .commit();
    let weight_scale: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

pub(crate) type WeightsS = (QsWeight, KsWeight, KsWeight);

pub(crate) fn load_weights_s(
    device: &mut Device,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
) -> WeightsS {
    let q = load_q_strips(device, q_weight);
    let k: KsWeight = k_weight.to_dm(&mut device.tdma);
    let v: KsWeight = v_weight.to_dm(&mut device.tdma);
    (q, k, v)
}

/// All three projections in the strip geometry from one strip TRF.
pub(crate) fn project_s(
    device: &mut Device,
    x: &XsBf16,
    w: WeightsS,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let (q_w, k_w, v_w) = w;
    let xs = split_s(device, x);
    let xs_trf = q_strip_trf(device, &xs);
    let k = contract_kv_strips(device, &xs_trf, k_w);
    let q = contract_q_strips(device, &xs_trf, q_w);
    let v = contract_kv_strips(device, &xs_trf, v_w);
    let q = gather_q_strips(device, &q);
    let q_scale: DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, 1 # 64], m![Qs % 512]> =
        q_weight_scale.to_dm(&mut device.tdma);
    let (q, q_scale) = (unsafe { q.reshape() }, unsafe { q_scale.reshape() });
    let (k, k_scale) = gather_kv_strips(device, &k, k_weight_scale);
    let (v, v_scale) = gather_kv_strips(device, &v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_iv(device, &v, &v_scale);
    (q, q_scale, k, k_scale, v)
}
