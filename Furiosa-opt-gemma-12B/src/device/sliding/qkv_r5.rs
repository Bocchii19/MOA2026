use furiosa_opt_std::prelude::*;

use super::rope::HeadCluster as Cluster;
use super::rope::KvHeadsAcrossSlices;
use crate::Chip;
use crate::axes::{Ds, Dummy8, Dummy256, Gs, H, Ps, Qs};

/// Q/K/V share one activation layout: 256 slices per cluster, every slice holding the whole
/// normalized hidden vector, and a slice's weights being a contiguous run of full rows.
///
/// The previous layout split K four ways (`m![Qs / 32 % 64, H / 960]`, 32 rows x 960 channels per
/// slice) so a slice's weights were 32 disjoint 960 B pieces strided through 3,840 B HBM rows:
/// the DMA ran at util 0.70-0.73 and the four K partials then needed an inter-slice reduce. Full
/// rows make a slice's weights one contiguous run (Q 8 x 3,840 B = 30 KiB, K/V 4 x 3,840 B =
/// 15 KiB) and delete the reduce. This is the layout `mlp.rs` already uses for the FFN matrices.
type QCluster = m![Qs / 2048];
type QRows = m![Qs / 8 % 256];
type QActivation = TrfTensor<f8e4m3, Chip, QCluster, QRows, m![1], m![Gs, H]>;
// Load one eight-slice activation group per cluster; broadcast only after RMS and hi/lo.
type QInputRows = m![1 # 32, H / 480];
// FP8 hi is finite for abs(normalized activation) <= 28 at scale 16.
// Both components are reconstructed exactly on the supplied PRNG fixture,
// but arbitrary BF16 activations are not guaranteed to be represented exactly.
const QKV_ACTIVATION_SCALE: f32 = 16.0;
fn split_qkv_activation(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, QCluster, QInputRows, m![H % 480]>,
) -> DmTensor<f8e4m3, Chip, QCluster, QInputRows, m![Gs, H % 480]> {
    type Rows = QInputRows;
    let mut out: DmTensor<f8e4m3, Chip, QCluster, Rows, m![Gs, H % 480]> = DmTensor::new();
    device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), QKV_ACTIVATION_SCALE)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, H % 480]>(0));
    // -hi (exact) as the VRF operand of the lo pass: `SubF` with a VRF operand evaluates
    // `vrf - stream` and a MulF may not follow it, so the lo pass adds the negated hi.
    let hi_neg: DmTensor<f8e4m3, Chip, QCluster, Rows, m![H % 480]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -QKV_ACTIVATION_SCALE)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    let hi_neg_vrf: VrfTensor<f32, Chip, QCluster, Rows, m![H % 480]> = device
        .sub
        .begin(hi_neg.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), QKV_ACTIVATION_SCALE)
        .vector_fp_binary(FpBinaryOp::AddF, &hi_neg_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, H % 480]>(1));
    out
}

type QContraction = DmTensor<bf16, Chip, QCluster, QRows, m![Qs % 8]>;

type KvCluster = m![Ps / 1024];
type KvRows = m![Ps / 4 % 256];
type KvActivation = TrfTensor<f8e4m3, Chip, KvCluster, KvRows, m![1], m![Gs, H]>;
type KvContraction = DmTensor<bf16, Chip, KvCluster, KvRows, m![Ps % 4]>;

/// Eight 480-channel shards on each cluster, before the ring-256 hi/lo broadcast.
pub(crate) type BroadcastInput =
    DmTensor<bf16, Chip, QCluster, QInputRows, m![H % 480]>;

fn broadcast_hilo(device: &mut Device, x: &BroadcastInput) -> DmTensor<f8e4m3, Chip, QCluster, QRows, m![Gs, H]> {
    let x = split_qkv_activation(device, x);
    let full: DmTensor<f8e4m3, Chip, QCluster, m![Dummy256], m![Gs, H]> = device.main
        .begin(x.view())
        .fetch::<m![Gs, H / 32 % 15], m![H % 32]>()
        .switch::<m![Dummy256], m![Gs, H / 32 % 15, H / 480]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![Gs, H / 32 % 15, H / 480], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    unsafe { full.reshape() }
}

/// One Q weight tile (rows `r0..r0+ROWS` of every slice's 8-row group, `ROWS x 3,840 B`
/// contiguous per slice) as its own DM tensor, and its contraction into those rows.
///
/// Tiles are separate live tensors with their own consumer chains, which is what makes the load
/// queue stream the next tile while the previous one is on Main; a single whole-matrix `to_dm` is
/// only issued after the preceding Main pass finishes (see the same note in `mlp.rs`). Tiles are
/// a multiple of 4 rows because the Transpose Engine's `in_rows` is capped at 4 for 16-bit
/// elements (`computing-tensors/transpose-engine.html`).
macro_rules! q_tile_fns {
    ($load:ident, $tile:ident, $rows:literal) => {
        fn $load(
            device: &mut Device,
            weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
            r0: usize,
        ) -> DmTensor<f8e4m3, Chip, QCluster, QRows, m![Qs % 8 = $rows, H]> {
            weight
                .view()
                .tile::<m![Qs % 8], $rows, m![Qs / 8, Qs % 8 = $rows # 8, H]>(r0)
                .to_dm(&mut device.tdma)
        }

        fn $tile(
            device: &mut Device,
            x: &QActivation,
            weight: DmTensor<f8e4m3, Chip, QCluster, QRows, m![Qs % 8 = $rows, H]>,
            contraction: &mut QContraction,
            r0: usize,
        ) {
            device.main
                .begin(weight.view())
                .fetch::<m![Qs % 8 = $rows, H / 32 % 120, Gs], m![H % 32]>()
                .collect::<m![Qs % 8 = $rows, H / 32 % 120, Gs], m![H % 32]>()
                .contract_outer::<m![Qs % 8 = $rows, H / 32 % 120, Gs], m![H % 32], _, _, _>(x)
                .contract_packet::<m![1]>()
                .contract_time::<m![Qs % 8 = $rows]>()
                .contract_lane::<m![Qs % 8 = $rows], m![1 # 8]>(LaneMode::Interleaved)
                // Restore activation scale before the original BF16 rounding boundary.
                // Per-channel weight scaling still happens after gather as before.
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .cast::<bf16, m![1 # 16]>()
                .transpose::<m![Qs % 8 = $rows / 4], m![Qs % 8 = $rows % 4 # 16]>()
                .commit_trim::<m![Qs % 8 = $rows % 4]>()
                .commit_view(contraction.view_mut().tile::<m![Qs % 8], $rows, m![Qs % 8 = $rows #{!} 8]>(r0));
        }
    };
}
q_tile_fns!(load_q_tile8, q_tile8, 8);

pub(crate) type QTile8 = DmTensor<f8e4m3, Chip, QCluster, QRows, m![Qs % 8 = 8, H]>;
pub(crate) type KvWeight = DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps % 4, H]>;

/// All three weight matrices, with independent contraction consumers. Source ordering alone
/// does not force DMA overlap: the compiler schedules their dataflow and SRAM lifetimes.
pub(crate) type Weights = (QTile8, KvWeight, KvWeight);

pub(crate) fn load_weights(
    device: &mut Device,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
) -> Weights {
    let q0 = load_q_tile8(device, q_weight, 0);
    let k = load_kv(device, k_weight);
    let v = load_kv(device, v_weight);
    (q0, k, v)
}

fn contract_query(
    device: &mut Device,
    x: &DmTensor<f8e4m3, Chip, QCluster, QRows, m![Gs, H]>,
    w0: QTile8,
) -> QContraction {
    let x_trf: QActivation = device
        .sub
        .begin(x.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let mut contraction: QContraction = DmTensor::new();
    q_tile8(device, &x_trf, w0, &mut contraction, 0);
    contraction
}

fn finish_query(
    device: &mut Device,
    contraction: QContraction,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> {
    // Gather four complete heads per cluster via Switch, then apply the scale there.
    // Source slice = head * 64 + within_head; TransposedBroadcast1 moves head to the
    // innermost output slice and within_head to Time. Only slices 0..3 are consumed.
    //
    // The gather is an exact BF16 move, so scaling after it is bit-identical to scaling on the
    // contraction mapping -- and far cheaper to feed. On the contraction mapping a slice holds
    // only 8 (Q) or 4 (K/V) scale values, so `weight_scale.to_dm` was 512 descriptors of
    // 8-16 B and cost 792-838 cycles for 4-8 KiB (util 0.005-0.009). On the head mapping a
    // slice holds 512 (Q) or 256 (K/V) contiguous values: eight descriptors instead of 512,
    // and 512 f32 = 2 KiB still fits the 8 KiB VRF (the one-slice layout would not). The BF16
    // rounding boundary before the scale is unchanged, which is what the reference expects.
    let gathered: DmTensor<bf16, Chip, QCluster, m![1 # 64, Qs / 512 % 4], m![Qs % 512]> =
        device.main
            .begin(contraction.view())
            .fetch::<m![1], m![Qs % 8 # 16]>()
            .switch::<m![1 # 64, Qs / 512 % 4], m![Qs / 8 % 64]>(
                SwitchConfig::TransposedBroadcast1 { slice1: 4, slice0: 64 },
            )
            .collect::<m![Qs / 8 % 64], m![Qs % 8 # 16]>()
            .commit_trim::<m![Qs % 8]>()
            .commit();
    let weight_scale: DmTensor<bf16, Chip, QCluster, m![1 # 64, Qs / 512 % 4], m![Qs % 512]> =
        weight_scale.to_dm(&mut device.tdma);
    let weight_scale: VrfTensor<f32, Chip, QCluster, m![1 # 64, Qs / 512 % 4], m![Qs % 512]> = device
        .sub
        .begin(weight_scale.view())
        .fetch::<m![Qs / 16 % 32], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 64], m![Qs % 8]>()
        .to_vrf();
    let scaled: DmTensor<bf16, Chip, QCluster, m![1 # 64, Qs / 512 % 4], m![Qs % 512]> = device
        .main
        .begin(gathered.view())
        .fetch::<m![Qs / 16 % 32], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 64], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 128], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale)
        .vector_widen_concat::<m![Qs / 8 % 64], m![Qs % 8]>()
        .vector_final()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();

    unsafe { scaled.reshape() }
}

/// One whole K or V matrix: with only four rows per slice there is nothing left to tile, but the
/// two matrices are still issued as separate live tensors ahead of either contraction.
fn load_kv(
    device: &mut Device,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
) -> DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps % 4, H]> {
    weight.to_dm(&mut device.tdma)
}

fn contract_one_kv_matrix(
    device: &mut Device,
    x: &KvActivation,
    weight: DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps % 4, H]>,
) -> KvContraction {
    device.main
        .begin(weight.view())
        .fetch::<m![Ps % 4, H / 32 % 120, Gs], m![H % 32]>()
        .collect::<m![Ps % 4, H / 32 % 120, Gs], m![H % 32]>()
        .contract_outer::<m![Ps % 4, H / 32 % 120, Gs], m![H % 32], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ps % 4 / 4], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit()
}

fn finish_one_kv_matrix(
    device: &mut Device,
    contraction: KvContraction,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, KvCluster, m![1 # 64, Ps / 256 % 4], m![Ps % 256]> {
    // Same logical mapping as `finish_query`: four heads per cluster, each made from
    // 64 source slices, gathered by Switch before the per-channel scale.
    let gathered: DmTensor<bf16, Chip, KvCluster, m![1 # 64, Ps / 256 % 4], m![Ps % 256]> =
        device.main
            .begin(contraction.view())
            .fetch::<m![1], m![Ps % 4 # 16]>()
            .switch::<m![1 # 64, Ps / 256 % 4], m![Ps / 4 % 64]>(
                SwitchConfig::TransposedBroadcast1 { slice1: 4, slice0: 64 },
            )
            .collect::<m![Ps / 4 % 64], m![Ps % 4 # 16]>()
            .commit_trim::<m![Ps % 4]>()
            .commit();
    let weight_scale: DmTensor<bf16, Chip, KvCluster, m![1 # 64, Ps / 256 % 4], m![Ps % 256]> =
        weight_scale.to_dm(&mut device.tdma);
    let weight_scale: VrfTensor<f32, Chip, KvCluster, m![1 # 64, Ps / 256 % 4], m![Ps % 256]> = device
        .sub
        .begin(weight_scale.view())
        .fetch::<m![Ps / 16 % 16], m![Ps % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ps / 8 % 32], m![Ps % 8]>()
        .to_vrf();
    device.main
        .begin(gathered.view())
        .fetch::<m![Ps / 16 % 16], m![Ps % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ps / 8 % 32], m![Ps % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ps / 4 % 64], m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale)
        .vector_widen_concat::<m![Ps / 8 % 32], m![Ps % 8]>()
        .vector_final()
        .cast::<bf16, m![Ps % 8 # 16]>()
        .commit_trim::<m![Ps % 8]>()
        .commit()
}

/// V: the gathered heads and their per-channel scale, both relabelled to the head layout, so the
/// scale, RMS and normalization can run as one three-pass chain (`qkv_r5_norm::normalize_value_scaled`).
fn gather_value(
    device: &mut Device,
    contraction: KvContraction,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let gathered: DmTensor<bf16, Chip, KvCluster, m![1 # 64, Ps / 256 % 4], m![Ps % 256]> =
        device.main
            .begin(contraction.view())
            .fetch::<m![1], m![Ps % 4 # 16]>()
            .switch::<m![1 # 64, Ps / 256 % 4], m![Ps / 4 % 64]>(
                SwitchConfig::TransposedBroadcast1 { slice1: 4, slice0: 64 },
            )
            .collect::<m![Ps / 4 % 64], m![Ps % 4 # 16]>()
            .commit_trim::<m![Ps % 4]>()
            .commit();
    let weight_scale: DmTensor<bf16, Chip, KvCluster, m![1 # 64, Ps / 256 % 4], m![Ps % 256]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

fn contract_key_value(
    device: &mut Device,
    x: DmTensor<f8e4m3, Chip, QCluster, QRows, m![Gs, H]>,
    k_weight: KvWeight,
    v_weight: KvWeight,
) -> (KvContraction, KvContraction) {
    // Q and K/V shard both clusters over 256 slices holding the whole hidden vector, so the
    // K/V activation view is a relabelling of the Q one, not another broadcast.
    let x: DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Gs, H]> = unsafe { x.reshape() };
    let x_trf: KvActivation = device
        .sub
        .begin(x.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();

    let k = contract_one_kv_matrix(device, &x_trf, k_weight);
    let v = contract_one_kv_matrix(device, &x_trf, v_weight);
    (k, v)
}

pub(crate) fn project(
    device: &mut Device,
    x: &BroadcastInput,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    // Keep contractions and epilogues separate; each contraction returns only a small tensor.
    // Finish on four heads per cluster, preserving the native cluster ownership throughout
    // normalization, RoPE and output stores (no gather-through-HBM between these stages).
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo(device, x);
    let q = contract_query(device, &xq, q0);
    let (k, v) = contract_key_value(device, xq, k_weight, v_weight);

    let q = finish_query(device, q, q_weight_scale);
    let k = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value(device, v, v_weight_scale);
    let v = super::qkv_r5_norm::normalize_value_scaled(device, &v, &v_scale);
    (q, unsafe { k.reshape() }, v)
}

/// Normalize independently in each cluster's eight-slice seed group. The RMS reduction,
/// division, weight multiplication and BF16 rounding match shared::rmsnorm::normalize.
/// Returning the shards directly avoids gathering to a slice and staging through HBM.
pub(crate) fn normalize_input(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> BroadcastInput {
    type ReducingSlices = QInputRows;

    let x: DmTensor<bf16, Chip, QCluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut device.tdma);

    let reduced_mean_square: DmTensor<f32, Chip, QCluster, m![1 # 32, Dummy8], m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, QCluster, m![1 # 32, Dummy8], m![1 # 8]> = device
        .main
        .begin(reduced_mean_square.view())
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
    let rms: DmTensor<f32, Chip, QCluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, QCluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut device.tdma);
    let weight_vrf: VrfTensor<f32, Chip, QCluster, ReducingSlices, m![H % 480]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, QCluster, ReducingSlices, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, QCluster, ReducingSlices, m![H % 480]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    normalized
}
