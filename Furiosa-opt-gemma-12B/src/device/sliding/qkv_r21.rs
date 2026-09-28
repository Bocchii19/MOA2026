use furiosa_opt_std::prelude::*;

use super::rope::HeadCluster as Cluster;
use super::rope::KvHeadsAcrossSlices;
use crate::Chip;
use crate::axes::{Ds, Dummy2, Dummy8, Gs, H, Ps, Qs};

/// Q/K/V share one activation layout: 256 slices per cluster, every slice holding the whole
/// normalized hidden vector, and a slice's weights being a contiguous run of full rows.
///
/// The previous layout split K four ways (`m![Qs / 32 % 64, H / 960]`, 32 rows x 960 channels per
/// slice) so a slice's weights were 32 disjoint 960 B pieces strided through 3,840 B HBM rows:
/// the DMA ran at util 0.70-0.73 and the four K partials then needed an inter-slice reduce. Full
/// rows make a slice's weights one contiguous run (Q 8 x 3,840 B = 30 KiB, K/V 4 x 3,840 B =
/// 15 KiB) and delete the reduce. This is the layout `mlp.rs` already uses for the FFN matrices.
type QCluster = m![Qs / 2048];
/// Head-grouped strided rows: head h (512 rows) lives on slice group h; inside a head, adjacent
/// rows sit on adjacent slices (row r -> slice 64 * (r / 512 % 4) + r % 64, element r / 64 % 8),
/// so concurrently reading slices touch adjacent HBM rows: 602 B/cycle vs 505 for 8 consecutive
/// rows per slice (experiments/qkv-lab/RESULT.md, jobs 73542/73583/73595).
type QRows = m![Qs / 512 % 4, Qs % 64];
type QActivation = TrfTensor<f8e4m3, Chip, QCluster, QRows, m![1], m![Gs, H]>;
// x is loaded replicated onto all 32 eight-slice groups of each cluster (the slice axis
// `Qs / 64 % 32` is absent from the `[H]` source, so the DMA replicates: 32 x 7.5 KiB per
// cluster). RMS and the hi/lo split then run on all 256 slices at the same per-pass cost as on
// 8, and the broadcast that follows is a ring-8 inside each group instead of a ring-256 across
// the cluster: 8,455 -> ~1,400 static, 17k -> ~3k on hardware (the longest pass of v11, see
// experiments/dma-docs/hw/qkv_tr2_job71711.log: `qkv_r5.rs:102 switch` 23,805..40,933).
type QInputRows = m![Qs / 64 % 32, H / 480];
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

type QContraction = DmTensor<bf16, Chip, QCluster, QRows, m![Qs / 64 % 8]>;

type KvCluster = m![Ps / 1024];
type KvRows = m![Ps / 256 % 4, Ps % 64];
type KvActivation = TrfTensor<f8e4m3, Chip, KvCluster, KvRows, m![1], m![Gs, H]>;
type KvContraction = DmTensor<bf16, Chip, KvCluster, KvRows, m![Ps / 64 % 4]>;

/// Eight 480-channel shards on each cluster, before the ring-256 hi/lo broadcast.
pub(crate) type BroadcastInput =
    DmTensor<bf16, Chip, QCluster, QInputRows, m![H % 480]>;

fn broadcast_hilo(device: &mut Device, x: &BroadcastInput) -> DmTensor<f8e4m3, Chip, QCluster, QRows, m![Gs, H]> {
    let x = split_qkv_activation(device, x);
    // All-gather of the eight shards inside each eight-slice group (named config: no config DMA).
    let full: DmTensor<f8e4m3, Chip, QCluster, m![Qs / 64 % 32, Dummy8], m![Gs, H]> = device.main
        .begin(x.view())
        .fetch::<m![Gs, H / 32 % 15], m![H % 32]>()
        .switch::<m![Qs / 64 % 32, Dummy8], m![Gs, H / 32 % 15, H / 480]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![Gs, H / 32 % 15, H / 480], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    // Slice (g, d) of the ring layout is slice 8g + d of the head-grouped layout; all hold all of x.
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
fn load_q_tile8(device: &mut Device, weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>) -> QTile8 {
    weight.to_dm(&mut device.tdma)
}

fn q_tile8(device: &mut Device, x: &QActivation, weight: QTile8) -> QContraction {
    device.main
        .begin(weight.view())
        .fetch::<m![Qs / 64 % 8, H / 32 % 120, Gs], m![H % 32]>()
        .collect::<m![Qs / 64 % 8, H / 32 % 120, Gs], m![H % 32]>()
        .contract_outer::<m![Qs / 64 % 8, H / 32 % 120, Gs], m![H % 32], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 64 % 8]>()
        .contract_lane::<m![Qs / 64 % 8], m![1 # 8]>(LaneMode::Interleaved)
        // Restore activation scale before the original BF16 rounding boundary.
        // Per-channel weight scaling still happens after gather as before.
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 64 % 8 / 4], m![Qs / 64 % 8 % 4 # 16]>()
        .commit_trim::<m![Qs / 64 % 8 % 4]>()
        .commit()
}

pub(crate) type QTile8 = DmTensor<f8e4m3, Chip, QCluster, QRows, m![Qs / 64 % 8, H]>;
pub(crate) type KvWeight = DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps / 64 % 4, H]>;

/// All three weight matrices, with independent contraction consumers. Source ordering alone
/// does not force DMA overlap: the compiler schedules their dataflow and SRAM lifetimes.
pub(crate) type Weights = (QTile8, KvWeight, KvWeight);

pub(crate) fn load_weights(
    device: &mut Device,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
) -> Weights {
    let q0 = load_q_tile8(device, q_weight);
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
    q_tile8(device, &x_trf, w0)
}

/// Gathered heads plus the per-channel scale, both on the head mapping; the scale is applied
/// inside the head RMSNorm (`qkv_r21_norm::normalize_query_scaled`), which saves the scale pass.
fn finish_query(
    device: &mut Device,
    contraction: QContraction,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
) {
    let gathered_p: DmTensor<bf16, Chip, QCluster, m![1 # 64, Qs / 512 % 4], m![Qs % 64, Qs / 64 % 8]> =
        device.main
            .begin(contraction.view())
            .fetch::<m![1], m![Qs / 64 % 8 # 16]>()
            .switch::<m![1 # 64, Qs / 512 % 4], m![Qs % 64]>(
                SwitchConfig::TransposedBroadcast1 { slice1: 4, slice0: 64 },
            )
            .collect::<m![Qs % 64], m![Qs / 64 % 8 # 16]>()
            .commit_trim::<m![Qs / 64 % 8]>()
            .commit();
    let gathered: DmTensor<bf16, Chip, QCluster, m![1 # 64, Qs / 512 % 4], m![Qs % 512]> = device
        .main
        .begin(gathered_p.view())
        .fetch::<m![Qs / 64 % 8, Qs % 64], m![1 # 16]>()
        .collect::<m![Qs / 64 % 8, Qs % 64], m![1 # 16]>()
        .transpose::<m![Qs % 512 / 4], m![Qs % 512 % 4 # 16]>()
        .commit_trim::<m![Qs % 512 % 4]>()
        .commit();
    let weight_scale: DmTensor<bf16, Chip, QCluster, m![1 # 64, Qs / 512 % 4], m![Qs % 512]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

/// One whole K or V matrix: with only four rows per slice there is nothing left to tile, but the
/// two matrices are still issued as separate live tensors ahead of either contraction.
fn load_kv(
    device: &mut Device,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
) -> DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps / 64 % 4, H]> {
    weight.to_dm(&mut device.tdma)
}

fn contract_one_kv_matrix(
    device: &mut Device,
    x: &KvActivation,
    weight: DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps / 64 % 4, H]>,
) -> KvContraction {
    device.main
        .begin(weight.view())
        .fetch::<m![Ps / 64 % 4, H / 32 % 120, Gs], m![H % 32]>()
        .collect::<m![Ps / 64 % 4, H / 32 % 120, Gs], m![H % 32]>()
        .contract_outer::<m![Ps / 64 % 4, H / 32 % 120, Gs], m![H % 32], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps / 64 % 4]>()
        .contract_lane::<m![Ps / 64 % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ps / 64 % 4 / 4], m![Ps / 64 % 4 # 16]>()
        .commit_trim::<m![Ps / 64 % 4]>()
        .commit()
}

/// K: gathered heads plus the per-channel scale (applied inside `normalize_key_scaled`).
fn finish_one_kv_matrix(
    device: &mut Device,
    contraction: KvContraction,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    gather_value(device, contraction, weight_scale)
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
    let gathered_p: DmTensor<bf16, Chip, KvCluster, m![1 # 64, Ps / 256 % 4], m![Ps % 64, Ps / 64 % 4]> =
        device.main
            .begin(contraction.view())
            .fetch::<m![1], m![Ps / 64 % 4 # 16]>()
            .switch::<m![1 # 64, Ps / 256 % 4], m![Ps % 64]>(
                SwitchConfig::TransposedBroadcast1 { slice1: 4, slice0: 64 },
            )
            .collect::<m![Ps % 64], m![Ps / 64 % 4 # 16]>()
            .commit_trim::<m![Ps / 64 % 4]>()
            .commit();
    let gathered: DmTensor<bf16, Chip, KvCluster, m![1 # 64, Ps / 256 % 4], m![Ps % 256]> = device
        .main
        .begin(gathered_p.view())
        .fetch::<m![Ps / 64 % 4, Ps % 64], m![1 # 16]>()
        .collect::<m![Ps / 64 % 4, Ps % 64], m![1 # 16]>()
        .transpose::<m![Ps % 256 / 4], m![Ps % 256 % 4 # 16]>()
        .commit_trim::<m![Ps % 256 % 4]>()
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
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
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

    let (q, q_scale) = finish_query(device, q, q_weight_scale);
    let (k, k_scale) = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value(device, v, v_weight_scale);
    let v = super::qkv_r21_norm::normalize_value_scaled(device, &v, &v_scale);
    (q, q_scale, k, k_scale, v)
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
    let reduced_mean_square: DmTensor<f32, Chip, QCluster, m![Qs / 64 % 32, Dummy8], m![1 # 8]> = device
        .sub
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
        .vector_inter_slice_reduce::<m![Qs / 64 % 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, QCluster, m![Qs / 64 % 32, Dummy8], m![1 # 8]> = device
        .sub
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
