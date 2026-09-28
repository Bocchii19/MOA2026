use furiosa_opt_std::prelude::*;

use super::rope_h4::HeadCluster as Cluster;
use super::rope_h4::KvHeadsAcrossSlices;
use crate::Chip;
use crate::axes::{Ds, Dummy8, Gs, H, Ps, Qs};

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

// FP8 hi is finite for abs(normalized activation) <= 28 at scale 16.
// Both components are reconstructed exactly on the supplied PRNG fixture,
// but arbitrary BF16 activations are not guaranteed to be represented exactly.
const QKV_ACTIVATION_SCALE: f32 = 16.0;

type QContraction = DmTensor<bf16, Chip, QCluster, QRows, m![Qs / 64 % 8]>;

type KvCluster = m![Ps / 1024];

type KvRows = m![Ps / 256 % 4, Ps % 64];

type KvContraction = DmTensor<bf16, Chip, KvCluster, KvRows, m![Ps / 64 % 4]>;

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

/// One whole K or V matrix: with only four rows per slice there is nothing left to tile, but the
/// two matrices are still issued as separate live tensors ahead of either contraction.
fn load_kv(
    device: &mut Device,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
) -> DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps / 64 % 4, H]> {
    weight.to_dm(&mut device.tdma)
}

/// K: gathered heads plus the per-channel scale (applied inside `normalize_key_scaled`).
/// One block-transpose gather pass (`gather_value_b`) instead of gather + transpose: the K chain ends
/// ~0.8k earlier under the Q stream. On hardware the Q -> V transition is fast (~1.3k from the last Q
/// byte to the first V byte) only when the K chain finishes >= ~0.9k before the Q stream ends; later
/// than that the q scale DMA and the Q contraction stall and V starts 2.5-4.4k late (span traces
/// sy4_ctl_tr / sy4_v5_tr). It also keeps the static scheduler listing the whole K chain before the
/// Q contraction once the V tail loses its sqrt pass (`normalize_value_scaled_iv`); without it the
/// K norm/rope passes are deferred into the V window and K1 loses ~1.8k.
fn finish_one_kv_matrix(
    device: &mut Device,
    contraction: KvContraction,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    gather_value_b(device, contraction, weight_scale)
}

/// K: ring-64 all-gather (`Broadcast1{64, 1}`) of each head's 64 contraction slices onto the first
/// slice of its PE (`rope_h4::KvHeadsAcrossSlices = [Ns % 4, 1 # 64]`), then the channel transpose.
/// The v7 gather (`TransposedBroadcast1{4, 64}` onto slices 0..3) was a ring-256 pass: 2.43k HW vs 0.79k.
pub(crate) fn gather_value(
    device: &mut Device,
    contraction: KvContraction,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let gathered_p: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 64, Ps / 64 % 4]> =
        device.main
            .begin(contraction.view())
            .fetch::<m![1], m![Ps / 64 % 4 # 16]>()
            .switch::<m![Ps / 256 % 4, 1 # 64], m![Ps % 64]>(
                SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 },
            )
            .collect::<m![Ps % 64], m![Ps / 64 % 4 # 16]>()
            .commit_trim::<m![Ps / 64 % 4]>()
            .commit();
    let gathered: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> = device
        .main
        .begin(gathered_p.view())
        .fetch::<m![Ps / 64 % 4, Ps % 64], m![1 # 16]>()
        .collect::<m![Ps / 64 % 4, Ps % 64], m![1 # 16]>()
        .transpose::<m![Ps % 256 / 4], m![Ps % 256 % 4 # 16]>()
        .commit_trim::<m![Ps % 256 % 4]>()
        .commit();
    let weight_scale: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

/// K/V weights relabelled onto the Q axes (same cluster/slice/element extents), so one TRF load of
/// the activation serves all three contractions.
type KvWeightQ = DmTensor<f8e4m3, Chip, QCluster, QRows, m![Qs / 64 % 4, H]>;

type KvContractionQ = DmTensor<bf16, Chip, QCluster, QRows, m![Qs / 64 % 4]>;

pub(crate) fn contract_one_kv_matrix_q(device: &mut Device, x: &QActivation, weight: KvWeight) -> KvContraction {
    let weight: KvWeightQ = unsafe { weight.reshape() };
    let out: KvContractionQ = device.main
        .begin(weight.view())
        .fetch::<m![Qs / 64 % 4, H / 32 % 120, Gs], m![H % 32]>()
        .collect::<m![Qs / 64 % 4, H / 32 % 120, Gs], m![H % 32]>()
        .contract_outer::<m![Qs / 64 % 4, H / 32 % 120, Gs], m![H % 32], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 64 % 4]>()
        .contract_lane::<m![Qs / 64 % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 64 % 4 / 4], m![Qs / 64 % 4 # 16]>()
        .commit_trim::<m![Qs / 64 % 4]>()
        .commit();
    unsafe { out.reshape() }
}

/// `contract_one_kv_matrix_q` with full 64 B packets: two 32 B weight flits are packed into one
/// packet (`H / 32 % 2` absorbed into `OutPacket`) and the hi/lo activation pair (`Gs`) is a TRF-only
/// time broadcast, so each weight flit is fetched once and every multiply uses the whole 64 B MAC
/// row (the 32 B form fetches the weight twice and leaves half the MACs multiplying zeros).
pub(crate) fn contract_one_kv_matrix_q64(device: &mut Device, x: &QActivation, weight: KvWeight) -> KvContraction {
    let weight: KvWeightQ = unsafe { weight.reshape() };
    let out: KvContractionQ = device.main
        .begin(weight.view())
        .fetch::<m![Qs / 64 % 4, H / 64 % 60, H / 32 % 2], m![H % 32]>()
        .collect::<m![Qs / 64 % 4, H / 64 % 60, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Qs / 64 % 4, H / 64 % 60, Gs], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 64 % 4]>()
        .contract_lane::<m![Qs / 64 % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 64 % 4 / 4], m![Qs / 64 % 4 # 16]>()
        .commit_trim::<m![Qs / 64 % 4]>()
        .commit();
    unsafe { out.reshape() }
}

/// V gathered straight into natural channel order: the contraction output is fetched one channel per
/// packet with its 4 rows in time, the ring-64 all-gather appends the 64 slices innermost
/// (time = [row group j, slice s] = channel 64 j + s), and the Transpose Engine packs 4 steps per
/// packet. One pass instead of gather + transpose.
pub(crate) fn gather_value_f(
    device: &mut Device,
    contraction: KvContraction,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let gathered: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> = device
        .main
        .begin(contraction.view())
        .fetch::<m![Ps / 64 % 4], m![1 # 16]>()
        .switch::<m![Ps / 256 % 4, 1 # 64], m![Ps / 64 % 4, Ps % 64]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ps / 64 % 4, Ps % 64], m![1 # 16]>()
        .transpose::<m![Ps % 256 / 4], m![Ps % 256 % 4 # 16]>()
        .commit_trim::<m![Ps % 256 % 4]>()
        .commit();
    let weight_scale: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

/// V gathered in 64 ring steps with a block transpose into natural channel order: each contraction
/// slice sends its 4 rows as ONE packet (`Ps / 64 % 4` in the packet), the ring-64 all-gather puts
/// the 64 slices in time (`[s]`, packet `[j]`), and the Transpose Engine turns every 4 consecutive
/// slices x 4 rows block into 4 packets of 4 consecutive channels (time `[s / 4, j]`, packet
/// `[s % 4]`), which the commit writes at channel `64 j + s`. A quarter of the ring steps of
/// `gather_value_f` (256 one-element steps), no separate transpose pass (`gather_value`).
pub(crate) fn gather_value_b(
    device: &mut Device,
    contraction: KvContraction,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let gathered: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> = device
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Ps / 64 % 4 # 16]>()
        .switch::<m![Ps / 256 % 4, 1 # 64], m![Ps % 64]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ps % 64], m![Ps / 64 % 4 # 16]>()
        .transpose::<m![Ps % 64 / 4, Ps / 64 % 4], m![Ps % 64 % 4 # 16]>()
        .commit_trim::<m![Ps % 64 % 4]>()
        .commit();
    let weight_scale: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

// x and the input RMS weight replicated 8x instead of 32x: one valid eight-slice shard group per 32
// slices (`1 # 4` between the group and the shard axis; the inter-slice reduce needs its shard axis
// innermost), so each replicated load writes a quarter of the bytes (2.75k -> ~1.5k HW each). A
// `Broadcast01{4, 8, 30}` then gives every slice all eight shards; the three padding slices' packets
// land in the `1 # 4` element padding of the result, which the TRF load skips.
axes![QkvQuad = 4];

type XRows8 = m![Qs / 256 % 8, 1 # 4, H / 480];

pub(crate) type BroadcastInput8 = DmTensor<bf16, Chip, QCluster, XRows8, m![H % 480]>;

type XHilo8 = DmTensor<f8e4m3, Chip, QCluster, QRows, m![1 # 4, Gs, H / 32 % 15, H / 480, H % 32]>;

fn split_qkv_activation8(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, QCluster, XRows8, m![H % 480]>,
) -> DmTensor<f8e4m3, Chip, QCluster, XRows8, m![Gs, H % 480]> {
    type Rows = XRows8;
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

/// Input RMSNorm on the 8x-replicated layout. The rms weight is loaded first and the mean square adds
/// `sum(0 * w)` (exactly 0 for finite weights): that data dependency makes the scheduler list the K
/// weight load right after x instead of after the whole Sub RMS chain (K streams from x landing).
pub(crate) fn normalize_input_x8w(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> BroadcastInput8 {
    type ReducingSlices = XRows8;

    // The rms weight is loaded first and the mean-square adds `sum(0 * w)` (exactly 0 for finite w): the
    // scheduler then lists the K weight load right after x instead of after the whole RMS chain.
    let weight_dm: DmTensor<bf16, Chip, QCluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut device.tdma);
    let zero: DmTensor<f32, Chip, QCluster, ReducingSlices, m![1 # 8]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let zero_vrf: VrfTensor<f32, Chip, QCluster, ReducingSlices, m![1 # 8]> = device
        .sub
        .begin(zero.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let x: DmTensor<bf16, Chip, QCluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut device.tdma);
    let reduced_mean_square: DmTensor<f32, Chip, QCluster, m![Qs / 256 % 8, 1 # 4, Dummy8], m![1 # 8]> = device
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
        .vector_fp_binary(FpBinaryOp::AddF, &zero_vrf)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Qs / 256 % 8, 1 # 4, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, QCluster, m![Qs / 256 % 8, 1 # 4, Dummy8], m![1 # 8]> = device
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

fn broadcast_hilo8(device: &mut Device, x: &BroadcastInput8) -> XHilo8 {
    let x = split_qkv_activation8(device, x);
    let full: DmTensor<f8e4m3, Chip, QCluster, m![Qs / 256 % 8, QkvQuad, Dummy8], m![1 # 4, Gs, H / 32 % 15, H / 480, H % 32]> =
        device.main
            .begin(x.view())
            .fetch::<m![Gs, H / 32 % 15], m![H % 32]>()
            .switch::<m![Qs / 256 % 8, QkvQuad, Dummy8], m![1 # 4, Gs, H / 32 % 15, H / 480]>(
                SwitchConfig::Broadcast01 { slice1: 4, slice0: 8, time0: 30 },
            )
            .collect::<m![1 # 4, Gs, H / 32 % 15, H / 480], m![H % 32]>()
            .commit_trim::<m![H % 32]>()
            .commit();
    unsafe { full.reshape() }
}

/// Q gathered straight into natural channel order (see `gather_value_f`): 8 rows in time, one
/// channel per packet, ring-64 all-gather appends the 64 slices innermost.
pub(crate) fn finish_query_f(
    device: &mut Device,
    contraction: QContraction,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
) {
    let gathered: DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, 1 # 64], m![Qs % 512]> = device
        .main
        .begin(contraction.view())
        .fetch::<m![Qs / 64 % 8], m![1 # 16]>()
        .switch::<m![Qs / 512 % 4, 1 # 64], m![Qs / 64 % 8, Qs % 64]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Qs / 64 % 8, Qs % 64], m![1 # 16]>()
        .transpose::<m![Qs % 512 / 4], m![Qs % 512 % 4 # 16]>()
        .commit_trim::<m![Qs % 512 % 4]>()
        .commit();
    let weight_scale: DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, 1 # 64], m![Qs % 512]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

/// Q, K, V projections from one TRF load (K/V weights relabelled onto the Q axes), heads gathered
/// onto one slice per PE, Q and V gathered straight into channel order, V normalized with its scale
/// folded in and the rms written to VRF by the sqrt pass.
pub(crate) fn project_x8g(
    device: &mut Device,
    x: &BroadcastInput8,
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
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let q = q_tile8(device, &x_trf, q0);
    let k = contract_one_kv_matrix_q(device, &x_trf, k_weight);
    let v = contract_one_kv_matrix_q64(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let (k, k_scale) = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value_b(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_iv(device, &v, &v_scale);
    (q, q_scale, k, k_scale, v)
}
