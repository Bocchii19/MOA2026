use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{C, Ds, Dummy2, Dummy8, E, Gf, Gs, Ns};
/// Four KV heads (and their two Q groups) remain in each projection cluster.
pub(crate) type HeadCluster = m![Ns / 4];
type Cluster = HeadCluster;

pub(crate) type HeadSlices = m![Ns % 4, 1 # 64];

/// The selected cos and sin rows, `[cos, sin]`, on the four head slices of both clusters.
pub(crate) type RopeRows = DmTensor<bf16, Chip, Cluster, HeadSlices, m![Dummy2, Ds]>;

/// RoPE rows computed on chip (step 40). The `cos`/`sin` inputs are the model's standard
/// theta = 10,000 tables, so the row a position selects is a pure function of that position:
/// `cos_row[i] = cos(POS * f(i))`, `f(i) = 10000^(-(i mod 128) / 128)`, and
/// `sin_row[i] = -/+ sin(POS * f(i))` with the low half negated (the `rotate_half` convention the
/// reference bakes into its sin table). Computing them here replaces two `dma_gather`s, an HBM
/// staging store/load pair and the random write->read sync they carried (ban 02 hardware trace:
/// 6.6k cycles of DMA lane plus 1-20k of sync, then 5k of RoPE waiting on the tail) with one 4-byte
/// load of `rope_offset` and ~22 tiny Main passes on the four head slices, which run while Main is
/// otherwise idle under the weight loads.
///
/// The ramp `f(i)` is built without any index primitive (the toolchain does not lower
/// `TagMode::AxisToggle`): eight doublings of a geometric sequence, each writing `[v, v * r^n]` into
/// the two tiles of a length-doubled tensor, in a one-value-per-flit `[N, 1 # 8]` layout, then one
/// Transpose pass packs the 256 flits into a dense `[Ds]` vector. The last doubling uses factor 1
/// because `f` repeats over the two halves. Precision: <= 8 f32 multiplies per value (~5e-7
/// relative), and the results are rounded to bf16 exactly as the tables are.
type RopeF32<E> = DmTensor<f32, Chip, Cluster, HeadSlices, E>;
const ROPE_RATIO_1: f32 = 0.930_572_032_9;
const ROPE_RATIO_2: f32 = 0.865_964_353_1;
const ROPE_RATIO_4: f32 = 0.749_894_201_8;
const ROPE_RATIO_8: f32 = 0.562_341_332_4;
const ROPE_RATIO_16: f32 = 0.316_227_763_9;
const ROPE_RATIO_32: f32 = 0.1;
const ROPE_RATIO_64: f32 = 0.01;
/// `rope_offset` is a byte offset (`POS * 256 * 2`); read as a Q24.7 fixed point by
/// `vector_fxp_to_fp(24)` (value = raw * 2^(24 - 31)) it is `4 * POS`.
const ROPE_OFFSET_TO_POS: f32 = 0.25;
/// Range reduction constants in the same Q24.7 fixed point (`int_width` 24, 7 fraction bits):
/// `turns = round(angle / 2pi * 128)` is stored as an i32 whose value read back is `n` (exact),
/// scaled back by `-2pi * 128`... i.e. `angle * (1 / 2pi) / 128` goes in, `raw / 128 * (-2pi * 128)`
/// comes out.
const ROPE_INV_TWO_PI_Q24: f32 = 0.001_243_398_0;
const ROPE_NEG_TWO_PI_Q24: f32 = -804.247_719_3;
const ROPE_TURNS_INT_WIDTH: u32 = 24;

macro_rules! rope_ramp_double {
    ($name:ident, $in_time:ty, $in_elem:ty, $mid_elem:ty, $mid_tile:ty, $out_elem:ty) => {
        /// `[v, v * factor]`: the first tile copies `v`, the second scales it, and the doubled
        /// tensor is relabelled to its canonical length axis (a pure regrouping of the wire order).
        fn $name(device: &mut Device, v: &RopeF32<$in_elem>, factor: f32) -> RopeF32<$out_elem> {
            let mut mid: RopeF32<$mid_elem> = DmTensor::new();
            device.main
                .begin(v.view())
                .fetch::<$in_time, m![1 # 8]>()
                .collect::<$in_time, m![1 # 8]>()
                .commit_trim::<m![1 # 8]>()
                .commit_view(mid.view_mut().tile::<m![Dummy2], 1, $mid_tile>(0));
            device.main
                .begin(v.view())
                .fetch::<$in_time, m![1 # 8]>()
                .collect::<$in_time, m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), factor)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit_view(mid.view_mut().tile::<m![Dummy2], 1, $mid_tile>(1));
            unsafe { mid.reshape() }
        }
    };
}
rope_ramp_double!(rope_ramp_2, m![1], m![1 # 8], m![Dummy2, 1 # 8], m![Dummy2 = 1 #{!} 2, 1 # 8], m![Gs, 1 # 8]);
rope_ramp_double!(rope_ramp_4, m![Gs], m![Gs, 1 # 8], m![Dummy2, Gs, 1 # 8], m![Dummy2 = 1 #{!} 2, Gs, 1 # 8], m![Dummy8 / 2, 1 # 8]);
rope_ramp_double!(rope_ramp_8, m![Dummy8 / 2], m![Dummy8 / 2, 1 # 8], m![Dummy2, Dummy8 / 2, 1 # 8], m![Dummy2 = 1 #{!} 2, Dummy8 / 2, 1 # 8], m![Dummy8, 1 # 8]);
rope_ramp_double!(rope_ramp_16, m![Dummy8], m![Dummy8, 1 # 8], m![Dummy2, Dummy8, 1 # 8], m![Dummy2 = 1 #{!} 2, Dummy8, 1 # 8], m![Gf, 1 # 8]);
rope_ramp_double!(rope_ramp_32, m![Gf], m![Gf, 1 # 8], m![Dummy2, Gf, 1 # 8], m![Dummy2 = 1 #{!} 2, Gf, 1 # 8], m![C / 2, 1 # 8]);
rope_ramp_double!(rope_ramp_64, m![C / 2], m![C / 2, 1 # 8], m![Dummy2, C / 2, 1 # 8], m![Dummy2 = 1 #{!} 2, C / 2, 1 # 8], m![C, 1 # 8]);
rope_ramp_double!(rope_ramp_128, m![C], m![C, 1 # 8], m![Dummy2, C, 1 # 8], m![Dummy2 = 1 #{!} 2, C, 1 # 8], m![Ds / 2, 1 # 8]);
rope_ramp_double!(rope_ramp_256, m![Ds / 2], m![Ds / 2, 1 # 8], m![Dummy2, Ds / 2, 1 # 8], m![Dummy2 = 1 #{!} 2, Ds / 2, 1 # 8], m![Ds, 1 # 8]);

/// 1/Ds: summed over the Ds channels of a head it is exactly 1.0.
const ROPE_ONE_OVER_DS: f32 = 0.00390625;

pub(crate) fn stage_rope_rows(
    device: &mut Device,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    after: &DmTensor<f32, Chip, Cluster, HeadSlices, m![Ds]>,
) -> RopeRows {
    // `0.0 * sum(after)`: a data dependency and nothing else. Hardware issues commands in the
    // static schedule's order, one context at a time, so the tiny passes below -- which the
    // scheduler would otherwise place at the very head of the list, where they run 3-12x slower
    // than modelled -- held back the issue of the K / Q / V weight loads behind them.
    // `sum(0 * k + 1/256) = 1.0` on every head slice: the data dependency on the finished K that
    // keeps the chain off the head of the list (it would hold back the weight issue there), now
    // entering at `one` so that `pos` and the 4-byte rope_offset load stay at the head instead of
    // sitting between the K and Q weight loads.
    let one_k: RopeF32<m![1 # 8]> = device
        .main
        .begin(after.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, ROPE_ONE_OVER_DS)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let one_vrf: VrfTensor<f32, Chip, Cluster, HeadSlices, m![1 # 8]> = device
        .sub
        .begin(one_k.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    // POS on every head slice: the one small DMA of this path, consumed at the head.
    let off: DmTensor<i32, Chip, Cluster, HeadSlices, m![1 # 8]> = rope_offset.to_dm(&mut device.tdma);
    let pos: RopeF32<m![1 # 8]> = device
        .main
        .begin(off.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(24)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_OFFSET_TO_POS)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let pos_vrf: VrfTensor<f32, Chip, Cluster, HeadSlices, m![1 # 8]> = device
        .sub
        .begin(pos.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    // f(0) = 1, then the geometric ramp by doubling: [1] -> [1, r] -> [1, r, r^2, r^3] -> ...
    let one: RopeF32<m![1 # 8]> = device
        .main
        .begin(pos.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, &one_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let v = rope_ramp_2(device, &one, ROPE_RATIO_1);
    let v = rope_ramp_4(device, &v, ROPE_RATIO_2);
    let v = rope_ramp_8(device, &v, ROPE_RATIO_4);
    let v = rope_ramp_16(device, &v, ROPE_RATIO_8);
    let v = rope_ramp_32(device, &v, ROPE_RATIO_16);
    let v = rope_ramp_64(device, &v, ROPE_RATIO_32);
    let v = rope_ramp_128(device, &v, ROPE_RATIO_64);
    let v = rope_ramp_256(device, &v, 1.0f32);
    // One value per flit -> dense [Ds]: the Transpose Engine packs two f32 rows per flit.
    let f: RopeF32<m![Ds]> = device
        .main
        .begin(v.view())
        .fetch::<m![Ds], m![1 # 8]>()
        .collect::<m![Ds], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![Ds / 2], m![Ds % 2 # 8]>()
        .commit_trim::<m![Ds % 2]>()
        .commit();
    // angle = POS * f as f32 on every head slice (one pass), then reduced to [-pi, pi]:
    // turns = round(angle / 2pi) through the Vector Engine's f32 -> i32 conversion (the host model
    // `float_to_fixedpoint` rounds to nearest; int_width 24 = the same fixed point the POS read uses,
    // so `turns` holds 128 * n), r = angle - 2pi * n. cos(angle) = cos(r), sin(angle) = sin(r), and
    // the hardware Sin/Cos never see an argument beyond pi (ban 40_01 fed them up to POS = 137 rad
    // and q/k came back non-finite).
    let angle: RopeF32<m![Ds]> = device
        .main
        .begin(f.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &pos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let angle_vrf: VrfTensor<f32, Chip, Cluster, HeadSlices, m![Ds]> = device
        .sub
        .begin(angle.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let turns: DmTensor<i32, Chip, Cluster, HeadSlices, m![Ds]> = device
        .main
        .begin(angle.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_INV_TWO_PI_Q24)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_fp_to_fxp(ROPE_TURNS_INT_WIDTH)
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    // cos(r) over all 256 channels, -sin(r) over the low half, +sin(r) over the high half.
    let mut rows: RopeRows = DmTensor::new();
    device.main
        .begin(turns.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(ROPE_TURNS_INT_WIDTH)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_NEG_TWO_PI_Q24)
        .vector_fp_binary(FpBinaryOp::AddF, &angle_vrf)
        .vector_fp_unary(FpUnaryOp::Cos)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit_view(rows.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(0));
    // The two sin halves as a macro: a vector-engine immediate must be a literal (an `if`-selected
    // sign does not const-fold). They land in their own [Ds] tensor through single-level tiles (a
    // nested Dummy2 x Ds tile write fails verification) and are then copied into `rows`.
    let mut sin_full: DmTensor<bf16, Chip, Cluster, HeadSlices, m![Ds]> = DmTensor::new();
    macro_rules! sin_half {
        ($half:literal, $sign:literal) => {
            let angle_half: VrfTensor<f32, Chip, Cluster, HeadSlices, m![Ds = 128]> = device
                .sub
                .begin(angle.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128 * $half))
                .fetch::<m![1], m![Ds = 128]>()
                .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .to_vrf();
            device.main
                .begin(turns.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128 * $half))
                .fetch::<m![1], m![Ds = 128]>()
                .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp_to_fp(ROPE_TURNS_INT_WIDTH)
                .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_NEG_TWO_PI_Q24)
                .vector_fp_binary(FpBinaryOp::AddF, &angle_half)
                .vector_fp_unary(FpUnaryOp::Sin)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), $sign)
                .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                .commit_trim::<m![Ds = 128 % 8]>()
                .commit_view(sin_full.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128 * $half));
        };
    }
    sin_half!(0, -1.0f32);
    sin_half!(1, 1.0f32);
    device.main
        .begin(sin_full.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(rows.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(1));
    rows
}

/// `apply_rope` with the sin products computed straight from the two half tiles of q / k (the low
/// half of the output is the high half of the input times the low half of sin, and vice versa):
/// two passes per tensor instead of two copy passes plus one product pass.
pub(crate) fn apply_rope(
    device: &mut Device,
    q: &DmTensor<bf16, Chip, Cluster, HeadSlices, m![Gs, Ds]>,
    k: &DmTensor<bf16, Chip, Cluster, HeadSlices, m![Ds]>,
    rows: &RopeRows,
) -> (
    DmTensor<bf16, Chip, Cluster, HeadSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, HeadSlices, m![Ds]>,
) {
    let cos_vrf: VrfTensor<f32, Chip, Cluster, HeadSlices, m![Ds]> = device
        .sub
        .begin(rows.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(0))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let sin = rows.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(1);
    let sin_lo: VrfTensor<f32, Chip, Cluster, HeadSlices, m![Ds = 128]> = device
        .sub
        .begin(sin.tile::<m![Ds], 128, m![Dummy2 = 1 # 2, Ds = 128 # 256]>(0))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();
    let sin_hi: VrfTensor<f32, Chip, Cluster, HeadSlices, m![Ds = 128]> = device
        .sub
        .begin(sin.tile::<m![Ds], 128, m![Dummy2 = 1 # 2, Ds = 128 # 256]>(128))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();

    let mut q_sin: DmTensor<f32, Chip, Cluster, HeadSlices, m![Gs, Ds]> = DmTensor::new();
    // q_sin[.., 0..128] = q[.., 128..256] * sin[0..128]
    device.main
        .begin(q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(128))
        .fetch::<m![Gs], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_lo)
        .vector_widen_concat::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(q_sin.view_mut().tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(0));
    // q_sin[.., 128..256] = q[.., 0..128] * sin[128..256]
    device.main
        .begin(q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(0))
        .fetch::<m![Gs], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_hi)
        .vector_widen_concat::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(q_sin.view_mut().tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(128));
    let q_sin_vrf: VrfTensor<f32, Chip, Cluster, HeadSlices, m![Gs, Ds]> = device
        .sub
        .begin(q_sin.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let result_q: DmTensor<bf16, Chip, Cluster, HeadSlices, m![Gs, Ds]> = device
        .main
        .begin(q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &q_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let mut k_sin: DmTensor<f32, Chip, Cluster, HeadSlices, m![Ds]> = DmTensor::new();
    device.main
        .begin(k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_lo)
        .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(k_sin.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0));
    device.main
        .begin(k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_hi)
        .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(k_sin.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128));
    let k_sin_vrf: VrfTensor<f32, Chip, Cluster, HeadSlices, m![Ds]> = device
        .sub
        .begin(k_sin.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let result_k: DmTensor<bf16, Chip, Cluster, HeadSlices, m![Ds]> = device
        .main
        .begin(k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &k_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    (result_q, result_k)
}
