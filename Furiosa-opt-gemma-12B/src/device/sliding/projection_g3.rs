//! The attention output projection in the 16 x 16 strip geometry (WORK_LOG.md, redesign panel).
//!
//! Per cluster (cluster c owns H rows 1920c..), slice (g, s) = row group g (120 rows) x column strip
//! s (256 Qs columns): its weight piece is 120 runs of exactly 256 B (30,720 B), streamed as two
//! tiles (rows 0..92 and 92..120 of every slice) so the second tile's contraction is short. Each
//! slice needs only its 256-column strip of x (hi/lo, 512 B of TRF). The contraction sums the 16
//! strips with the Inter-Slice Reducer and leaves 120 f32 rows on slice (g, 0), which is exactly one
//! epilogue slice's worth: the cross-cluster gather moves 32 pieces of 480 B.
use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy2, H, Qs};
use crate::device::layout::Cluster;
use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

pub(crate) type G3Cluster = m![H / 1920];
/// 16 row groups of 120 rows x 16 column strips of 256.
pub(crate) type G3Slices = m![H / 120 % 16, Qs / 256];
/// The contraction output: one live slice per row group (strip digit padded).
pub(crate) type G3Rows = m![H / 120 % 16, 1 # 16];

type G3X = DmTensor<bf16, Chip, G3Cluster, G3Slices, m![Qs % 256]>;
type G3Activation = TrfTensor<f8e4m3, Chip, G3Cluster, G3Slices, m![Dummy2], m![Qs % 256]>;
pub(crate) type G3Weight = DmTensor<f8e4m3, Chip, G3Cluster, G3Slices, m![H % 120, Qs % 256]>;
type G3Result = DmTensor<f32, Chip, G3Cluster, G3Rows, m![H % 120]>;

const ACT_SCALE: f32 = 128.0;

axes![G3Epi32 = 32];
pub(crate) type G3EpilogueSlices = m![1 # 8, H / 120];
type G3Operand = DmTensor<bf16, Chip, Cluster, G3EpilogueSlices, m![H % 120]>;

/// x on every row group of both clusters, each slice holding its 256-column strip. The seed lives on
/// PE 0 only (slice (0, a, s) holds columns 256 s + 64 a .. +64, 128 B each), so the x DMA runs on
/// one engine per cluster (64 destinations) and tile A's weight DMA, issued right behind it, is not
/// held up on the other three engines. A ring-16 Broadcast1 over the 16 outer slice digits
/// (stride 16) gives every row group all four 64-column pieces of its strip (live ring positions
/// 0..3); the dead positions land in the `1 # 4` element padding, which the compaction pass skips.
type G3Seed = DmTensor<bf16, Chip, G3Cluster, m![1 # 4, Qs / 64 % 4, Qs / 256], m![Qs % 64]>;
pub(crate) fn load_input_g3(device: &mut Device, x: HbmTensorView<'_, bf16, Chip, m![Qs]>) -> G3X {
    let seed: G3Seed = x.to_dm(&mut device.tdma);
    // The switch writes into the first half of a buffer cleared by one Sub memset whose static time
    // outlasts the x DMA, so the first x consumer is listed after tile A's weight DMA and the core
    // issues tile A right behind x (x uses only PE 0's engine per cluster).
    let mut padded: DmTensor<bf16, Chip, G3Cluster, G3Slices, m![Dummy2, 1 # 4, Qs / 64 % 4, Qs % 64]> = DmTensor::new();
    padded.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    device
        .main
        .begin(seed.view())
        .fetch::<m![1], m![Qs % 64]>()
        .switch::<G3Slices, m![1 # 4, Qs / 64 % 4]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 16 })
        .collect::<m![1 # 4, Qs / 64 % 4, Qs / 16 % 4], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit_view(padded.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, 1 # 4, Qs / 64 % 4, Qs % 64]>(0));
    let gathered: DmTensor<bf16, Chip, G3Cluster, G3Slices, m![Qs / 64 % 4, Qs % 64]> = device
        .main
        .begin(padded.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, 1 # 4, Qs / 64 % 4, Qs % 64]>(0))
        .fetch::<m![Qs / 64 % 4, Qs / 16 % 4], m![Qs % 16]>()
        .collect::<m![Qs / 64 % 4, Qs / 16 % 4], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit();
    unsafe { gathered.reshape() }
}

/// `[hi, lo]` with `hi = f8(128 x)`, `lo = f8(128 x - hi)`, per slice 2 x 256 FP8.
fn split_g3(device: &mut Device, x: &G3X) -> DmTensor<f8e4m3, Chip, G3Cluster, G3Slices, m![Dummy2, Qs % 256]> {
    let mut out: DmTensor<f8e4m3, Chip, G3Cluster, G3Slices, m![Dummy2, Qs % 256]> = DmTensor::new();
    device.main
        .begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ACT_SCALE)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(out.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Qs % 256]>(0));
    let hi_neg: DmTensor<f8e4m3, Chip, G3Cluster, G3Slices, m![Qs % 256]> = device
        .main
        .begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -ACT_SCALE)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let hi_neg_vrf: VrfTensor<f32, Chip, G3Cluster, G3Slices, m![Qs % 256]> = device
        .sub
        .begin(hi_neg.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .to_vrf();
    device.main
        .begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ACT_SCALE)
        .vector_fp_binary(FpBinaryOp::AddF, &hi_neg_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(out.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Qs % 256]>(1));
    out
}

axes![G3Q = 32, G3R = 120];

/// Rows `r0..r0+ROWS` of every slice's 120 rows (ROWS x 256 B per slice) as its own DM tensor, and
/// its contraction into rows `r0..r0+ROWS` of `result` (ver16 `output_tile_fns!` in the G3 geometry).
macro_rules! g3_tile_fns {
    ($load:ident, $tile:ident, $rows:literal, $r0:literal) => {
        fn $load(device: &mut Device, weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>) -> DmTensor<f8e4m3, Chip, G3Cluster, G3Slices, m![H % 120 = $rows, Qs % 256]> {
            let v: HbmTensorView<'_, f8e4m3, Chip, m![G3Q, G3R, Qs]> = unsafe { weight.view().reshape() };
            let d: DmTensor<f8e4m3, Chip, m![G3Q / 16], m![G3Q % 16, Qs / 256], m![G3R = $rows, Qs % 256]> =
                v.tile::<m![G3R], $rows, m![G3Q, G3R = $rows # 120, Qs]>($r0).to_dm(&mut device.tdma);
            unsafe { d.reshape() }
        }

        fn $tile(
            device: &mut Device,
            x_trf: &G3Activation,
            w: DmTensor<f8e4m3, Chip, G3Cluster, G3Slices, m![H % 120 = $rows, Qs % 256]>,
            result: &mut G3Result,
        ) {
            device.main
                .begin(w.view())
                .fetch::<m![H % 120 = $rows, Qs / 32 % 8], m![Qs % 32]>()
                .collect::<m![H % 120 = $rows, Qs / 32 % 8], m![Qs % 32]>()
                .contract_outer::<m![H % 120 = $rows, Qs / 64 % 4], m![Qs % 64], _, _, _>(x_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![H % 120 = $rows]>()
                .contract_lane::<m![H % 120 = $rows, Dummy2], m![1 # 8]>(LaneMode::Sequential)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_intra_slice_reduce::<Dummy2, m![H % 120 = $rows], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<G3Rows, m![H % 120 = $rows]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .transpose::<m![H % 120 = $rows / 2], m![H % 120 = $rows % 2 # 8]>()
                .commit_trim::<m![H % 120 = $rows % 2]>()
                .commit_view(result.view_mut().tile::<m![H % 120], $rows, m![H % 120 = $rows #{!} 120]>($r0));
        }
    };
}
g3_tile_fns!(load_tile_a, contract_tile_a, 92, 0);
g3_tile_fns!(load_tile_b, contract_tile_b, 28, 92);

/// Identity copy of x (a delay device, see ver16 `delay_output_input`): makes x_trf ready late in the
/// static schedule so the second tile's DMA is listed before the first contraction.
fn copy_g3(device: &mut Device, x: &G3X) -> G3X {
    device.main
        .begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .collect::<m![Qs / 16 % 16], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit()
}

/// 24 copies: with the PE-0 x seed and the memset in front of the switch, 24 copies list tile B's
/// weight DMA after the split passes and the x StoTrf but before the wait on tile A, so tile B is
/// queued before tile A lands and x_trf is ready when it does (with 32 copies tile B was issued
/// 0.2-1.0k after tile A landed, and with 22 or fewer it is released only at contraction A's
/// dispatch). History: with the 16-destination quarter seed the window was 31..32 copies (33 put a
/// gather wait before the residual DMA, 34 listed weight_scale before the last contraction); a copy
/// is ~0.35-1.5k on hardware against 280 static.
fn delay_g3(device: &mut Device, x: &G3X) -> G3X {
    let x = copy_g3(device, x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    let x = copy_g3(device, &x);
    x
}

fn operand_vrf(device: &mut Device, t: &G3Operand) -> VrfTensor<f32, Chip, Cluster, G3EpilogueSlices, m![H % 120]> {
    device
        .sub
        .begin(t.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf()
}

/// The gather destination, produced by a Sub pass from x so its readiness sync is released after
/// x lands instead of at the kernel start (ver16 J1).
fn gather_buffer_g3(device: &mut Device, x: &G3X) -> DmTensor<f32, Chip, Cluster, G3EpilogueSlices, m![H % 120]> {
    let t: DmTensor<f32, Chip, G3Cluster, G3Slices, m![Qs % 256 = 120]> = device
        .sub
        .begin(x.view().tile::<m![Qs % 256], 120, m![Qs % 256 = 120 # 256]>(0))
        .fetch::<m![1], m![Qs % 256 = 120]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs % 256 = 120 / 8], m![Qs % 256 = 120 % 8]>()
        .commit_trim::<m![Qs % 256 = 120 % 8]>()
        .commit();
    unsafe { t.reshape() }
}

/// k2c projection + fused epilogue (the epilogue is ver16's, unchanged: gather into cluster 0's
/// 32 x 120 slices, mean square, ring-32 all-gather reduce, sqrt, final).
pub(crate) fn project_output_g3(
    device: &mut Device,
    x: &G3X,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    norm_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, G3EpilogueSlices, m![H % 120]> {
    let y_buffer = gather_buffer_g3(device, x);
    let xd = delay_g3(device, x);
    let x2 = split_g3(device, &xd);
    let x_trf: G3Activation = device
        .sub
        .begin(x2.view())
        .fetch::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();
    let weight_scale: G3Operand = weight_scale.to_dm(&mut device.tdma);
    let norm_weight: G3Operand = norm_weight.to_dm(&mut device.tdma);
    let residual: G3Operand = residual.to_dm(&mut device.tdma);
    let w0 = load_tile_a(device, weight);
    let w1 = load_tile_b(device, weight);
    let mut result: G3Result = DmTensor::new();
    contract_tile_a(device, &x_trf, w0, &mut result);
    contract_tile_b(device, &x_trf, w1, &mut result);
    let mut y = y_buffer;
    result.view().to_dm_view(&mut device.tdma, y.view_mut());

    let weight_scale_vrf = operand_vrf(device, &weight_scale);
    let residual_vrf = operand_vrf(device, &residual);
    let y_vrf: VrfTensor<f32, Chip, Cluster, G3EpilogueSlices, m![H % 120]> = device
        .sub
        .begin(y.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();

    let partial_mean_square: DmTensor<f32, Chip, Cluster, G3EpilogueSlices, m![1 # 8]> = device
        .main
        .begin(y.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![1 # 8, G3Epi32], m![1 # 8]> = device
        .main
        .begin(partial_mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![1 # 8, G3Epi32], m![H / 120]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
        .collect::<m![H / 120], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let mean_square: DmTensor<f32, Chip, Cluster, G3EpilogueSlices, m![1 # 8]> = unsafe { reduced_mean_square.reshape() };
    let rms_vrf: VrfTensor<f32, Chip, Cluster, G3EpilogueSlices, m![1 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS * ACT_SCALE * ACT_SCALE)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
    device.main
        .begin(norm_weight.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &y_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
