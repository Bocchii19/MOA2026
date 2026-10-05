
use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy2, Dummy8, Dummy256, Gs, H, L};
use crate::device::layout::{Cluster, Slice};
use crate::{Chip, EPS};

const INVSQRT2: f32 = 0.70710678118f32;
const H_F32: f32 = H::SIZE as f32;

/// Layout of the fused Down epilogue (global scale, post-FFN RMSNorm, residual, layer gate):
/// 16 slices of cluster 0 with 240 channels each. The kernel stores the result to HBM straight
/// from it.
pub(crate) type EpilogueSlices = m![1 # 16, H / 240];
pub(crate) type EpilogueElem = m![H % 240];
axes![Epi16 = 16];
/// The raw bf16 residual as `normalize_input` loaded it (eight 480-channel slices), viewed on
/// cluster 0: both clusters hold the same copy and HBM is not written before the final store, so
/// reusing it for the epilogue's residual add is bit-exact and saves the second residual DMA.
pub(crate) type RawResidual = DmTensor<bf16, Chip, Cluster, m![1 # 32, H / 480], m![H % 480]>;

// ---- Prologue span: the two H-sized prologue operands (residual | pre_ff_rms_weight) are contiguous in HBM
// ONE DMA reads the 15,360-B span from the
// residual's base (`HbmTensorView::pad` over-read, then `reshape`) onto the eight 480-channel
// seed slices of each cluster; the operands are read as tiles of that buffer. post_ff keeps its own tail DMA.
axes![FfOp = 2];
const OP_RESIDUAL: usize = 0;
const OP_PRE_RMS: usize = 1;
type FfSpanHbm = m![H # 7680];
const _: () = assert!(<FfSpanHbm as M>::SIZE == FfOp::SIZE * H::SIZE);
pub(crate) type FfSpan = DmTensor<bf16, Chip, UpCluster, UpPairRows, m![FfOp, H % 480]>;
type FfOpView<'l> = DmTensorView<'l, bf16, Chip, UpCluster, UpPairRows, m![FfOp = 1 # 2, H % 480]>;

fn span_tile(span: &FfSpan, op: usize) -> FfOpView<'_> {
    span.view().tile::<m![FfOp], 1, m![FfOp = 1 # 2, H % 480]>(op)
}

/// The one span DMA (anchor = the residual, the lowest address).
pub(crate) fn load_ff_span(device: &mut Device, residual: &HbmTensor<bf16, Chip, m![H]>) -> FfSpan {
    let span: HbmTensorView<'_, bf16, Chip, FfSpanHbm> = residual.view().pad();
    let span: HbmTensorView<'_, bf16, Chip, m![FfOp, H]> = unsafe { span.reshape() };
    span.to_dm(&mut device.tdma)
}

/// The raw seed relaid onto the 16 epilogue slices, on chip: (1) ring-16 InterTranspose moves each
/// 480-channel seed slice's two 240-channel halves to slices h2 * 8 + h8; (2) ring-16 Transpose
/// swaps those two slice digits into the epilogue order 2 * h8 + h2. Named switch configs (no
/// config DMA). Pass (1) now writes a FRESH buffer on Main (was: Sub, into the dead
/// normalized-input buffer through an unsafe reshape). With one up+gate scale DMA the list
/// schedule placed the aliased write before x's last reads and lir refused it ("T30 has an
/// inconsistent state ... owner mismatch"); on Sub it also held the first table DMA (Sub-context
/// wait) for ~330 static cycles. On Main it lands in Main's idle window during the Up stream.
fn relayout_residual(
    device: &mut Device,
    seed: FfOpView<'_>,
    dead: NormalizedInput,
) -> DmTensor<bf16, Chip, Cluster, EpilogueSlices, EpilogueElem> {
    let mut halves: DmTensor<bf16, Chip, UpCluster, m![1 # 16, H / 240 % 2, H / 480], m![H % 240 / 16, 1 # 2, H % 16]> =
        { let _ = dead; DmTensor::new() };
    device
        .main
        .begin(seed)
        .fetch::<m![H / 240 % 2, H % 240 / 16], m![H % 16]>()
        .switch::<m![1 # 16, H / 240 % 2, H / 480], m![H % 240 / 16, 1 # 2]>(SwitchConfig::InterTranspose {
            slice1: 2,
            slice0: 8,
            time0: 15,
        })
        .collect::<m![H % 240 / 16, 1 # 2], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit_view(halves.view_mut());
    let out: DmTensor<bf16, Chip, UpCluster, EpilogueSlices, EpilogueElem> = device
        .sub
        .begin(halves.view())
        .fetch::<m![H % 240 / 16], m![H % 16]>()
        .switch::<EpilogueSlices, m![H % 240 / 16]>(SwitchConfig::Transpose { slice1: 2, slice0: 8 })
        .collect::<m![H % 240 / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    unsafe { out.reshape() }
}
axes![WmSlot = 4];

type UpCluster = m![L / 7680];
/// 256 slices per cluster, each holding 30 consecutive Up/Gate rows over the full K = H.
///
/// A slice's packed weights are then one contiguous 57.6 KiB run in HBM (30 rows x 1920 B) and
/// its block scales one contiguous 7.2 KiB run (30 x 240 B). The previous
/// `[128 row groups x 2 K-halves]` layout fetched 60 x 960 B and 60 x 120 B pieces per slice
/// (DMA util 0.74 and 0.28) and then needed an inter-slice add of the two K-half partials plus
/// its own relayout. Per-slice work is unchanged (30 x 3840 = 60 x 1920 weights).
type UpRows = m![L / 30 % 256];
type UpActivation = TrfTensor<f8e4m3, Chip, UpCluster, UpRows, m![1], m![Gs, H]>;

type UpPairRows = m![1 # 32, H / 480];
pub(crate) type NormalizedInput = DmTensor<bf16, Chip, UpCluster, UpPairRows, m![H % 480]>;
// Finite FP8 hi requires abs(x) <= 28. This is checked on the expected inputs in
// numeric_check.py; arbitrary out-of-range activations need a different scaling scheme.
const UP_ACTIVATION_SCALE: f32 = 16.0;
fn split_up_activation(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, UpCluster, UpPairRows, m![H % 480]>,
) -> DmTensor<f8e4m3, Chip, UpCluster, UpPairRows, m![Gs, H % 480]> {
    type Rows = UpPairRows;
    let hi: DmTensor<f8e4m3, Chip, UpCluster, Rows, m![H % 480]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), UP_ACTIVATION_SCALE)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    // -hi (exact) as the VRF operand of the lo pass: `SubF` with a VRF operand evaluates
    // `vrf - stream` and a MulF may not follow it, so the lo pass adds the negated hi.
    let hi_neg: DmTensor<f8e4m3, Chip, UpCluster, Rows, m![H % 480]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -UP_ACTIVATION_SCALE)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    let hi_neg_vrf: VrfTensor<f32, Chip, UpCluster, Rows, m![H % 480]> = device
        .sub
        .begin(hi_neg.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let lo: DmTensor<f8e4m3, Chip, UpCluster, Rows, m![H % 480]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), UP_ACTIVATION_SCALE)
        .vector_fp_binary(FpBinaryOp::AddF, &hi_neg_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    let mut out: DmTensor<f8e4m3, Chip, UpCluster, Rows, m![Gs, H % 480]> = DmTensor::new();
    device.main
        .begin(hi.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .collect::<m![H / 32 % 15], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, H % 480]>(0));
    device.main
        .begin(lo.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .collect::<m![H / 32 % 15], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, H % 480]>(1));
    out
}



// Split after the BF16 GeGLU rounding, while each active slice holds 120 values.
// The GeGLU here is not multiplied by the Up global scale (applied in the Down epilogue instead),
// so it is `raw_up` (2048..16384) times the true GeGLU: 1/16 keeps the FP8 hi finite for
// abs(unscaled GeGLU) <= 7168, i.e. true abs(GeGLU) <= 3.5 at the smallest global scale
// (the previous 256 on the scaled GeGLU allowed 1.75).
const DOWN_ACTIVATION_SCALE: f32 = 0.0625;
fn split_down_activation(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, UpCluster, UpPairedRows, m![L % 240]>,
) -> DmTensor<f8e4m3, Chip, UpCluster, UpPairedRows, m![Gs, L % 240]> {
    type Rows = UpPairedRows;
    let mut out: DmTensor<f8e4m3, Chip, UpCluster, Rows, m![Gs, L % 240]> = DmTensor::new();
    device
        .main
        .begin(x.view())
        .fetch::<m![L / 8 % 30], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 30], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 60], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), DOWN_ACTIVATION_SCALE)
        .vector_widen_concat::<m![L / 8 % 30], m![L % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>()
        .commit_trim::<m![L % 8]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, L % 240]>(0));
    // -hi (exact) as the VRF operand of the lo pass: `SubF` with a VRF operand evaluates
    // `vrf - stream` and a MulF may not follow it, so the lo pass adds the negated hi.
    let hi_neg: DmTensor<f8e4m3, Chip, UpCluster, Rows, m![L % 240]> = device
        .main
        .begin(x.view())
        .fetch::<m![L / 8 % 30], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 30], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 60], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -DOWN_ACTIVATION_SCALE)
        .vector_widen_concat::<m![L / 8 % 30], m![L % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>()
        .commit_trim::<m![L % 8]>()
        .commit();
    let hi_neg_vrf: VrfTensor<f32, Chip, UpCluster, Rows, m![L % 240]> = device
        .sub
        .begin(hi_neg.view())
        .fetch::<m![L / 8 % 30], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 30], m![L % 8]>()
        .to_vrf();
    device
        .main
        .begin(x.view())
        .fetch::<m![L / 8 % 30], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 30], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 60], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), DOWN_ACTIVATION_SCALE)
        .vector_fp_binary(FpBinaryOp::AddF, &hi_neg_vrf)
        .vector_widen_concat::<m![L / 8 % 30], m![L % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>()
        .commit_trim::<m![L % 8]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, L % 240]>(1));
    out
}



/// Block scales, each slice's own 30 rows (7,200 B contiguous in HBM), loaded directly in the
/// layout the scale passes read: no pair seed and no InterTranspose pass on Main.
type UpScale = DmTensor<f8e4m3, Chip, UpCluster, UpRows, m![L % 30, H / 16]>;

fn load_up_gate_scale(device: &mut Device, scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>) -> UpScale {
    scale.to_dm(&mut device.tdma)
}

axes![Dq4 = 4];
/// LAB q4: 120-row quad seed (28.8 KB per leader slice; every other quad starts 256 B aligned,
/// the pair seed only every fourth pair) and a ring-4 InterTranspose.
type UpScale4 = DmTensor<f8e4m3, Chip, UpCluster, UpRows, m![Dq4, L % 30, H / 16]>;
type UpScaleSeed4 = DmTensor<f8e4m3, Chip, UpCluster, m![L / 120 % 64, 1 # 4], m![L % 120, H / 16]>;

fn load_up_gate_scale4(device: &mut Device, scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>) -> UpScale4 {
    let seed: UpScaleSeed4 = scale.to_dm(&mut device.tdma);
    let seed: DmTensor<f8e4m3, Chip, UpCluster, m![L / 120 % 64, Dq4], m![L % 120 / 2, H / 8]> =
        unsafe { seed.reshape() };
    let switched: DmTensor<f8e4m3, Chip, UpCluster, m![L / 120 % 64, L % 120 / 30], m![Dq4, L % 30 / 2, H / 8]> = device.main
        .begin(seed.view())
        .fetch::<m![L % 120 / 30, L % 30 / 2], m![H / 8]>()
        .switch::<m![L / 120 % 64, L % 120 / 30], m![L % 30 / 2, Dq4]>(
            SwitchConfig::InterTranspose { slice1: 4, slice0: 1, time0: 15 },
        )
        .collect::<m![L % 30 / 2, Dq4, H / 8 / 32 % 15], m![H / 8 % 32]>()
        .commit_trim::<m![H / 8 % 32]>()
        .commit();
    unsafe { switched.reshape() }
}

fn up_gate_scale_pass4(device: &mut Device, blocks: &UpBlocks, scale: &UpScale4, result: &mut UpResult, r0: usize) {
    let scale_vrf: VrfTensor<f32, Chip, UpCluster, UpRows, m![L % 30 = 6, H / 16]> = device
        .sub
        .begin(
            scale
                .view()
                .tile::<m![Dq4], 1, m![Dq4 = 1 # 4, L % 30, H / 16]>(0)
                .tile::<m![L % 30], 6, m![Dq4 = 1 # 4, L % 30 = 6 # 30, H / 16]>(r0),
        )
        .fetch::<m![L % 30 = 6], m![H / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 30 = 6, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .to_vrf();
    device.main
        .begin(blocks.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16 % 240]>(r0))
        .fetch::<m![L % 30 = 6, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .collect::<m![L % 30 = 6, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 30 = 6, H / 16 / 4 % 60], m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(UP_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![L % 30 = 6 / 2], m![L % 30 = 6 % 2 # 8]>()
        .commit_trim::<m![L % 30 = 6 % 2]>()
        .commit_view(result.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30]>(r0));
}

/// Block-16 partial dots for the slice's 30 rows (30 x 240 x 4 B = 28.8 KiB).
type UpBlocks = DmTensor<f32, Chip, UpCluster, UpRows, m![L % 30, H / 16 % 240]>;

/// f32 result: a 30 x bf16 = 60 B in-slice extent is rejected (SRAM access width is 8 B).
type UpResult = DmTensor<f32, Chip, UpCluster, UpRows, m![L % 30]>;

/// Decode FP4 once to FP8, then stream it twice against scaled activation hi/lo.
/// Keep each block-16 dot separate until its own block scale has been applied.
/// Each tile owns a separate block buffer to avoid mutable-view owner conflicts in
/// this compiler. All input weights retain the original external layout.
macro_rules! up_gate_tile_fns {
    ($load:ident, $tile:ident, $rows:literal) => {
        fn $load(
            device: &mut Device,
            weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
            r0: usize,
        ) -> DmTensor<f4e2m1, Chip, UpCluster, UpRows, m![L % 30 = $rows, H]> {
            weight
                .view()
                .tile::<m![L % 30], $rows, m![L / 30, L % 30 = $rows # 30, H]>(r0)
                .to_dm(&mut device.tdma)
        }

        fn $tile(
            device: &mut Device,
            x: &UpActivation,
            packed: DmTensor<f4e2m1, Chip, UpCluster, UpRows, m![L % 30 = $rows, H]>,
            blocks: &mut UpBlocks,
            r0: usize,
        ) {
            // api081 P1: FP4 -> FP8 lookup fused into the contraction pass; Gs (hi/lo) is a
            // stream-adapter time broadcast, so each weight packet is fetched and decoded once.
            device.main
                .begin(packed.view())
                .fetch::<m![L % 30 = $rows], m![H]>()
                .fetch_table_lookup::<f8e4m3>()
                .collect::<m![L % 30 = $rows, H / 64 % 60, H / 32 % 2], m![H % 32]>()
                .contract_outer::<m![L % 30 = $rows, H / 64 % 60, Gs], m![H % 64], _, _, _>(x)
                .contract_packet::<m![H / 16 % 4]>()
                .contract_time::<m![L % 30 = $rows, H / 64 % 60]>()
                .contract_lane::<m![L % 30 = $rows, H / 64 % 60], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
                .commit_trim::<m![H / 16 % 4]>()
                .commit_view(blocks.view_mut().tile::<m![L % 30], $rows, m![L % 30 = $rows #{!} 30, H / 16 % 240]>(r0));
        }
    };
}
up_gate_tile_fns!(load_up_gate_tile6, up_gate_tile6, 6);
up_gate_tile_fns!(load_up_gate_tile12, up_gate_tile12, 12);
up_gate_tile_fns!(load_up_gate_tile18, up_gate_tile18, 18);
up_gate_tile_fns!(load_up_gate_tile24, up_gate_tile24, 24);
up_gate_tile_fns!(load_up_gate_tile30, up_gate_tile30, 30);

/// Block scales applied and summed over K for rows `r0..r0+6`. The VRF holds at most 8 KiB =
/// 2,048 f32 block scales (8.5 rows x 240), so tiles are finished in 6-row passes; each pass
/// writes whole 8 B row pairs.
macro_rules! up_gate_scale_pass_fn {
    ($name:ident, $g:literal) => {
fn $name(device: &mut Device, blocks: &UpBlocks, scale: &UpScale, result: &mut UpResult, r0: usize) {
    // Written from Main through the vector engine: a Sub `to_vrf` that reads the directly loaded
    // scale would be scheduled eagerly after the scale DMA and hold the next weight DMA's issue.
    // 1/UP_ACTIVATION_SCALE (a power of two, exact) is folded into the operand.
    let scale_vrf: VrfTensor<f32, Chip, UpCluster, UpRows, m![L % 30 = $g, H / 16]> = device
        .main
        .begin(scale.view().tile::<m![L % 30], $g, m![L % 30 = $g # 30, H / 16]>(r0))
        .fetch::<m![L % 30 = $g], m![H / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 30 = $g, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 30 = $g, H / 16 / 4 % 60], m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1f32 / UP_ACTIVATION_SCALE)
        .vector_widen_concat::<m![L % 30 = $g, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
    device.main
        .begin(blocks.view().tile::<m![L % 30], $g, m![L % 30 = $g # 30, H / 16 % 240]>(r0))
        .fetch::<m![L % 30 = $g, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .collect::<m![L % 30 = $g, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 30 = $g, H / 16 / 4 % 60], m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 30 = $g], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![L % 30 = $g / 2], m![L % 30 = $g % 2 # 8]>()
        .commit_trim::<m![L % 30 = $g % 2]>()
        .commit_view(result.view_mut().tile::<m![L % 30], $g, m![L % 30 = $g #{!} 30]>(r0));
}
    };
}
up_gate_scale_pass_fn!(up_gate_scale_pass, 6);
up_gate_scale_pass_fn!(up_gate_scale_pass8, 8);

// ---- Interleaved-pair (IP) Up/Gate layout --------------------------------------------------------
// Per cluster 32 blocks of 240 rows; slice 8 k + c holds the row pairs 240 k + 16 g + 2 c + {0, 1},
// g = 0..14. A slice's packed weights are then 15 runs of 3,840 B (15 whole 256 B units) and the
// slice stride is 3,840 B, an odd multiple of 256 B, so each DMA engine's inner loop still toggles
// HBM address bit 8 (the stream stays at the ceiling). The block scales, which no equal 30-row
// share can load 256 B aligned (7,200 B = 28.125 units per slice), now load as whole 16-row groups
// (3,840 B, 256 B aligned): the scale-source slice (super-block, g) holds group g of both 240-row
// blocks of its 480-row super-block (slot 15 of every 16 is padding), and one ring-16
// InterTranspose on Main gives slice (block, c) pair c of every group of its block -- exactly its
// 15 row pairs, in the weight's row order.
/// L = 480 Sk2 + 240 Sgs + 16 Sg + 2 Sp + (L % 2), per chip; Spr = one row pair's 480 B of scale.
axes![Sk2 = 32, Sgs = 2, Sg = 15, Sp = 8, Spr = 480];
type IpRows = m![L / 240 % 32, L / 2 % 8];
type IpActivation = TrfTensor<f8e4m3, Chip, UpCluster, IpRows, m![1], m![Gs, H]>;
type IpWeight = DmTensor<f4e2m1, Chip, UpCluster, IpRows, m![L / 16 % 15, L % 2, H]>;
type IpScaleSrc = DmTensor<f8e4m3, Chip, m![Sk2 / 16], m![Sk2 % 16, Sg # 16], m![Sgs, Sp, Spr]>;
type IpScaleSw = DmTensor<f8e4m3, Chip, m![Sk2 / 16], m![Sk2 % 16, Sgs, Sp], m![Sg # 16, Spr]>;
/// A slice's block scales after the hand-over: its 15 row pairs (plus one padding pair) x 240.
type IpScale = DmTensor<f8e4m3, Chip, UpCluster, IpRows, m![L / 16 % 15 # 16, L % 2, H / 16]>;
type IpBlocks = DmTensor<f32, Chip, UpCluster, IpRows, m![L / 16 % 15, L % 2, H / 16 % 240]>;
type IpBlocksBf = DmTensor<bf16, Chip, UpCluster, IpRows, m![L / 16 % 15, L % 2, H / 16 % 240]>;
type IpScaleBf = DmTensor<bf16, Chip, UpCluster, IpRows, m![L / 16 % 15, L % 2, H / 16]>;
type IpScaleTrf = TrfTensor<bf16, Chip, UpCluster, IpRows, m![1], m![L / 16 % 15, L % 2, H / 16]>;
/// f32 result, one 8 B unit per row pair.
type IpResult = DmTensor<f32, Chip, UpCluster, IpRows, m![L / 16 % 15, L % 2]>;

fn load_ip_weight(device: &mut Device, weight: &HbmTensor<f4e2m1, Chip, m![L, H]>) -> IpWeight {
    weight.to_dm(&mut device.tdma)
}

axes![Iug = 2];
/// Up (Iug 0) and Gate (Iug 1) block scales after one shared hand-over pass.
type IpScale2 = DmTensor<f8e4m3, Chip, UpCluster, IpRows, m![L / 16 % 15 # 16, Iug, L % 2, H / 16]>;
type IpScaleSrc2 = DmTensor<f8e4m3, Chip, m![Sk2 / 16], m![Sk2 % 16, Sg # 16], m![Iug, Sgs, Sp, Spr]>;
type IpScaleSw2 = DmTensor<f8e4m3, Chip, m![Sk2 / 16], m![Sk2 % 16, Sgs, Sp], m![Sg # 16, Iug, Spr]>;
/// Up and Gate block scales as one HBM span from `up_weight_scale`'s base (Iug outermost).
type UpGateScaleSpan = m![L # 30720, H / 16];
const _: () = assert!(<UpGateScaleSpan as M>::SIZE == Iug::SIZE * L::SIZE * (H::SIZE / 16));

/// Both scale sources through ONE ring-16 InterTranspose (interleaved on `Iug`): the Gate scale then
/// has a consumer right after the Up contraction, so its DMA is queued ahead of the Gate weight and
/// its hand-over runs in the Up phase instead of after the Gate contraction.
fn load_ip_scales(
    device: &mut Device,
    up: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> IpScale2 {
    // ONE DMA for both scales by HBM over-read:
    // `gate_weight_scale` directly follows `up_weight_scale` (offsets 0 /
    // 3,686,400 in every run; 3,686,400 = 14,400 x 256, no allocator gap), so the span read from
    // up's base is [Iug = up, gate] x L x H/16. Same 3,840-B / 256-B aligned runs per slice as the
    // two tile DMAs it replaces, one command startup fewer in the saturated prologue. `gate` is
    // not read through its own handle.
    let _ = gate;
    let span: HbmTensorView<'_, f8e4m3, Chip, UpGateScaleSpan> = up.view().pad();
    let span: HbmTensorView<'_, f8e4m3, Chip, m![Iug, Sk2, Sgs, Sg, Sp, Spr]> = unsafe { span.reshape() };
    let mut src: IpScaleSrc2 = DmTensor::new();
    span.to_dm_view(&mut device.tdma, src.view_mut());
    let sw: IpScaleSw2 = device.main
        .begin(src.view())
        .fetch::<m![Iug, Sgs, Sp], m![Spr]>()
        .switch::<m![Sk2 % 16, Sgs, Sp], m![Iug, Sg # 16]>(SwitchConfig::InterTranspose { slice1: 16, slice0: 1, time0: 1 })
        .collect::<m![Iug, Sg # 16, Spr / 32], m![Spr % 32]>()
        .commit_trim::<m![Spr % 32]>()
        .commit();
    unsafe { sw.reshape() }
}

fn load_ip_scale(device: &mut Device, scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>) -> IpScale {
    let v: HbmTensorView<'_, f8e4m3, Chip, m![Sk2, Sgs, Sg, Sp, Spr]> = unsafe { scale.view().reshape() };
    let src: IpScaleSrc = v.to_dm(&mut device.tdma);
    let sw: IpScaleSw = device.main
        .begin(src.view())
        .fetch::<m![Sgs, Sp], m![Spr]>()
        .switch::<m![Sk2 % 16, Sgs, Sp], m![Sg # 16]>(SwitchConfig::InterTranspose { slice1: 16, slice0: 1, time0: 1 })
        .collect::<m![Sg # 16, Spr / 32], m![Spr % 32]>()
        .commit_trim::<m![Spr % 32]>()
        .commit();
    unsafe { sw.reshape() }
}

/// FP4 -> FP8 lookup fused into the contraction; block-16 partial dots of the slice's 30 rows.
fn up_gate_ip(device: &mut Device, x: &IpActivation, packed: IpWeight, blocks: &mut IpBlocks) {
    device.main
        .begin(packed.view())
        .fetch::<m![L / 16 % 15, L % 2], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L / 16 % 15, L % 2, H / 64 % 60, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![L / 16 % 15, L % 2, H / 64 % 60, Gs], m![H % 64], _, _, _>(x)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![L / 16 % 15, L % 2, H / 64 % 60]>()
        .contract_lane::<m![L / 16 % 15, L % 2, H / 64 % 60], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
        .commit_trim::<m![H / 16 % 4]>()
        .commit_view(blocks.view_mut());
}

/// Up block scales applied and summed over K for the row pairs of groups g0..g0+3 (6 rows).
fn up_scale_pass_ip(device: &mut Device, blocks: &IpBlocks, scale: &IpScale2, result: &mut IpResult, g0: usize) {
    let scale_vrf: VrfTensor<f32, Chip, UpCluster, IpRows, m![L / 16 % 15 = 3, L % 2, H / 16]> = device
        .sub
        .begin(
            scale
                .view()
                .unpad::<m![L / 16 % 15, Iug, L % 2, H / 16]>()
                .tile::<m![Iug], 1, m![L / 16 % 15, Iug = 1 # 2, L % 2, H / 16]>(0)
                .tile::<m![L / 16 % 15], 3, m![L / 16 % 15 = 3 # 15, Iug = 1 # 2, L % 2, H / 16]>(g0),
        )
        .fetch::<m![L / 16 % 15 = 3, L % 2], m![H / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 16 % 15 = 3, L % 2, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .to_vrf();
    device.main
        .begin(blocks.view().tile::<m![L / 16 % 15], 3, m![L / 16 % 15 = 3 # 15, L % 2, H / 16 % 240]>(g0))
        .fetch::<m![L / 16 % 15 = 3, L % 2, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .collect::<m![L / 16 % 15 = 3, L % 2, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 16 % 15 = 3, L % 2, H / 16 / 4 % 60], m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L / 16 % 15 = 3, L % 2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(UP_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![L / 16 % 15 = 3], m![L % 2 # 8]>()
        .commit_trim::<m![L % 2]>()
        .commit_view(result.view_mut().tile::<m![L / 16 % 15], 3, m![L / 16 % 15 = 3 #{!} 15, L % 2]>(g0));
}

// census_fp4 D1 for Gate: the block scale applied by a second, diagonal bf16 contraction (stream = bf16
// block partials, TRF = bf16 block scales, both [rows, H / 16]: matched axes, reduced over H / 16).
fn up_gate_ip_bf(device: &mut Device, x: &IpActivation, packed: IpWeight, blocks: &mut IpBlocksBf) {
    device.main
        .begin(packed.view())
        .fetch::<m![L / 16 % 15, L % 2], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L / 16 % 15, L % 2, H / 64 % 60, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![L / 16 % 15, L % 2, H / 64 % 60, Gs], m![H % 64], _, _, _>(x)
        .contract_packet::<m![H / 16 % 4]>()
        .contract_time::<m![L / 16 % 15, L % 2, H / 64 % 60]>()
        .contract_lane::<m![L / 16 % 15, L % 2, H / 64 % 60], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
        .commit_trim::<m![H / 16 % 4]>()
        .commit_cast::<bf16>()
        .commit_view(blocks.view_mut());
}

fn ip_scale_to_bf(device: &mut Device, scale: &IpScale2, which: usize) -> IpScaleBf {
    device.main
        .begin(
            scale
                .view()
                .unpad::<m![L / 16 % 15, Iug, L % 2, H / 16]>()
                .tile::<m![Iug], 1, m![L / 16 % 15, Iug = 1 # 2, L % 2, H / 16]>(which),
        )
        .fetch::<m![L / 16 % 15, L % 2, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 16 % 15, L % 2, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .commit_trim::<m![H / 16 % 8]>()
        .commit_cast::<bf16>()
        .commit()
}

fn ip_scale_trf(device: &mut Device, s: &IpScaleBf) -> IpScaleTrf {
    device.sub
        .begin(s.view())
        .fetch::<m![L / 16 % 15, L % 2, H / 16 / 16 % 15], m![H / 16 % 16]>()
        .collect::<m![L / 16 % 15, L % 2, H / 16 / 16 % 15], m![H / 16 % 16]>()
        .to_trf()
}

fn ip_scale_diag(device: &mut Device, blocks: &IpBlocksBf, s: &IpScaleTrf, result: &mut IpResult) {
    device.main
        .begin(blocks.view())
        .fetch::<m![L / 16 % 15, L % 2, H / 16 / 16 % 15], m![H / 16 % 16]>()
        .collect::<m![L / 16 % 15, L % 2, H / 16 / 16 % 15], m![H / 16 % 16]>()
        .contract_outer::<m![L / 16 % 15, L % 2, H / 16 / 16 % 15], m![H / 16 % 16], _, _, _>(s)
        .contract_packet::<m![1]>()
        .contract_time::<m![L / 16 % 15, L % 2]>()
        .contract_lane::<m![L / 16 % 15, L % 2], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_div(UP_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![L / 16 % 15], m![L % 2 # 8]>()
        .commit_trim::<m![L % 2]>()
        .commit_view(result.view_mut());
}

/// Up and Gate each use one weight DMA and a diagonal BF16 contraction for the block scales.
fn project_up_ip(
    device: &mut Device,
    x: &IpActivation,
    weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    scale: &IpScale2,
) -> IpResult {
    let w0 = load_ip_weight(device, weight);
    let s_bf = ip_scale_to_bf(device, scale, 0);
    let s_trf = ip_scale_trf(device, &s_bf);
    let mut result: IpResult = DmTensor::new();
    let mut blocks: IpBlocksBf = DmTensor::new();
    up_gate_ip_bf(device, x, w0, &mut blocks);
    ip_scale_diag(device, &blocks, &s_trf, &mut result);
    result
}

/// Gate uses the same block-scale path, selecting its half of the shared scale hand-over.
fn project_gate_ip(
    device: &mut Device,
    x: &IpActivation,
    weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    scale: &IpScale2,
) -> IpResult {
    let w0 = load_ip_weight(device, weight);
    let s_bf = ip_scale_to_bf(device, scale, 1);
    let s_trf = ip_scale_trf(device, &s_bf);
    let mut result: IpResult = DmTensor::new();
    let mut blocks: IpBlocksBf = DmTensor::new();
    up_gate_ip_bf(device, x, w0, &mut blocks);
    ip_scale_diag(device, &blocks, &s_trf, &mut result);
    result
}

fn project_up_and_gate(
    device: &mut Device,
    x: &NormalizedInput,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (IpResult, IpResult, EpiloguePartials) {
    // Normalize and split only eight 480-channel shards per cluster, then broadcast
    // [hi, lo] to every row slice while independent weight DMA runs.
    // Both FP8 components together occupy the same logical bytes as the BF16 input.
    let x_half = split_up_activation(device, x);
    // Two named Broadcast1 passes instead of a ring-256 CustomBroadcast, so no switch-config DMA
    // sits in the saturated prologue queue: (1) each seed slice d (0..8) replicates its shard along
    // its stride-8 ring of 32 slices (slice s then holds shard s % 8; the 31 inactive ring slots
    // stay as padding the next fetch skips); (2) the eight slices of each group all-gather their
    // shards (ring 8), which leaves the complete [hi, lo] vector on every slice.
    let x_rep: DmTensor<f8e4m3, Chip, UpCluster, m![Dummy256 / 8, H / 480], m![Gs, H / 32 % 15, 1 # 32, H % 32]> = device
        .main
        .begin(x_half.view())
        .fetch::<m![Gs, H / 32 % 15], m![H % 32]>()
        .switch::<m![Dummy256 / 8, H / 480], m![Gs, H / 32 % 15, 1 # 32]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 8 })
        .collect::<m![Gs, H / 32 % 15, 1 # 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    let x_full: DmTensor<f8e4m3, Chip, UpCluster, m![Dummy256 / 8, Dummy8], m![Gs, H]> = device
        .main
        .begin(x_rep.view())
        .fetch::<m![Gs, H / 32 % 15], m![H % 32]>()
        .switch::<m![Dummy256 / 8, Dummy8], m![Gs, H / 32 % 15, H / 480]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![Gs, H / 32 % 15, H / 480], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    // All 256 slices of each cluster now hold identical [hi, lo] full-H vectors.
    let x: DmTensor<f8e4m3, Chip, UpCluster, IpRows, m![Gs, H]> = unsafe { x_full.reshape() };
    let x_trf: IpActivation = device
        .sub
        .begin(x.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let partials = init_epilogue_partials(device, x);
    let scales = load_ip_scales(device, up_weight_scale, gate_weight_scale);
    let up = project_up_ip(device, &x_trf, up_weight_packed, &scales);
    let gate = project_gate_ip(device, &x_trf, gate_weight_packed, &scales);
    (up, gate, partials)
}

/// Cluster 0's buffer for both clusters' Down partials (the cross-cluster transfer's destination).
pub(crate) type EpiloguePartials = DmTensor<f32, Chip, Cluster, EpilogueSlices, m![L / 7680, H % 240]>;

/// Writes the transfer's destination once, locally, in the prologue (from the replicated x, dead after
/// its TRF load; the values are overwritten by the transfer). A buffer that cluster 1 writes gets a
/// destination-readiness `cluster_sync` after its last local write; for a fresh buffer that is the
/// kernel start, where the ~1.9k sync command sat in front of the second DMA and left the queue idle
/// for ~1.2k on both clusters. Here it follows this pass, while the Up weight streams and nothing
/// else waits to be issued.
fn init_epilogue_partials(
    device: &mut Device,
    x: DmTensor<f8e4m3, Chip, UpCluster, IpRows, m![Gs, H]>,
) -> EpiloguePartials {
    let x: DmTensor<f8e4m3, Chip, Cluster, EpilogueSlices, m![L / 7680, Epi16, H % 240]> = unsafe { x.reshape() };
    device.main
        .begin(x.view().tile::<m![Epi16], 1, m![L / 7680, Epi16 = 1 # 16, H % 240]>(0))
        .fetch::<m![L / 7680, H / 8 % 30], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 7680, H / 8 % 30], m![H % 8]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// Up: one full 30-row tile; decoded once, with scales consumed in six-row passes.
fn project_up_gate_small_first(
    device: &mut Device,
    x: &UpActivation,
    weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> UpResult {
    let w0 = load_up_gate_tile30(device, weight, 0);
    let scale = load_up_gate_scale4(device, scale);
    let mut result: UpResult = DmTensor::new();
    let mut blocks0: UpBlocks = DmTensor::new();
    up_gate_tile30(device, x, w0, &mut blocks0, 0);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 0);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 6);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 12);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 18);
    up_gate_scale_pass4(device, &blocks0, &scale, &mut result, 24);
    result
}

/// Gate: one full 30-row tile, reducing LUT loads and DMA command startup.
fn project_up_gate_small_last(
    device: &mut Device,
    x: &UpActivation,
    weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> UpResult {
    let w0 = load_up_gate_tile30(device, weight, 0);
    let scale = load_up_gate_scale(device, scale);
    let mut result: UpResult = DmTensor::new();
    let mut blocks0: UpBlocks = DmTensor::new();
    up_gate_tile30(device, x, w0, &mut blocks0, 0);
    up_gate_scale_pass8(device, &blocks0, &scale, &mut result, 0);
    up_gate_scale_pass8(device, &blocks0, &scale, &mut result, 8);
    up_gate_scale_pass8(device, &blocks0, &scale, &mut result, 16);
    up_gate_scale_pass(device, &blocks0, &scale, &mut result, 24);
    result
}


/// The 7,680 Up (or Gate) values of a cluster on 64 of its slices, 120 per slice: the same rows
/// the contraction produced, regrouped in fours. 120 elements is what makes every packet whole
/// -- 8 f32 is exactly one 32 B flit, and 120 bf16 is 240 B -- where the contraction's own 30
/// rows are neither (30 bf16 = 60 B is not a multiple of the 8 B SRAM access width, and 30 f32
/// is 3.75 flits). This is a DM-to-DM permutation inside the cluster, not a gather: no HBM.
type UpPairedRows = m![L / 240 % 32, 1 # 8];
type UpPaired = DmTensor<f32, Chip, UpCluster, UpPairedRows, m![L % 240]>;

/// Gather the eight interleaved row-pair slices of each 240-row block onto its first slice.
/// GeGLU runs on 32 such slices per cluster. Down consumes the same cluster's K-half, so the
/// resulting activations stay in DM and reach every Down slice through the switch network.
fn gather_up_gate(device: &mut Device, x: IpResult) -> UpPaired {
    // IP layout: the 8 slices of a 240-row block onto its first slice, [g, c, pair] = rows in order.
    device.main
        .begin(x.view())
        .fetch::<m![L / 16 % 15], m![L % 2 # 8]>()
        .switch::<UpPairedRows, m![L / 16 % 15, L / 2 % 8]>(
            SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 },
        )
        .collect::<m![L / 16 % 15, L / 2 % 8], m![L % 2 # 8]>()
        .commit_trim::<m![L % 2]>()
        .commit()
}

fn geglu_local(
    device: &mut Device,
    up: IpResult,
    gate: IpResult,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DownActivation {
    let up = gather_up_gate(device, up);
    let gate = gather_up_gate(device, gate);

    // The Up global scale is only a factor of the product, so it is applied in the Down
    // epilogue's RMS (the Down contraction is linear): no replicated tiny load to 64
    // slices per cluster between the weight streams. The gate one sits inside the erf and stays.
    // Global-scale VRF written from Main through the vector engine (x 1.0, exact): an eager Sub
    // `to_vrf` right after the tiny DMA was listed before the next weight DMA, which on hardware was
    // then not issued until the tiny load had landed. (A Main store straight off `collect` hangs.)
    // Loaded onto ONE slice per cluster (one DMA engine, one destination each, instead of four
    // engines x eight destinations for the 32 GeGLU slices: ~1.76k of DMA queue at the Gate/Up
    // junction) and spread along the stride-8 ring of 32 slices by a Main switch pass.
    let gate_scale_seed: DmTensor<f32, Chip, UpCluster, m![1 # 32, 1 # 8], m![1 # 8]> =
        gate_global_scale.to_dm(&mut device.tdma);
    let gate_scale: DmTensor<f32, Chip, UpCluster, UpPairedRows, m![1 # 32, 1 # 8]> = device
        .main
        .begin(gate_scale_seed.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<UpPairedRows, m![1 # 32]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 8 })
        .collect::<m![1 # 32], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let gate_scale_vrf: VrfTensor<f32, Chip, UpCluster, UpPairedRows, m![1 # 8]> = device
        .main
        .begin(gate_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);

    let gate_scaled: UpPaired = device
        .main
        .begin(gate.view())
        .fetch::<m![L / 8 % 30], m![L % 8]>()
        .collect::<m![L / 8 % 30], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 60], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 30], m![L % 8]>()
        .vector_final()
        .commit_trim::<m![L % 8]>()
        .commit();
    // g * (1 + erf(g / sqrt(2))); GeGLU's factor 1/2 is folded into the product below.
    let gelu: UpPaired = device
        .sub
        .begin(gate_scaled.view())
        .fetch::<m![L / 8 % 30], m![L % 8]>()
        .collect::<m![L / 8 % 30], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 60], m![L % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L / 8 % 30], m![L % 8]>()
        .vector_final()
        .commit_trim::<m![L % 8]>()
        .commit();
    let gelu_vrf: VrfTensor<f32, Chip, UpCluster, UpPairedRows, m![L % 240]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![L / 8 % 30], m![L % 8]>()
        .collect::<m![L / 8 % 30], m![L % 8]>()
        .to_vrf();
    let out: DmTensor<bf16, Chip, UpCluster, UpPairedRows, m![L % 240]> = device
        .main
        .begin(up.view())
        .fetch::<m![L / 8 % 30], m![L % 8]>()
        .collect::<m![L / 8 % 30], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 60], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L / 8 % 30], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();
    // First replicate each 240-value shard over its eight-slice group, then all-gather the
    // 32 shards along stride-eight rings. Main is substantially faster than Sub for this
    // switch/commit pass. Pack 24 useful FP8 values per flit (240 is divisible by 24).
    // The first pass retains the seven inactive source slots as padding; the next fetch
    // reads only the valid slot, and every destination receives the complete local K-half.
    let out = split_down_activation(device, &out);
    let rep: DmTensor<f8e4m3, Chip, UpCluster, m![L / 240 % 32, Dummy8], m![Gs, L % 240 / 24, 1 # 8, L % 24]> = device.main
        .begin(out.view())
        .fetch::<m![Gs, L % 240 / 24], m![L % 24]>()
        .switch::<m![L / 240 % 32, Dummy8], m![Gs, L % 240 / 24, 1 # 8]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![Gs, L % 240 / 24, 1 # 8], m![L % 24 # 32]>()
        .commit_trim::<m![L % 24]>().commit();
    let all: DmTensor<f8e4m3, Chip, UpCluster, m![Dummy256 / 8, Dummy8], m![Gs, L % 7680]> = device.main
        .begin(rep.view())
        .fetch::<m![Gs, L % 240 / 24], m![L % 24]>()
        .switch::<m![Dummy256 / 8, Dummy8], m![Gs, L % 240 / 24, L / 240 % 32]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 8 })
        .collect::<m![Gs, L % 240 / 24, L / 240 % 32], m![L % 24 # 32]>()
        .commit_trim::<m![L % 24]>().commit();
    // All 256 slices now contain identical data. Relabeling their positions preserves the
    // wire order, and DownCluster equals UpCluster's L / 7680 partition.
    let all: DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![Gs, L % 7680]> = unsafe { all.reshape() };
    // TRF load on Main (idle after the all-gather; ~3x faster per flit than Sub): the decode is
    // then released by w0's landing instead of a Sub TRF load that ended 1.6k after it, and the
    // block-scale DMA listed behind the decode enters the queue as soon as w0 is done.
    device.main.begin(all.view())
        .fetch::<m![Gs, L / 32 % 240], m![L % 32]>()
        .collect::<m![Gs, L / 32 % 240], m![L % 32]>()
        .to_trf()
}

pub(crate) fn feedforward(
    device: &mut Device,
    x: NormalizedInput,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    span: FfSpan,
) -> DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![H % 240]> {
    let (up, gate, partials) = project_up_and_gate(
        device,
        &x,
        up_weight_packed,
        gate_weight_packed,
        up_weight_scale,
        gate_weight_scale,
    );
    let residual = relayout_residual(device, span_tile(&span, OP_RESIDUAL), x);
    let x = geglu_local(device, up, gate, gate_global_scale);
    // The epilogue's small operands, issued here so they sit on the DMA queue long before the
    // tail needs them.
    let norm_weight: DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![H % 240]> = post_ff_rms_weight.to_dm(&mut device.tdma);
    let global_scale: DmTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = down_global_scale.to_dm(&mut device.tdma);
    let up_scale: DmTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = up_global_scale.to_dm(&mut device.tdma);
    let layer_scalar: DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![1 # 8]> = layer_scalar.to_dm(&mut device.tdma);
    let down = project_down(device, &x, down_weight_packed, down_weight_scale);
    down_epilogue(device, down, partials, &global_scale, &up_scale, &norm_weight, &residual, &layer_scalar)
}

type DownCluster = m![L / 7680];
// Down row h = 60 g + 4 r + q  (g = H / 60, r = H % 60 / 4 in 0..15, q = H % 4).
axes![KhG = 64, KhR = 15, KhQ = 4];
/// Both clusters own all H rows, with one 7,680-element K-half per cluster. Within a cluster,
/// 64 groups of four slices hold interleaved rows h = 60 g + 4 r + q, r in 0..15.
/// Packed weight pieces remain 3,840 B and block-scale pieces 480 B. Only the completed
/// H-sized partial sums cross clusters, in the epilogue.
type DownRows = m![H / 60, Dummy8 / 2];
type DownRowsH = m![H / 60, H % 4];
type DownActivation = TrfTensor<f8e4m3, Chip, DownCluster, DownRows, m![1], m![Gs, L % 7680]>;
type DownResult = DmTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4, 1 # 2]>;
type DownGatheredRows = m![H / 60, 1 # 4];
type DownReducedRows = m![H / 60, 1 # 4];
type DownReduced = DmTensor<f32, Chip, DownCluster, DownReducedRows, m![H % 60]>;

macro_rules! down_tile_fns {
    ($load:ident, $tile:ident, $rows:literal) => {
        fn $load(device: &mut Device, weight: &HbmTensor<f4e2m1, Chip, m![H, L]>, r0: usize)
        -> DmTensor<f4e2m1, Chip, DownCluster, DownRows, m![H % 60 / 4 = $rows, L % 7680]> {
            // Relabel H as (KhG, KhR, KhQ) so the row window sits on an axis the view names directly
            // (0.8.1 rejects a non-divisor window on an inner digit of H).
            let v: HbmTensorView<'_, f4e2m1, Chip, m![KhG, KhR, KhQ, L]> = unsafe { weight.view().reshape() };
            let d: DmTensor<f4e2m1, Chip, m![L / 7680], m![KhG, KhQ], m![KhR = $rows, L % 7680]> =
                v.tile::<m![KhR], $rows, m![KhG, KhR = $rows # 15, KhQ, L]>(r0).to_dm(&mut device.tdma);
            unsafe { d.reshape() }
        }
        fn $tile(device: &mut Device,
            packed: DmTensor<f4e2m1, Chip, DownCluster, DownRows, m![H % 60 / 4 = $rows, L % 7680]>)
            -> DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = $rows, L % 7680]> {
            device.main.begin(packed.view())
                .fetch::<m![H % 60 / 4 = $rows, L / 3840 % 2], m![L % 3840]>()
                .fetch_table_lookup::<f8e4m3>()
                .collect::<m![H % 60 / 4 = $rows, L / 32 % 240], m![L % 32]>()
                .commit_trim::<m![L % 32]>().commit()
        }
    };
}
down_tile_fns!(load_down_tile3, decode_down_tile3_unused, 3);
down_tile_fns!(load_down_tile12, decode_down_tile12, 12);

macro_rules! down_scale_fn {
    ($name:ident, $rows:literal) => {
        fn $name(
            device: &mut Device,
            scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
            r0: usize,
        ) -> DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = $rows, L / 16 % 480]> {
            let v: HbmTensorView<'_, f8e4m3, Chip, m![KhG, KhR, KhQ, L / 16]> = unsafe { scale.view().reshape() };
            let d: DmTensor<f8e4m3, Chip, m![L / 7680], m![KhG, KhQ], m![KhR = $rows, L / 16 % 480]> =
                v.tile::<m![KhR], $rows, m![KhG, KhR = $rows # 15, KhQ, L / 16]>(r0).to_dm(&mut device.tdma);
            unsafe { d.reshape() }
        }
    };
}
down_scale_fn!(load_down_scale4, 4);
down_scale_fn!(load_down_scale3, 3);
down_scale_fn!(load_down_scale8, 8);

macro_rules! down_stream_group_fn {
    ($name:ident, $tile_rows:literal, $group:literal) => {
        fn $name(
            device: &mut Device,
            x: &DownActivation,
            fp8: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = $tile_rows, L % 7680]>,
            tile_offset: usize,
            scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = $group, L / 16 % 480]>,
            result: &mut DownResult,
            result_offset: usize,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = $group, L / 16 % 480]> = device
                .sub
                .begin(scale.view())
                .fetch::<m![H % 60 / 4 = $group, L / 16 / 240 % 2], m![L / 16 % 240]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 / 4 = $group, L / 16 / 8 % 60], m![L / 16 % 8]>()
                .to_vrf();
            device.main
                .begin(fp8.view().tile::<m![H % 60 / 4 = $tile_rows], $group, m![H % 60 / 4 = $group # $tile_rows, L % 7680]>(tile_offset))
                .fetch::<m![H % 60 / 4 = $group, L / 64 % 120, L / 32 % 2], m![L % 32]>()
                .collect::<m![H % 60 / 4 = $group, L / 64 % 120, L / 32 % 2], m![L % 32]>()
                .contract_outer::<m![H % 60 / 4 = $group, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
                .contract_packet::<m![L / 16 % 4]>()
                .contract_time::<m![H % 60 / 4 = $group, L / 64 % 120]>()
                .contract_lane::<m![H % 60 / 4 = $group, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![L / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_intra_slice_reduce::<L, m![H % 60 / 4 = $group], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_fp_div(DOWN_ACTIVATION_SCALE)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                // One row per 8 B SRAM write unit: [value, pad].
                .commit_trim::<m![1 # 2]>()
                .commit_view(result.view_mut().tile::<m![H % 60 / 4], $group, m![H % 60 / 4 = $group #{!} 15, 1 # 2]>(result_offset));
        }
    };
}
down_stream_group_fn!(down_stream_group4, 12, 4);

/// Group of 4 rows whose block scales are a 4-row tile of an 8-row scale load (rows 4..11 in one DMA).
fn down_stream_group4_s8(
    device: &mut Device,
    x: &DownActivation,
    fp8: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 12, L % 7680]>,
    tile_offset: usize,
    scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 8, L / 16 % 480]>,
    scale_offset: usize,
    result: &mut DownResult,
    result_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = 4, L / 16 % 480]> = device
        .sub
        .begin(scale.view().tile::<m![H % 60 / 4 = 8], 4, m![H % 60 / 4 = 4 # 8, L / 16 % 480]>(scale_offset))
        .fetch::<m![H % 60 / 4 = 4, L / 16 / 240 % 2], m![L / 16 % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 / 4 = 4, L / 16 / 8 % 60], m![L / 16 % 8]>()
        .to_vrf();
    device.main
        .begin(fp8.view().tile::<m![H % 60 / 4 = 12], 4, m![H % 60 / 4 = 4 # 12, L % 7680]>(tile_offset))
        .fetch::<m![H % 60 / 4 = 4, L / 64 % 120, L / 32 % 2], m![L % 32]>()
        .collect::<m![H % 60 / 4 = 4, L / 64 % 120, L / 32 % 2], m![L % 32]>()
        .contract_outer::<m![H % 60 / 4 = 4, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
        .contract_packet::<m![L / 16 % 4]>()
        .contract_time::<m![H % 60 / 4 = 4, L / 64 % 120]>()
        .contract_lane::<m![H % 60 / 4 = 4, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![L / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<L, m![H % 60 / 4 = 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DOWN_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit_view(result.view_mut().tile::<m![H % 60 / 4], 4, m![H % 60 / 4 = 4 #{!} 15, 1 # 2]>(result_offset));
}

/// The last 3-row tile: FP4 -> FP8 lookup fused into the group's contraction (no decode pass, no FP8
/// buffer); x hi/lo is a stream-adapter time broadcast as in Up/Gate.
fn down_fused_group3(
    device: &mut Device,
    x: &DownActivation,
    packed: &DmTensor<f4e2m1, Chip, DownCluster, DownRows, m![H % 60 / 4 = 3, L % 7680]>,
    scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 3, L / 16 % 480]>,
    result: &mut DownResult,
    result_offset: usize,
) {
    // Written from Main through the vector engine (x 1.0, exact): a Sub `to_vrf` starts at the
    // scale's static landing, ties with the table DMA there and is listed first, so the core
    // waited for the scale before issuing the table (first-after-idle, ~2k, then w1). On Main,
    // which is still decoding w0 at that static time, the table DMA is listed right behind the
    // scale and queues behind it.
    let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = 3, L / 16 % 480]> = device
        .main
        .begin(scale.view())
        .fetch::<m![H % 60 / 4 = 3, L / 16 / 240 % 2], m![L / 16 % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 / 4 = 3, L / 16 / 8 % 60], m![L / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 60 / 4 = 3, L / 16 / 4 % 120], m![L / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1f32)
        .vector_widen_concat::<m![H % 60 / 4 = 3, L / 16 / 8 % 60], m![L / 16 % 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
    device.main
        .begin(packed.view())
        .fetch::<m![H % 60 / 4 = 3], m![L % 7680]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60 / 4 = 3, L / 64 % 120, L / 32 % 2], m![L % 32]>()
        .contract_outer::<m![H % 60 / 4 = 3, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
        .contract_packet::<m![L / 16 % 4]>()
        .contract_time::<m![H % 60 / 4 = 3, L / 64 % 120]>()
        .contract_lane::<m![H % 60 / 4 = 3, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![L / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<L, m![H % 60 / 4 = 3], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DOWN_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit_view(result.view_mut().tile::<m![H % 60 / 4], 3, m![H % 60 / 4 = 3 #{!} 15, 1 # 2]>(result_offset));
}


// The first 12 rows share one FP4 lookup/contraction and one diagonal scale contraction.
// BF16 rounds each block-16 partial once, as in the Up/Gate diagonal path. Scales convert
// exactly from FP8 to BF16. The final three rows retain their fused FP4/VRF pass, allowing
// their weight DMA to overlap the first tile instead of delaying a single 15-row tile.
down_scale_fn!(load_down_scale_diag, 12);

type DownDiagWeight = DmTensor<f4e2m1, Chip, DownCluster, DownRows, m![H % 60 / 4 = 12, L % 7680]>;
type DownDiagBlocks = DmTensor<bf16, Chip, DownCluster, DownRows, m![H % 60 / 4 = 12, L / 16 % 480]>;
type DownDiagScale = DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 12, L / 16 % 480]>;
type DownDiagTrf = TrfTensor<bf16, Chip, DownCluster, DownRows, m![1], m![H % 60 / 4 = 12, L / 16 % 480]>;

/// FP8 -> BF16 conversion and the TRF load both on Main, right after the 12-row decode: Main is
/// ~3.5x faster per flit than Sub for these passes and the finish that consumes the TRF is not on
/// the critical path. With the conversion on Sub, the Sub context was still busy when the table
/// DMA became due, and the lowering then drained Sub before that DMA (`raw.wait sub_context` with
/// no source), which held the table and w1 DMAs 2.6k behind an idle queue.
fn down_diag_scale(device: &mut Device, s: &DownDiagScale) -> DownDiagTrf {
    let s: DownDiagBlocks = device.main.begin(s.view())
        .fetch::<m![H % 60 / 4 = 12, L / 16 / 8 % 60], m![L / 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 / 4 = 12, L / 16 / 8 % 60], m![L / 16 % 8]>()
        .commit_trim::<m![L / 16 % 8]>()
        .commit_cast::<bf16>()
        .commit();
    device.main.begin(s.view())
        .fetch::<m![H % 60 / 4 = 12, L / 16 / 16 % 30], m![L / 16 % 16]>()
        .collect::<m![H % 60 / 4 = 12, L / 16 / 16 % 30], m![L / 16 % 16]>()
        .to_trf()
}

fn down_diag_blocks(device: &mut Device, x: &DownActivation, packed: &DownDiagWeight) -> DownDiagBlocks {
    device.main.begin(packed.view())
        .fetch::<m![H % 60 / 4 = 12], m![L % 7680]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60 / 4 = 12, L / 64 % 120, L / 32 % 2], m![L % 32]>()
        .contract_outer::<m![H % 60 / 4 = 12, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
        .contract_packet::<m![L / 16 % 4]>()
        .contract_time::<m![H % 60 / 4 = 12, L / 64 % 120]>()
        .contract_lane::<m![H % 60 / 4 = 12, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
        .commit_trim::<m![L / 16 % 4]>()
        .commit_cast::<bf16>().commit()
}

fn down_diag_finish(device: &mut Device, b: &DownDiagBlocks, s: &DownDiagTrf, result: &mut DownResult) {
    device.main.begin(b.view())
        .fetch::<m![H % 60 / 4 = 12, L / 16 / 16 % 30], m![L / 16 % 16]>()
        .collect::<m![H % 60 / 4 = 12, L / 16 / 16 % 30], m![L / 16 % 16]>()
        .contract_outer::<m![H % 60 / 4 = 12, L / 16 / 16 % 30], m![L / 16 % 16], _, _, _>(s)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60 / 4 = 12]>()
        .contract_lane::<m![H % 60 / 4 = 12], m![1 # 8]>(LaneMode::Sequential)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_div(DOWN_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit_view(result.view_mut().tile::<m![H % 60 / 4], 12, m![H % 60 / 4 = 12 #{!} 15, 1 # 2]>(0));
}

/// Down projection of the local K-half; returns one f32 partial sum per output channel.
fn project_down(
    device: &mut Device,
    x: &DownActivation,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DownReduced {
    // Weight tiles stream independently of the GeGLU hand-over; x is already in local TRF.
    let w0 = load_down_tile12(device, down_weight_packed, 0);
    let scale = load_down_scale_diag(device, down_weight_scale, 0);
    let scale = down_diag_scale(device, &scale);
    let blocks = down_diag_blocks(device, x, &w0);
    let mut result: DownResult = DmTensor::new();
    down_diag_finish(device, &blocks, &scale, &mut result);
    let w1 = load_down_tile3(device, down_weight_packed, 12);
    let scale3 = load_down_scale3(device, down_weight_scale, 12);
    down_fused_group3(device, x, &w1, &scale3, &mut result, 12);
    let result: DmTensor<f32, Chip, DownCluster, DownRowsH, m![H % 60 / 4, 1 # 2]> = unsafe { result.reshape() };
    // Gather each group's four interleaved row slices into rows 4r+q. The other K-half
    // belongs to the other cluster and is added after the DM-to-DM transfer below.
    device.main
        .begin(result.view())
        .fetch::<m![H % 60 / 4], m![1 # 2]>()
        .switch::<DownGatheredRows, m![H % 60 / 4, H % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 1 })
        .collect::<m![H % 60 / 4, H % 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_final()
        .transpose::<m![H % 60 / 4, H / 2 % 2], m![H % 2 # 8]>()
        .commit_trim::<m![H % 2]>()
        .commit()
}


/// Down global scale, post-FFN RMSNorm, residual add and layer gate on 8 slices x 480 channels.
///
/// Replaces: gather to one slice, HBM round trip, f32 -> bf16 cast pass, global-scale pass,
/// `rmsnorm::normalize` (its own 8-slice relayout and 1-slice broadcast back), a 2-tile residual
/// add and a layer-gate pass -- 14k static cycles of serial small operations after the last
/// contraction. Rounding differs from that chain in two places, each by at most one bf16 ulp:
/// `bf16(y * gs)` instead of `bf16(bf16(y) * gs)`, and `bf16((n + r) * gate)` instead of
/// `bf16(bf16(n + r) * gate)`. The normalized value is still rounded to bf16 before the add.
fn down_epilogue(
    device: &mut Device,
    down: DownReduced,
    mut y2: EpiloguePartials,
    global_scale: &DmTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]>,
    up_scale: &DmTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]>,
    norm_weight: &DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![H % 240]>,
    residual: &DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![H % 240]>,
    layer_scalar: &DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![H % 240]> {
    // Transfer both clusters' 64 row-group partials into cluster 0's 16 epilogue slices.
    // Keep the two K-halves in a separate dimension, then sum them in one Main pass.
    // This avoids staging one half in VRF and a second vector pass to add it.
    down.view().to_dm_view(&mut device.tdma, y2.view_mut());
    let y: DmTensor<f32, Chip, Cluster, EpilogueSlices, m![H % 240]> = device.main
        .begin(y2.view())
        .fetch::<m![H / 4 % 60, L / 7680], m![H % 4]>()
        .collect::<m![H / 4 % 60, L / 7680], m![H % 4 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H % 4]>()
        .vector_intra_slice_reduce::<L, m![H / 4 % 60], m![H % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![H % 4 # 8]>().vector_final()
        .commit_trim::<m![H % 4]>().commit();

    let global_scale_vrf: VrfTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = device
        .sub
        .begin(global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    // The GeGLU arrives without the Up global scale `su`, so y here is y_true / su: the mean square
    // is taken of y * gs * su (= y_true * gs), and su rides in the final pass's scalar gain below,
    // so y * gain * w / sqrt(ms + eps) = y_true * gs * w * gate / sqrt(ms_true + eps) as before.
    let up_scale_vrf: VrfTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = device
        .sub
        .begin(up_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let layer_scalar_vrf: VrfTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = device
        .sub
        .begin(layer_scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    // Operands the final pass needs, prepared on Sub while Main is still contracting:
    // the residual pre-multiplied by the layer gate (exact: bf16 x bf16 fits f32).
    // Written straight into the VRF by the Sub vector pass (no f32 DM copy and no separate
    // `to_vrf` pass): two Sub passes instead of four ahead of the y hand-off's cluster release.
    let residual_gated_vrf: VrfTensor<f32, Chip, Cluster, EpilogueSlices, m![H % 240]> = device
        .sub
        .begin(residual.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &layer_scalar_vrf)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        .to_vrf();
    // ... and the norm weight as a plain f32 VRF (no vector pass, so it does not wait for Main's
    // vector work).
    let weight_vrf: VrfTensor<f32, Chip, Cluster, EpilogueSlices, m![H % 240]> = device
        .sub
        .begin(norm_weight.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();
    // The final pass's scalar gain = gate * gs * su (one Sub vector pass on three scalars that land
    // long before the tail): the numerator then needs only Mul0 (gain) and Mul1 (w), which leaves
    // FpFma free for the inline rms (`h + h`) and FpFpu for its sqrt.
    let gain_vrf: VrfTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = device
        .sub
        .begin(layer_scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &global_scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &up_scale_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf();

    // Two passes after the partial sum on the serial tail. (1) half mean of (y * gs * su)^2 from the
    // f32 sums, summed over the 8 slices, + eps / 2 = h, written into VRF; (2) y * gain * w
    // / sqrt(h + h) + r * gate -> bf16, the rms rebuilt inline (same
    // idiom: stash the numerator, h + h, sqrt, stash / rms on the FpDiv stage). Against the
    // previous chain this drops the bf16 rounding of `y * gs` before both the square and the
    // normalization, and of the normalized value before the residual add: each at most one bf16
    // ulp of the output, the same order as the roundings that chain already skipped.
    let partial_mean_square: DmTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = device
        .main
        .begin(y.view())
        .fetch::<m![H / 8 % 30], m![H % 8]>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &global_scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Fma), &up_scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(2.0f32 * H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // furiosa-opt 0.8.1 prices an inter-slice reduce over these 8 slices at ~2.3k static: all-gather
    // the 8 per-slice partials into Time (named Switch config, no config DMA) and sum them with the
    // intra-slice reducer, which leaves the total on every one of the 8 slices.
    let half_ms: VrfTensor<f32, Chip, Cluster, m![1 # 16, Epi16], m![1 # 8]> = device
        .main
        .begin(partial_mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![1 # 16, Epi16], m![H / 240]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![H / 240], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, 0.5f32 * EPS)
        .vector_final()
        .to_vrf(&mut device.sub);
    let half_ms: VrfTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = unsafe { half_ms.reshape() };

    device.main
        .begin(y.view())
        .fetch::<m![H / 8 % 30], m![H % 8]>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gain_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_stash()
        .vector_fp_binary_with_mode(FpBinaryOp::AddF, BinaryArgMode::Mode11, &half_ms)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, Stash)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &residual_gated_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// Normalize independently in each cluster's eight-slice seed group. The RMS reduction,
/// division, weight multiplication and BF16 rounding match shared::rmsnorm::normalize.
/// Returning the shards directly avoids gathering to a slice and staging through HBM.
pub(crate) fn normalize_input(device: &mut Device, span: &FfSpan) -> NormalizedInput {
    type ReducingSlices = UpPairRows;

    let reduced_mean_square: DmTensor<f32, Chip, UpCluster, m![1 # 32, Dummy8], m![1 # 8]> = device
        .main
        .begin(span_tile(span, OP_RESIDUAL))
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

    let rms: DmTensor<f32, Chip, UpCluster, m![1 # 32, Dummy8], m![1 # 8]> = device
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
    let rms: DmTensor<f32, Chip, UpCluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    // A Main copy of the weight (Main is busy with the reduce when it lands) so the Sub `to_vrf` below
    // is listed after the Up weight DMA: on hardware a Sub command listed before that DMA waits behind
    // the head lookup-table StoTab (5.7k cold) and holds the Up weight issue.
    // The copy writes into the first slot of a buffer cleared by one Sub memset whose static time
    // outlasts the weight DMA, so the copy (the weight's first consumer) no longer ties with the Up
    // scale DMA at the weight's static landing: the scale DMA is listed right behind the weight and
    // queued before it lands instead of ~0.6k after.
    let mut weight_buf: DmTensor<bf16, Chip, UpCluster, ReducingSlices, m![WmSlot, H % 480]> = DmTensor::new();
    weight_buf.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    device
        .main
        .begin(span_tile(span, OP_PRE_RMS))
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .collect::<m![H / 16 % 30], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit_view(weight_buf.view_mut().tile::<m![WmSlot], 1, m![WmSlot = 1 #{!} 4, H % 480]>(0));
    let weight_vrf: VrfTensor<f32, Chip, UpCluster, ReducingSlices, m![H % 480]> = device
        .sub
        .begin(weight_buf.view().tile::<m![WmSlot], 1, m![WmSlot = 1 # 4, H % 480]>(0))
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, UpCluster, ReducingSlices, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, UpCluster, ReducingSlices, m![H % 480]> = device
        .main
        .begin(span_tile(span, OP_RESIDUAL))
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

    // The raw seed stays live in the span as the epilogue's residual operand (see `feedforward`).
    normalized
}
