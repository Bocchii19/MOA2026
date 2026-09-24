
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Dummy2, Dummy256, H, L};
use crate::device::layout::{Cluster, Replicated};
use crate::device::shared::hidden::{HiddenChunks, LiveRows};

const INVSQRT2: f32 = 0.70710678118f32;
const UPGATE_HILO_SCALE: f32 = 64.0;

// The feed-forward block runs on both clusters. The L axis (15360 rows of up/gate, 15360
// columns of down) is split in halves across the two clusters, so every weight load moves half
// the bytes per cluster and every slice holds half the rows.
//
// * Up/gate: row l = 7680 c + 30 s + r lives on cluster c, slice s (30 whole rows a slice),
//   decoded f4 -> bf16 and contracted against the replicated activation.
// * GeGLU: eight consecutive slices pool their 30 rows onto one slice (240 rows per live slice).
// * Down: cluster c contracts its half of L on 256 slices of 15 whole rows (3,840-byte aligned
//   weight pieces). The f4 rows stream straight through the f4 -> f8 lookup into the contraction
//   against GeGLU's output held as `hi + lo` f8 lanes (no decode pass after the down_w load); a
//   vector pass sums the two lanes' block partials and a second contraction folds the block
//   scales. The eight-slice pools of each cluster's row sums cross to cluster 0 through HBM and
//   reload straight into `HiddenChunks`, where the tail runs on 32 slices.
// * Every scale decode runs on the sub context so it streams under the main-context passes.
pub(crate) type Cluster2 = m![L / 7680];
pub(crate) type UpGateRows = m![L / 30 % 256];
pub(crate) type UpGateRowsPaired = m![L / 240 % 32, 1 # 8];

type DownSl = m![H / 15];
type DownW = m![H % 15, L % 7680];
type DownS = m![H % 15, L % 7680 / 16];



/// A matrix half's packed rows (30 per slice) and its block scales.
type UpGateChunk = m![L % 30, H];
type UpGateChunkScale = m![L % 30, H / 16];

macro_rules! load_up_gate_chunk {
    ($ctx:expr, $weight_packed:expr, $weight_scale:expr) => {{
        let w: DmTensor<f4e2m1, Chip, Cluster2, UpGateRows, UpGateChunk> =
            $weight_packed.to_dm(&mut $ctx.tdma);
        let s: DmTensor<f8e4m3, Chip, Cluster2, UpGateRows, UpGateChunkScale> =
            $weight_scale.to_dm(&mut $ctx.tdma);
        (w, s)
    }};
}

/// Direct NVFP4 -> FP8 contraction for one up/gate matrix.  The normalized activation is
/// already replicated on all 256 slices, so its shared hi/lo encoding needs no switch-ring
/// all-gather.  Lane reduction and the reciprocal fixed scale are fused into the contraction
/// epilogue; the BF16 weight materialization pass is removed.
macro_rules! up_gate_chunk {
    ($ctx:expr, $x_trf:expr, $chunk:expr, $out:expr) => {{
        let summed: DmTensor<bf16, Chip, Cluster2, UpGateRows, m![L % 30, H / 64, H % 64 / 16]> = $ctx
            .main
            .begin($chunk.0.view())
            .fetch::<m![L % 30, H / 64], m![H % 64]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![L % 30, H / 32], m![H % 32]>()
            .contract_outer::<m![L % 30, H / 64], m![H % 64], _, _, _>($x_trf)
            .contract_packet::<m![H % 64 / 16]>()
            .contract_time::<m![L % 30, H / 64]>()
            .contract_lane::<m![L % 30, H / 64, Dummy2], m![H % 64 / 16 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![H % 64 / 16]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0 / UPGATE_HILO_SCALE)
            .vector_intra_slice_reduce::<Dummy2, m![L % 30, H / 64], m![H % 64 / 16]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![H % 64 / 16 # 8]>()
            .vector_final()
            // Move the existing BF16 boundary to the producer; compact DM only.
            .commit_trim::<m![H % 64 / 16]>()
            // Commit Adapter preserves the rounding while freeing Cast Engine.
            .commit_cast::<bf16>()
            .commit();
        let partials: DmTensor<bf16, Chip, Cluster2, UpGateRows, m![L % 30, H / 16]> =
            unsafe { summed.reshape() };

        // Stage 2: y[l] = sum_b y1[l, b] * s[l, b], with the scales stationary in the TRF
        // (f8 -> bf16 is a table lookup, which only the main context has). The output stays
        // f32: 30 rows per slice is not a multiple of 4, so the transposer packs 2 rows.
        let scale_bf16: DmTensor<bf16, Chip, Cluster2, UpGateRows, UpGateChunkScale> = $ctx
            .sub
            .begin($chunk.1.view())
            .fetch::<m![L % 30, H / 16 / 16], m![H / 16 % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 30, H / 16 / 8], m![H / 16 % 8]>()
            .cast::<bf16, m![H / 16 % 8 # 16]>()
            .commit_trim::<m![H / 16 % 8]>()
            .commit();
        let scale_trf: TrfTensor<bf16, Chip, Cluster2, UpGateRows, m![1], UpGateChunkScale> = $ctx
            .sub
            .begin(scale_bf16.view())
            .fetch::<m![L % 30, H / 16 / 16], m![H / 16 % 16]>()
            .collect::<m![L % 30, H / 16 / 16], m![H / 16 % 16]>()
            .to_trf();

        $ctx.main
            .begin(partials.view())
            .fetch::<m![L % 30, H / 16 / 8], m![H / 16 % 8]>()
            .collect::<m![L % 30, H / 16 / 8], m![H / 16 % 8 # 16]>()
            .contract_outer::<m![L % 30, H / 16 / 8], m![H / 16 % 8 # 16], _, _, _>(&scale_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![L % 30]>()
            .contract_lane::<m![L % 30], m![1 # 8]>(LaneMode::Interleaved)
            .transpose::<m![L % 30 / 2], m![L % 30 % 2 # 8]>()
            .commit_trim::<m![L % 30 % 2]>()
            .commit_view($out.view_mut());
    }};
}

/// Encode the replicated normalized hidden vector once and reuse it for both up and gate.
/// The fixture maximum is 1.703125 (109 after scaling), while FP8 e4m3fn reaches 448; the
/// fixed power-of-two scale therefore has ample headroom and an exact reciprocal.
fn encode_up_gate_replicated(
    ctx: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster2, Replicated, m![H]>,
) -> TrfTensor<f8e4m3, Chip, Cluster2, UpGateRows, m![Dummy2], m![H]> {
    let x_rows: DmTensorView<'_, bf16, Chip, Cluster2, UpGateRows, m![H / 1920, H % 1920]> =
        unsafe { x.view().reshape() };
    let mut lanes: DmTensor<
        f8e4m3,
        Chip,
        Cluster2,
        UpGateRows,
        m![Dummy2, H / 1920, H % 1920],
    > = DmTensor::new();

    macro_rules! encode_half {
        ($half:expr) => {{
        let x_half = x_rows
            .tile::<m![H / 1920], 1, m![H / 1920 = 1 # 2, H % 1920]>($half);
        ctx.main
            .begin(x_half)
            .fetch::<m![H % 1920 / 8], m![H % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 1920 / 8], m![H % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 1920 / 4], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), UPGATE_HILO_SCALE)
            .vector_widen_concat::<m![H % 1920 / 8], m![H % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![H % 8 # 32]>()
            .commit_trim::<m![H % 8]>()
            .commit_view(
                lanes
                    .view_mut()
                    .tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, H / 1920, H % 1920]>(0)
                    .tile::<m![H / 1920], 1, m![Dummy2 = 1 #{!} 2, H / 1920 = 1 #{!} 2, H % 1920]>($half),
            );

        let hi_half_raw = lanes
            .view()
            .tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, H / 1920, H % 1920]>(0)
            .tile::<m![H / 1920], 1, m![Dummy2 = 1 # 2, H / 1920 = 1 # 2, H % 1920]>($half);
        let hi_vrf: VrfTensor<
            f32,
            Chip,
            Cluster2,
            UpGateRows,
            m![Dummy2 = 1, H / 1920 = 1, H % 1920],
        > = ctx
            .sub
            .begin(hi_half_raw)
            .fetch::<m![Dummy2 = 1, H / 1920 = 1, H % 1920 / 8], m![H % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Dummy2 = 1, H / 1920 = 1, H % 1920 / 8], m![H % 8]>()
            .to_vrf();

        let x_half = x_rows
            .tile::<m![H / 1920], 1, m![H / 1920 = 1 # 2, H % 1920]>($half);
        ctx.main
            .begin(x_half)
            .fetch::<m![H % 1920 / 8], m![H % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 1920 / 8], m![H % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 1920 / 4], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), UPGATE_HILO_SCALE)
            .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
            .vector_widen_concat::<m![H % 1920 / 8], m![H % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![H % 8 # 32]>()
            .commit_trim::<m![H % 8]>()
            .commit_view(
                lanes
                    .view_mut()
                    .tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, H / 1920, H % 1920]>(1)
                    .tile::<m![H / 1920], 1, m![Dummy2 = 1 #{!} 2, H / 1920 = 1 #{!} 2, H % 1920]>($half),
            );
        }};
    }

    encode_half!(0);
    encode_half!(1);

    let lanes: DmTensorView<'_, f8e4m3, Chip, Cluster2, UpGateRows, m![Dummy2, H]> =
        unsafe { lanes.view().reshape() };
    ctx.sub
        .begin(lanes)
        .fetch::<m![Dummy2, H / 32], m![H % 32]>()
        .collect::<m![Dummy2, H / 32], m![H % 32]>()
        .to_trf()
}

/// Pools the 30 rows of eight consecutive slices onto one slice (f32 in, f32 out).
fn pool_rows(
    ctx: &mut Device,
    x: DmTensor<f32, Chip, Cluster2, UpGateRows, m![L % 30]>,
) -> DmTensor<f32, Chip, Cluster2, UpGateRowsPaired, m![L % 240]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![L / 2 % 15], m![L % 2 # 8]>()
        .switch::<UpGateRowsPaired, m![L / 2 % 15, L / 30 % 8]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![L / 2 % 15, L / 30 % 8], m![L % 2 # 8]>()
        .commit_trim::<m![L % 2]>()
        .commit()
}

/// Multiplies 240 pooled f32 rows by a global scale and rounds to bf16.
fn scale_pooled(
    ctx: &mut Device,
    x: &DmTensor<f32, Chip, Cluster2, UpGateRowsPaired, m![L % 240]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster2, UpGateRowsPaired, m![L % 240]> {
    let global_scale: DmTensor<f32, Chip, Cluster2, UpGateRowsPaired, m![1 # 8]> = global_scale.to_dm(&mut ctx.tdma);
    let global_scale_vrf: VrfTensor<f32, Chip, Cluster2, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![L / 8 % 30], m![L % 8]>()
        .collect::<m![L / 8 % 30], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 60], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &global_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 30], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit()
}

pub(crate) fn geglu(
    ctx: &mut Device,
    up: DmTensor<f32, Chip, Cluster2, UpGateRows, m![L % 30]>,
    gate: DmTensor<f32, Chip, Cluster2, UpGateRows, m![L % 30]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster2, UpGateRowsPaired, m![L % 240]> {
    let up = pool_rows(ctx, up);
    let up = scale_pooled(ctx, &up, up_global_scale);
    let gate = pool_rows(ctx, gate);
    let gate = scale_pooled(ctx, &gate, gate_global_scale);

    let gelu: DmTensor<f32, Chip, Cluster2, UpGateRowsPaired, m![L % 240]> = ctx
        .sub
        .begin(gate.view())
        .fetch::<m![L / 16 % 15], m![L % 16]>()
        .fetch_cast::<f32>()
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

    let gelu_vrf: VrfTensor<f32, Chip, Cluster2, UpGateRowsPaired, m![L % 240]> = ctx
        .sub
        .begin(gelu.view())
        .fetch::<m![L / 8 % 30], m![L % 8]>()
        .collect::<m![L / 8 % 30], m![L % 8]>()
        .to_vrf();

    ctx.main
        .begin(up.view())
        .fetch::<m![L / 8 % 30], m![L % 8]>()
        .fetch_cast::<f32>()
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
        .commit()
}

const HILO_TARGET: f32 = 256.0;
const SQ_FLOOR: f32 = 1e-30;


/// The down projection on 256 slices of 15 whole rows per cluster with the NVFP4 rows streamed
/// straight into the contraction (f4 -> f8 lookup at fetch) against GeGLU's output held as
/// `hi + lo` f8 lanes, so no decode pass sits between the down_w load and the row sums.
fn down_rows(
    ctx: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster2, UpGateRowsPaired, m![L % 240]>,
    w: DmTensor<f4e2m1, Chip, Cluster2, DownSl, DownW>,
    s: DmTensor<f8e4m3, Chip, Cluster2, DownSl, DownS>,
) -> DmTensor<f32, Chip, Cluster2, DownSl, m![H % 15, 1 # 2]> {
    // --- per-cluster unscale = max|x| / 256, replicated to every slice ---
    let slice_max: DmTensor<f32, Chip, Cluster2, UpGateRowsPaired, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![L % 240 / 8], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 240 / 8], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 240 / 4], m![L % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<L, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let maxima: DmTensor<f32, Chip, Cluster2, m![Dummy256], m![L / 240 % 32, 1 # 8]> = ctx
        .main
        .begin(slice_max.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![Dummy256], m![L / 240 % 32]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![L / 240 % 32], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let unscale: DmTensor<f32, Chip, Cluster2, m![Dummy256], m![1 # 8]> = ctx
        .main
        .begin(maxima.view())
        .fetch::<m![L / 240 % 32], m![1 # 8]>()
        .collect::<m![L / 240 % 32], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<L, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let unscale: DmTensor<f32, Chip, Cluster2, m![Dummy256], m![1 # 8]> = ctx
        .main
        .begin(unscale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, SQ_FLOOR)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div(HILO_TARGET)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let unscale_live: DmTensorView<'_, f32, Chip, Cluster2, UpGateRowsPaired, m![1 # 8]> = unsafe { unscale.view().reshape() };
    let unscale_vrf: VrfTensor<f32, Chip, Cluster2, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(unscale_live)
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    // --- hi/lo f8 lanes on the 32 live slices ---
    let mut lanes: DmTensor<f8e4m3, Chip, Cluster2, UpGateRowsPaired, m![L % 240 / 16, Dummy2, L % 16]> = DmTensor::new();
    ctx.main
        .begin(x.view())
        .fetch::<m![L % 240 / 8], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 240 / 8], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 240 / 4], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &unscale_vrf)
        .vector_widen_concat::<m![L % 240 / 8], m![L % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>()
        .commit_trim::<m![L % 8]>()
        .commit_view(lanes.view_mut().tile::<m![Dummy2], 1, m![L % 240 / 16, Dummy2 = 1 #{!} 2, L % 16]>(0));
    let hi_vrf: VrfTensor<f32, Chip, Cluster2, UpGateRowsPaired, m![L % 240]> = ctx
        .sub
        .begin(lanes.view().tile::<m![Dummy2], 1, m![L % 240 / 16, Dummy2 = 1 # 2, L % 16]>(0))
        .fetch::<m![L % 240 / 8], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 240 / 8], m![L % 8]>()
        .to_vrf();
    ctx.main
        .begin(x.view())
        .fetch::<m![L % 240 / 8], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 240 / 8], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 240 / 4], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &unscale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![L % 240 / 8], m![L % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>()
        .commit_trim::<m![L % 8]>()
        .commit_view(lanes.view_mut().tile::<m![Dummy2], 1, m![L % 240 / 16, Dummy2 = 1 #{!} 2, L % 16]>(1));

    // --- all-gather both lanes to the 256 row slices ---
    let gathered: DmTensor<f8e4m3, Chip, Cluster2, m![Dummy256], m![L % 7680 / 240, L % 240 / 16, Dummy2, L % 16]> = ctx
        .main
        .begin(lanes.view())
        .fetch::<m![1], m![L % 240 / 16, Dummy2, L % 16]>()
        .switch::<m![Dummy256], m![L % 7680 / 240]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![L % 7680 / 240, L % 240 / 16], m![Dummy2, L % 16]>()
        .commit_trim::<m![Dummy2, L % 16]>()
        .commit();
    let gathered: DmTensorView<'_, f8e4m3, Chip, Cluster2, DownSl, m![L % 7680 / 240, L % 240 / 16, Dummy2, L % 16]> =
        unsafe { gathered.view().reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, Cluster2, DownSl, m![1], m![Dummy2, L % 7680]> = ctx
        .sub
        .begin(gathered)
        .fetch::<m![Dummy2, L % 7680 / 32], m![L % 32]>()
        .collect::<m![Dummy2, L % 7680 / 32], m![L % 32]>()
        .to_trf();
    let unscale_rows: DmTensorView<'_, f32, Chip, Cluster2, DownSl, m![1 # 8]> = unsafe { unscale.view().reshape() };
    let row_unscale_vrf: VrfTensor<f32, Chip, Cluster2, DownSl, m![1 # 8]> = ctx
        .sub
        .begin(unscale_rows)
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    // Sum the hi/lo contraction results in the contraction Time reducer,
    // not a Vector pass. This leaves the Vector/Cast resources available to
    // scale preparation on the sub context. Round each block16 partial to BF16 here;
    // stage2 consumes the same BF16 values, stored with half the DM footprint.
    let summed: DmTensor<bf16, Chip, Cluster2, DownSl, m![H % 15, L % 7680 / 64, L % 64 / 16]> = ctx
        .main
        .begin(w.view())
        .fetch::<m![H % 15, L % 7680 / 64], m![L % 64]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 15, L % 7680 / 32], m![L % 32]>()
        .contract_outer::<m![H % 15, L % 7680 / 64, Dummy2], m![L % 64], _, _, _>(&x_trf)
        .contract_packet::<m![L % 64 / 16]>()
        .contract_time::<m![H % 15, L % 7680 / 64]>()
        .contract_lane::<m![H % 15, L % 7680 / 64], m![L % 64 / 16 # 8]>(LaneMode::Sequential)
        .commit_trim::<m![L % 64 / 16]>()
        .commit_cast::<bf16>()
        .commit();
    let partials: DmTensor<bf16, Chip, Cluster2, DownSl, m![H % 15, L % 7680 / 16]> =
        unsafe { summed.reshape() };

    // --- stage 2: partials x block scales (decoded path layout) ---
    let scale_f32: DmTensor<bf16, Chip, Cluster2, DownSl, DownS> = ctx
        .sub
        .begin(s.view())
        .fetch::<m![H % 15, L % 7680 / 16 / 16], m![L / 16 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 15, L % 7680 / 16 / 8], m![L / 16 % 8]>()
        .cast::<bf16, m![L / 16 % 8 # 16]>()
        .commit_trim::<m![L / 16 % 8]>()
        .commit();
    let scale_trf: TrfTensor<bf16, Chip, Cluster2, DownSl, m![1], m![H % 15, L % 7680 / 16]> = ctx
        .sub
        .begin(scale_f32.view())
        .fetch::<m![H % 15, L % 7680 / 16 / 16], m![L / 16 % 16]>()
        .collect::<m![H % 15, L % 7680 / 16 / 16], m![L / 16 % 16]>()
        .to_trf();

    // Keep the same post-contraction unscale and eight-byte scalar cells,
    // but avoid committing/reloading a full flit for each intermediate row.
    let rows: DmTensor<f32, Chip, Cluster2, DownSl, m![H % 15, 1 # 2]> = ctx
        .main
        .begin(partials.view())
        .fetch::<m![H % 15, L % 7680 / 16 / 8], m![L / 16 % 8]>()
        .collect::<m![H % 15, L % 7680 / 16 / 8], m![L / 16 % 8 # 16]>()
        .contract_outer::<m![H % 15, L % 7680 / 16 / 8], m![L / 16 % 8 # 16], _, _, _>(&scale_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 15]>()
        .contract_lane::<m![H % 15], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &row_unscale_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit();

    rows
}


/// The feed-forward block; its output lands directly in the 32-slice `HiddenChunks` layout.
pub(crate) fn feedforward(
    ctx: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster2, Replicated, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &VrfTensor<f32, Chip, Cluster, HiddenChunks, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, HiddenChunks, m![H % 120]> {
    let x_trf = encode_up_gate_replicated(ctx, x);

    let mut up: DmTensor<f32, Chip, Cluster2, UpGateRows, m![L % 30]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, Cluster2, UpGateRows, m![L % 30]> = DmTensor::new();

    let up_w = load_up_gate_chunk!(ctx, up_weight_packed, up_weight_scale);
    let gate_w = load_up_gate_chunk!(ctx, gate_weight_packed, gate_weight_scale);
    let down_w: DmTensor<f4e2m1, Chip, Cluster2, DownSl, DownW> = down_weight_packed.to_dm(&mut ctx.tdma);
    let down_s: DmTensor<f8e4m3, Chip, Cluster2, DownSl, DownS> = down_weight_scale.to_dm(&mut ctx.tdma);

    up_gate_chunk!(ctx, &x_trf, up_w, up);
    up_gate_chunk!(ctx, &x_trf, gate_w, gate);

    let x = geglu(ctx, up, gate, up_global_scale, gate_global_scale);
    let rows = down_rows(ctx, &x, down_w, down_s);
    finish_chunks(ctx, rows, down_global_scale)
}

/// Move d15 rows directly across SRAM with one eight-byte cell per scalar.
/// Compact only in the TU epilogue, where four BF16 rows form an aligned store.
fn finish_chunks(
    ctx: &mut Device,
    rows: DmTensor<f32, Chip, Cluster2, DownSl, m![H % 15, 1 # 2]>,
    down_global_scale: &VrfTensor<f32, Chip, Cluster, HiddenChunks, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, HiddenChunks, m![H % 120]> {
    // Compact d15 row permutation needs four-byte DMA strides (unsupported).
    // Keep all scalar strides at eight bytes across the move, then compact via TU.
    let both: DmTensor<f32, Chip, Cluster, HiddenChunks, m![L / 7680, H % 120, 1 # 2]> = rows.to_dm(&mut ctx.tdma);
    let mine = both.view().tile::<m![L / 7680], 1, m![L / 7680 = 1 # 2, H % 120, 1 # 2]>(0);
    let theirs = both.view().tile::<m![L / 7680], 1, m![L / 7680 = 1 # 2, H % 120, 1 # 2]>(1);
    let theirs_vrf: VrfTensor<f32, Chip, Cluster, HiddenChunks, m![H % 120, 1 # 8]> = ctx
        .sub
        .begin(theirs)
        .fetch::<m![H % 120], m![1 # 2]>()
        .collect::<m![H % 120], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(mine)
        .fetch::<m![H % 120], m![1 # 2]>()
        .collect::<m![H % 120], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &theirs_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), down_global_scale)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 / 4], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit()
}
