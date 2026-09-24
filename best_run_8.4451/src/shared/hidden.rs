
use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy256, Dummy8, H};
use crate::device::layout::{Cluster, Replicated, Slice};
use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

axes![Dummy32 = 32, Ring256 = 256];

// The hidden vector split into 120-element chunks, one chunk per slice on 32 consecutive slices.
// Every axis that later moves through the switch is fully live, so no padding bookkeeping.
pub(crate) type HiddenChunks = m![1 # 8, H / 120];
// The same 32 chunks on slices 0, 8, 16, ...: the layout a ring-of-8 inter-slice reduce leaves
// a row-split GEMV output in.
pub(crate) type LiveRows = m![H / 120, 1 # 8];
// Every slice of ring g holds chunk g: what a ring-of-8 reduce leaves when its output slot is a
// broadcast axis, and the layout the post-projection tails run in (8 redundant copies, no padding).
pub(crate) type RowsReplicated = m![H / 120, Dummy8];
// Raw all-gather result: every slice holds every 120-element chunk in a 128-element cell (the
// last flit of each chunk is half padding), compacted to `m![H]` before use.
type HiddenReplicated = m![H / 120, H % 120 # 128];

/// RMS over the whole hidden vector from per-slice partial sums reduced across the 32-slice ring,
/// so no single slice ever has to hold the full vector.
fn root_mean_square<C: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, C, HiddenChunks, m![H % 120]>,
) -> VrfTensor<f32, Chip, C, HiddenChunks, m![1 # 8]> {
    let partial: DmTensor<f32, Chip, C, HiddenChunks, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120 / 4], m![H % 120 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let mean_square: DmTensor<f32, Chip, C, m![1 # 8, Dummy32], m![1 # 8]> = ctx
        .main
        .begin(partial.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 8, Dummy32], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, C, m![1 # 8, Dummy32], m![1 # 8]> = ctx
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
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, C, HiddenChunks, m![1 # 8]> = unsafe { rms.reshape() };

    ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// RMSNorm of an HBM hidden vector, returned replicated on every slice.
///
/// The vector is spread over 32 slices, normalized chunk-wise, then all-gathered with one ring
/// pass in which every slice injects only 8 flits (~2k cycles), instead of the ~61k cycles a
/// single-source broadcast of 240 flits costs.
///
/// Generic over the cluster mapping `C`: with `Cluster` (`m![1 # 2]`) only cluster 0 is live,
/// which is what the single-cluster kernels want; with a fully live 2-element mapping such as
/// `m![Dummy2]` the HBM loads broadcast to both clusters and each cluster runs its own reduce
/// and its own ring, so both end up holding the vector at no extra cost.
pub(crate) fn normalize_replicated<C: M>(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, C, Replicated, m![H]> {
    let x: DmTensor<bf16, Chip, C, HiddenChunks, m![H % 120]> = x.to_dm(&mut ctx.tdma);
    let weight: DmTensor<bf16, Chip, C, HiddenChunks, m![H % 120]> = rms_weight.to_dm(&mut ctx.tdma);

    let rms_vrf = root_mean_square(ctx, &x);
    let weight_vrf: VrfTensor<f32, Chip, C, HiddenChunks, m![H % 120]> = ctx
        .sub
        .begin(weight.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 / 8], m![H % 120 % 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, C, HiddenChunks, m![H % 120]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120 / 4], m![H % 120 % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 120 % 8 # 16]>()
        .commit_trim::<m![H % 120 % 8]>()
        .commit();

    replicate(ctx, &normalized)
}

/// Spreads an HBM hidden vector so that every slice of ring g holds chunk g: a 32-slice DMA plus
/// one ring-of-8 pass (~120 cycles), instead of 256 DMA copies.
pub(crate) fn spread_rows(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, RowsReplicated, m![H % 120]> {
    let rows: DmTensor<bf16, Chip, Cluster, LiveRows, m![H % 120]> = x.to_dm(&mut ctx.tdma);
    ctx.main
        .begin(rows.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .switch::<RowsReplicated, m![H % 120 / 8]>(SwitchConfig::CustomBroadcast { ring_size: 8 })
        .collect::<m![H % 120 / 8], m![H % 120 % 8 # 16]>()
        .commit_trim::<m![H % 120 % 8]>()
        .commit()
}

pub(crate) fn rows_vrf(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, RowsReplicated, m![H % 120]>,
) -> VrfTensor<f32, Chip, Cluster, RowsReplicated, m![H % 120]> {
    ctx.sub
        .begin(x.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 / 8], m![H % 120 % 8]>()
        .to_vrf()
}

/// RMS of a hidden vector on `RowsReplicated`, replicated to every slice. Every slice holds a
/// valid chunk (8 copies per chunk), so the ring reduce over all 256 slices sees 8x the sum.
fn root_mean_square_rows(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, RowsReplicated, m![H % 120]>,
) -> VrfTensor<f32, Chip, Cluster, RowsReplicated, m![1 # 8]> {
    const COPIES: f32 = 8.0;

    let partial: DmTensor<f32, Chip, Cluster, RowsReplicated, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120 / 4], m![H % 120 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32 * COPIES)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let partial: DmTensor<f32, Chip, Cluster, m![Ring256], m![1 # 8]> = unsafe { partial.reshape() };

    let mean_square: DmTensor<f32, Chip, Cluster, m![Dummy256], m![1 # 8]> = ctx
        .main
        .begin(partial.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Dummy256], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, m![Dummy256], m![1 # 8]> = ctx
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
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, RowsReplicated, m![1 # 8]> = unsafe { rms.reshape() };

    ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// RMSNorm of a hidden vector spread over `RowsReplicated`, staying in that layout.
pub(crate) fn normalize_rows(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, RowsReplicated, m![H % 120]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, RowsReplicated, m![H % 120]> {
    let rms_vrf = root_mean_square_rows(ctx, x);
    let weight = spread_rows(ctx, rms_weight);
    let weight_vrf = rows_vrf(ctx, &weight);

    ctx.main
        .begin(x.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120 / 4], m![H % 120 % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 120 % 8 # 16]>()
        .commit_trim::<m![H % 120 % 8]>()
        .commit()
}

/// `x + residual` on `RowsReplicated`, with the residual streamed straight from HBM.
pub(crate) fn add_residual_rows(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, RowsReplicated, m![H % 120]>,
    residual: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, RowsReplicated, m![H % 120]> {
    let residual = spread_rows(ctx, residual);
    let residual_vrf = rows_vrf(ctx, &residual);

    ctx.main
        .begin(x.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 120 % 8 # 16]>()
        .commit_trim::<m![H % 120 % 8]>()
        .commit()
}

/// Multiplies every element on `RowsReplicated` by one scalar held in an 8-wide VRF cell.
pub(crate) fn scale_rows(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, RowsReplicated, m![H % 120]>,
    scalar: &VrfTensor<f32, Chip, Cluster, RowsReplicated, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, RowsReplicated, m![H % 120]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120 / 4], m![H % 120 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scalar)
        .vector_widen_concat::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 120 % 8 # 16]>()
        .commit_trim::<m![H % 120 % 8]>()
        .commit()
}

/// Writes a `RowsReplicated` hidden vector back to HBM (one copy per row group).
pub(crate) fn store_rows(
    ctx: &mut Context,
    x: DmTensor<bf16, Chip, Cluster, RowsReplicated, m![H % 120]>,
    out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, LiveRows, m![H % 120]> = unsafe { x.reshape() };
    x.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}

/// One f32 scalar from HBM, replicated into every slice's VRF (one flit around the ring).
pub(crate) fn broadcast_scalar_f32(
    ctx: &mut Context,
    scalar: &HbmTensor<f32, Chip, m![1]>,
) -> VrfTensor<f32, Chip, Cluster, RowsReplicated, m![1 # 8]> {
    let scalar: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = scalar.to_dm(&mut ctx.tdma);
    let all: DmTensor<f32, Chip, Cluster, Replicated, m![1 # 8]> = ctx
        .main
        .begin(scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<Replicated, m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let all: DmTensor<f32, Chip, Cluster, RowsReplicated, m![1 # 8]> = unsafe { all.reshape() };
    ctx.sub
        .begin(all.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// One bf16 scalar (stored 8-wide in HBM), replicated into every slice's VRF as f32.
pub(crate) fn broadcast_scalar_bf16(
    ctx: &mut Context,
    scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> VrfTensor<f32, Chip, Cluster, RowsReplicated, m![1 # 8]> {
    let scalar: DmTensor<bf16, Chip, Cluster, Slice, m![1 # 8]> = scalar.to_dm(&mut ctx.tdma);
    // Widen before the switch so the collected packet is a full 32-byte flit.
    let all: DmTensor<f32, Chip, Cluster, Replicated, m![1 # 8]> = ctx
        .main
        .begin(scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .switch::<Replicated, m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let all: DmTensor<f32, Chip, Cluster, RowsReplicated, m![1 # 8]> = unsafe { all.reshape() };
    ctx.sub
        .begin(all.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// All-gather: every slice ends up with every chunk, in chunk order, as a compact `m![H]`.
pub(crate) fn replicate<C: M>(
    ctx: &mut Context,
    chunks: &DmTensor<bf16, Chip, C, HiddenChunks, m![H % 120]>,
) -> DmTensor<bf16, Chip, C, Replicated, m![H]> {
    let gathered: DmTensor<bf16, Chip, C, Replicated, HiddenReplicated> = ctx
        .main
        .begin(chunks.view())
        .fetch::<m![1], m![H % 120]>()
        .switch::<Replicated, m![H / 120]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 120, H % 120 # 128 / 16], m![H % 120 # 128 % 16]>()
        .commit_trim::<m![H % 120 # 128 % 16]>()
        .commit();

    // Drop the per-chunk padding: 8-element packets read only live cells, the trim writes them
    // back contiguously.
    let compact: DmTensor<bf16, Chip, C, Replicated, m![H / 120, H % 120 / 8, H % 120 % 8]> = ctx
        .main
        .begin(gathered.view())
        .fetch::<m![H / 120, H % 120 / 8], m![H % 120 % 8]>()
        .collect::<m![H / 120, H % 120 / 8], m![H % 120 % 8 # 16]>()
        .commit_trim::<m![H % 120 % 8]>()
        .commit();
    unsafe { compact.reshape() }
}

// --- the feed-forward tail on the 32-slice HiddenChunks layout -------------------------------

fn chunk_rows_vrf(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, HiddenChunks, m![H % 120]>,
) -> VrfTensor<f32, Chip, Cluster, HiddenChunks, m![H % 120]> {
    ctx.sub
        .begin(x.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 / 8], m![H % 120 % 8]>()
        .to_vrf()
}

pub(crate) fn normalize_chunk_rows(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, HiddenChunks, m![H % 120]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, HiddenChunks, m![H % 120]> {
    let rms_vrf = root_mean_square(ctx, x);
    let weight: DmTensor<bf16, Chip, Cluster, HiddenChunks, m![H % 120]> =
        rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf = chunk_rows_vrf(ctx, &weight);

    ctx.main
        .begin(x.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120 / 4], m![H % 120 % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 120 % 8 # 16]>()
        .commit_trim::<m![H % 120 % 8]>()
        .commit()
}

pub(crate) fn broadcast_scalar_f32_chunks(
    ctx: &mut Context,
    scalar: &HbmTensor<f32, Chip, m![1]>,
) -> VrfTensor<f32, Chip, Cluster, HiddenChunks, m![1 # 8]> {
    let all: DmTensor<f32, Chip, Cluster, HiddenChunks, m![1 # 8]> =
        scalar.to_dm(&mut ctx.tdma);
    ctx.sub
        .begin(all.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}


pub(crate) fn store_chunk_rows(
    ctx: &mut Context,
    x: DmTensor<bf16, Chip, Cluster, HiddenChunks, m![H % 120]>,
    out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    x.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}

pub(crate) fn add_residual_chunks(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, HiddenChunks, m![H % 120]>,
    residual: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, HiddenChunks, m![H % 120]> {
    let residual: DmTensor<bf16, Chip, Cluster, HiddenChunks, m![H % 120]> = residual.to_dm(&mut ctx.tdma);
    let residual_vrf = chunk_rows_vrf(ctx, &residual);

    ctx.main
        .begin(x.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 120 % 8 # 16]>()
        .commit_trim::<m![H % 120 % 8]>()
        .commit()
}

pub(crate) fn scale_chunks(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, HiddenChunks, m![H % 120]>,
    scalar: &VrfTensor<f32, Chip, Cluster, HiddenChunks, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, HiddenChunks, m![H % 120]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![H % 120 / 8], m![H % 120 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120 / 4], m![H % 120 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scalar)
        .vector_widen_concat::<m![H % 120 / 8], m![H % 120 % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 120 % 8 # 16]>()
        .commit_trim::<m![H % 120 % 8]>()
        .commit()
}

/// The layer scalar (one bf16 in an 8-wide HBM cell) widened straight into every chunk slice's VRF
/// on the sub context, so no main-context pass sits between the down projection's passes.
pub(crate) fn scalar_bf16_vrf(
    ctx: &mut Context,
    scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> VrfTensor<f32, Chip, Cluster, HiddenChunks, m![1 # 8]> {
    let scalar: DmTensor<bf16, Chip, Cluster, HiddenChunks, m![1 # 8]> = scalar.to_dm(&mut ctx.tdma);
    ctx.sub
        .begin(scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}
