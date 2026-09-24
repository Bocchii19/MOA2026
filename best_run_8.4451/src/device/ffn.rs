//! Fast NVFP4 feed-forward: both clusters, f4 -> f8 lookup fed straight into the contraction,
//! activations carried as an FP8 hi/lo pair, block scales applied on per-16 partial sums.
//!
//! up/gate: each cluster owns half of L (7680 rows): 256 slices x 30 rows, whole H per slice.
//! down:    each cluster owns the matching half of L as reduction axis: slice = (128 row groups of
//!          30 rows) x (2 chunks of 3840 columns); chunks are summed by the inter-slice reducer and
//!          the two clusters' partials meet in HBM.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, L};
use crate::device::layout::{Cluster, Slice};

axes![Hl = 2, Ac = 3840];

pub(crate) type UCl = m![L / 7680];
pub(crate) type USl = m![L / 30 % 256];

/// Rows per contraction pass (bounded by the 8 KB VRF: rows x 240 blocks x 4 B).
const ROWS: usize = 2; // down projection still uses 2-row passes
const PASSES: usize = 30 / ROWS;
const ACT_SCALE: f32 = 16.0;
const INVSQRT2: f32 = 0.70710678118f32;
/// activation (raw NVFP4 domain) encode scale for the down projection
const ACT_ENC: f32 = 0.5;

/// y[L] (f32, f4 weights x block scales x hi/lo activations, before any global scale) for the 30 rows
/// this slice owns, one padded flit per row (`[L % 30, 1 # 8]`).
pub(crate) fn project_up_or_gate(
    device: &mut Device,
    x_trf: &TrfTensor<f8e4m3, Chip, UCl, USl, m![1], m![Hl, H]>,
    weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> DmTensor<f32, Chip, UCl, USl, m![L % 30, 1 # 8]> {
    let weight: DmTensor<f4e2m1, Chip, UCl, USl, m![L % 30, H]> = weight.to_dm(&mut device.tdma);
    let scale: DmTensor<f8e4m3, Chip, UCl, USl, m![L % 30, H / 16]> = scale.to_dm(&mut device.tdma);

    let mut out: DmTensor<f32, Chip, UCl, USl, m![L % 30, 1 # 8]> = DmTensor::new();
    for i in 0..5 {
        let scale_vrf: VrfTensor<f32, Chip, UCl, USl, m![L % 30 = 6, H / 16]> = device
            .sub
            .begin(scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16]>(6 * i))
            .fetch::<m![L % 30 = 6], m![H / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 30 = 6, H / 16 / 8], m![H / 16 % 8]>()
            .to_vrf();

        device
            .main
            .begin(weight.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H]>(6 * i))
            .fetch::<m![L % 30 = 6], m![H]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![L % 30 = 6, H / 32], m![H % 32]>()
            .contract_outer::<m![L % 30 = 6, H / 64, Hl], m![H % 64], _, _, _>(x_trf)
            .contract_packet::<m![H / 16 % 4]>()
            .contract_time::<m![L % 30 = 6, H / 64]>()
            .contract_lane::<m![L % 30 = 6, H / 64, H / 16 % 4], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_intra_slice_reduce::<H, m![L % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(out.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30, 1 # 8]>(6 * i));
    }
    out
}

/// x (bf16 [H]) -> f8 hi/lo pair `[Hl, H]` (x scaled by 2^6 first; undo it in the epilogue).
/// Processed in 2 chunks of 1920 so the f32 copy of `hi` fits the 8 KB VRF.
pub(crate) fn encode_hi_lo<Cl: M, Sl: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cl, Sl, m![H]>,
) -> DmTensor<f8e4m3, Chip, Cl, Sl, m![Hl, H]> {
    let mut hi: DmTensor<f8e4m3, Chip, Cl, Sl, m![H]> = DmTensor::new();
    let mut lo: DmTensor<f8e4m3, Chip, Cl, Sl, m![H]> = DmTensor::new();
    for c in 0..2 {
        let xc = x.view().tile::<m![H], 1920, m![H = 1920 # 3840]>(1920 * c);
        let hi_c: DmTensor<f8e4m3, Chip, Cl, Sl, m![H = 1920]> = device
            .main
            .begin(xc)
            .fetch::<m![1], m![H = 1920]>()
            .fetch_cast::<f32>()
            .collect::<m![H = 1920 / 8], m![H = 1920 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H = 1920 / 4], m![H = 1920 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ACT_SCALE)
            .vector_widen_concat::<m![H = 1920 / 8], m![H = 1920 % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![H = 1920 % 8 # 32]>()
            .commit_trim::<m![H = 1920 % 8]>()
            .commit();

        let hi_vrf: VrfTensor<f32, Chip, Cl, Sl, m![H = 1920]> = device
            .sub
            .begin(hi_c.view())
            .fetch::<m![H = 1920 / 8], m![H = 1920 % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![H = 1920 / 8], m![H = 1920 % 8]>()
            .to_vrf();

        let xc = x.view().tile::<m![H], 1920, m![H = 1920 # 3840]>(1920 * c);
        device
            .main
            .begin(xc)
            .fetch::<m![1], m![H = 1920]>()
            .fetch_cast::<f32>()
            .collect::<m![H = 1920 / 8], m![H = 1920 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H = 1920 / 4], m![H = 1920 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ACT_SCALE)
            .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
            .vector_widen_concat::<m![H = 1920 / 8], m![H = 1920 % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![H = 1920 % 8 # 32]>()
            .commit_trim::<m![H = 1920 % 8]>()
            .commit_view(lo.view_mut().tile::<m![H], 1920, m![H = 1920 #{!} 3840]>(1920 * c));

        hi_c.view()
            .to_dm_view(&mut device.tdma, hi.view_mut().tile::<m![H], 1920, m![H = 1920 #{!} 3840]>(1920 * c));
    }

    let mut pair: DmTensor<f8e4m3, Chip, Cl, Sl, m![Hl, H]> = DmTensor::new();
    hi.view().to_dm_view(&mut device.tdma, pair.view_mut().tile::<m![Hl], 1, m![Hl = 1 #{!} 2, H]>(0));
    lo.view().to_dm_view(&mut device.tdma, pair.view_mut().tile::<m![Hl], 1, m![Hl = 1 #{!} 2, H]>(1));
    pair
}

type Act = DmTensor<f32, Chip, UCl, USl, m![L % 30, 1 # 8]>;
type Scalar1 = VrfTensor<f32, Chip, UCl, USl, m![1 # 8]>;

/// scalar (HBM f32[1]) * k -> VRF scalar on every slice.
fn scalar_vrf(device: &mut Device, g: &HbmTensor<f32, Chip, m![1]>, k: f32) -> Scalar1 {
    let g: DmTensor<f32, Chip, UCl, USl, m![1 # 8]> = g.to_dm(&mut device.tdma);
    let gk: DmTensor<f32, Chip, UCl, USl, m![1 # 8]> = device
        .main
        .begin(g.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), k)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    device
        .sub
        .begin(gk.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

fn act_to_vrf(device: &mut Device, a: &Act) -> VrfTensor<f32, Chip, UCl, USl, m![L % 30, 1 # 8]> {
    device
        .sub
        .begin(a.view())
        .fetch::<m![L % 30], m![1 # 8]>()
        .collect::<m![L % 30], m![1 # 8]>()
        .to_vrf()
}

/// a * k (constant)
fn mul_const(device: &mut Device, a: &Act, k: f32) -> Act {
    device
        .main
        .begin(a.view())
        .fetch::<m![L % 30], m![1 # 8]>()
        .collect::<m![L % 30], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), k)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// t = gate * c2 (c2 = gate_global / (ACT_SCALE * sqrt 2)); returns t * (1 + erf t).
fn gelu_core(device: &mut Device, gate: &Act, c2: &Scalar1) -> Act {
    device
        .main
        .begin(gate.view())
        .fetch::<m![L % 30], m![1 # 8]>()
        .collect::<m![L % 30], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), c2)
        .vector_stash()
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// a * b elementwise (b via VRF); the result is compacted to `[L % 30]` (pairs of f32).
fn mul_vec(device: &mut Device, a: &Act, b: &VrfTensor<f32, Chip, UCl, USl, m![L % 30, 1 # 8]>) -> DmTensor<f32, Chip, UCl, USl, m![L % 30]> {
    device
        .main
        .begin(a.view())
        .fetch::<m![L % 30], m![1 # 8]>()
        .collect::<m![L % 30], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), b)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![L % 30 / 2], m![L % 2 # 8]>()
        .commit_trim::<m![L % 2]>()
        .commit()
}

/// act[L] = up * gelu(gate), f32 compact `[L % 30]`, in the up/gate layout.
pub(crate) fn geglu(
    device: &mut Device,
    up_raw: &Act,
    gate_raw: &Act,
    gate_global: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<f32, Chip, UCl, USl, m![L % 30]> {
    // The up global scale is applied after the down projection (linear), so the activation stays
    // in the "raw" NVFP4 domain, where f8 hi/lo covers it with 2.8x headroom (see campaign/simffn.py).
    let c2 = scalar_vrf(device, gate_global, INVSQRT2 / ACT_SCALE);
    let up_s = mul_const(device, up_raw, ACT_ENC * INVSQRT2 / ACT_SCALE);
    let x = gelu_core(device, gate_raw, &c2);
    let up_vrf = act_to_vrf(device, &up_s);
    mul_vec(device, &x, &up_vrf)
}

// ---------------------------------------------------------------- down projection

pub(crate) type DCl = m![L / 7680];
pub(crate) type DSl = m![H / 30 % 128, L / 3840 % 2];

/// f32 activation chunk [L % 3840] -> f8 hi/lo pair (chunks of 1920 to fit the VRF).
fn encode_hi_lo_f32(
    device: &mut Device,
    x: &DmTensor<f32, Chip, DCl, DSl, m![L % 3840]>,
) -> DmTensor<f8e4m3, Chip, DCl, DSl, m![Hl, L % 3840]> {
    let mut hi: DmTensor<f8e4m3, Chip, DCl, DSl, m![Ac]> = DmTensor::new();
    let mut lo: DmTensor<f8e4m3, Chip, DCl, DSl, m![Ac]> = DmTensor::new();
    for c in 0..2 {
        let xc = unsafe { x.view().reshape::<Chip, DCl, DSl, m![Ac]>() }.tile::<m![Ac], 1920, m![Ac = 1920 # 3840]>(1920 * c);
        let hi_c: DmTensor<f8e4m3, Chip, DCl, DSl, m![Ac = 1920]> = device
            .main
            .begin(xc)
            .fetch::<m![1], m![Ac = 1920]>()
            .collect::<m![Ac = 1920 / 8], m![Ac = 1920 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Ac = 1920 / 4], m![Ac = 1920 % 4]>()
            .vector_widen_concat::<m![Ac = 1920 / 8], m![Ac = 1920 % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![Ac = 1920 % 8 # 32]>()
            .commit_trim::<m![Ac = 1920 % 8]>()
            .commit();

        let hi_vrf: VrfTensor<f32, Chip, DCl, DSl, m![Ac = 1920]> = device
            .sub
            .begin(hi_c.view())
            .fetch::<m![Ac = 1920 / 8], m![Ac = 1920 % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Ac = 1920 / 8], m![Ac = 1920 % 8]>()
            .to_vrf();

        let xc = unsafe { x.view().reshape::<Chip, DCl, DSl, m![Ac]>() }.tile::<m![Ac], 1920, m![Ac = 1920 # 3840]>(1920 * c);
        device
            .main
            .begin(xc)
            .fetch::<m![1], m![Ac = 1920]>()
            .collect::<m![Ac = 1920 / 8], m![Ac = 1920 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Ac = 1920 / 4], m![Ac = 1920 % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
            .vector_widen_concat::<m![Ac = 1920 / 8], m![Ac = 1920 % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![Ac = 1920 % 8 # 32]>()
            .commit_trim::<m![Ac = 1920 % 8]>()
            .commit_view(lo.view_mut().tile::<m![Ac], 1920, m![Ac = 1920 #{!} 3840]>(1920 * c));

        hi_c.view()
            .to_dm_view(&mut device.tdma, hi.view_mut().tile::<m![Ac], 1920, m![Ac = 1920 #{!} 3840]>(1920 * c));
    }
    let mut pair: DmTensor<f8e4m3, Chip, DCl, DSl, m![Hl, Ac]> = DmTensor::new();
    hi.view().to_dm_view(&mut device.tdma, pair.view_mut().tile::<m![Hl], 1, m![Hl = 1 #{!} 2, Ac]>(0));
    lo.view().to_dm_view(&mut device.tdma, pair.view_mut().tile::<m![Hl], 1, m![Hl = 1 #{!} 2, Ac]>(1));
    unsafe { pair.reshape() }
}

/// partial[c][H] (f32) = sum over cluster c's L half of down_w[H, L] * act[L] (blocks scaled), NOT yet
/// multiplied by the global scale / undone ACT_SCALE. Result lives in HBM as `[L / 7680, H]`.
pub(crate) fn project_down(
    device: &mut Device,
    act: &DmTensor<f32, Chip, UCl, USl, m![L % 30]>,
    weight: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> HbmTensor<f32, Chip, m![L / 3840, H]> {
    // activation exchange through HBM: every (row group, chunk) slice needs its chunk of L
    let act_hbm: HbmTensor<f32, Chip, m![L]> = act.to_hbm(&mut device.tdma);
    let act_dm: DmTensor<f32, Chip, DCl, DSl, m![L % 3840]> = act_hbm.to_dm(&mut device.tdma);
    let pair = encode_hi_lo_f32(device, &act_dm);
    let x_trf: TrfTensor<f8e4m3, Chip, DCl, DSl, m![1], m![Hl, L % 3840]> = device
        .sub
        .begin(pair.view())
        .fetch::<m![Hl], m![L % 3840]>()
        .collect::<m![Hl, L % 3840 / 32], m![L % 32]>()
        .to_trf();

    let weight: DmTensor<f4e2m1, Chip, DCl, DSl, m![H % 30, L % 3840]> = weight.to_dm(&mut device.tdma);
    let scale: DmTensor<f8e4m3, Chip, DCl, DSl, m![H % 30, L / 16 % 240]> = scale.to_dm(&mut device.tdma);

    let mut part_padded: DmTensor<f32, Chip, DCl, DSl, m![H % 30, 1 # 8]> = DmTensor::new();
    for i in 0..5 {
        let scale_vrf: VrfTensor<f32, Chip, DCl, DSl, m![H % 30 = 6, L / 16 % 240]> = device
            .sub
            .begin(scale.view().tile::<m![H % 30], 6, m![H % 30 = 6 # 30, L / 16 % 240]>(6 * i))
            .fetch::<m![H % 30 = 6], m![L / 16 % 240]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 30 = 6, L / 16 % 240 / 8], m![L / 16 % 8]>()
            .to_vrf();

        device
            .main
            .begin(weight.view().tile::<m![H % 30], 6, m![H % 30 = 6 # 30, L % 3840]>(6 * i))
            .fetch::<m![H % 30 = 6], m![L % 3840]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![H % 30 = 6, L % 3840 / 32], m![L % 32]>()
            .contract_outer::<m![H % 30 = 6, L % 3840 / 64, Hl], m![L % 64], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 4]>()
            .contract_time::<m![H % 30 = 6, L % 3840 / 64]>()
            .contract_lane::<m![H % 30 = 6, L % 3840 / 64, L / 16 % 4], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(part_padded.view_mut().tile::<m![H % 30], 6, m![H % 30 = 6 #{!} 30, 1 # 8]>(6 * i));
    }
    let part: DmTensor<f32, Chip, DCl, DSl, m![H % 30]> = device
        .main
        .begin(part_padded.view())
        .fetch::<m![H % 30], m![1 # 8]>()
        .collect::<m![H % 30], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![H % 30 / 2], m![H % 2 # 8]>()
        .commit_trim::<m![H % 2]>()
        .commit();

    // the four (cluster, chunk) partials meet in HBM; finish_down sums them
    part.to_hbm(&mut device.tdma)
}

type HiddenRows = m![H / 120, 1 # 8];

/// out[H] (bf16) = (partial[0] + partial[1]) * down_global / ACT_SCALE, on cluster 0 in the layout the
/// shared norm expects.
pub(crate) fn finish_down(
    device: &mut Device,
    part: &HbmTensor<f32, Chip, m![L / 3840, H]>,
    up_global: &HbmTensor<f32, Chip, m![1]>,
    down_global: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let both: DmTensor<f32, Chip, Cluster, HiddenRows, m![L / 3840, H % 120]> = part.to_dm(&mut device.tdma);
    let ug: DmTensor<f32, Chip, Cluster, HiddenRows, m![1 # 8]> = up_global.to_dm(&mut device.tdma);
    let ug_vrf: VrfTensor<f32, Chip, Cluster, HiddenRows, m![1 # 8]> = device
        .sub
        .begin(ug.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let dg: DmTensor<f32, Chip, Cluster, HiddenRows, m![1 # 8]> = down_global.to_dm(&mut device.tdma);
    let dg_vrf: VrfTensor<f32, Chip, Cluster, HiddenRows, m![1 # 8]> = device
        .sub
        .begin(dg.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let out: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> = device
        .main
        .begin(both.view())
        .fetch::<m![H % 120 / 8, L / 3840], m![H % 8]>()
        .collect::<m![H % 120 / 8, L / 3840], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120 / 8, L / 3840, H / 4 % 2], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &ug_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &dg_vrf)
        .vector_intra_slice_reduce::<L, m![H % 120 / 4], m![H % 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(ACT_ENC)
        .vector_widen_concat::<m![H % 120 / 8], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    out.to_dm(&mut device.tdma)
}

/// x_norm (cluster 0 / single slice) -> f8 hi/lo pair replicated as TRF over both clusters.
pub(crate) fn replicate_pair_trf(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
) -> TrfTensor<f8e4m3, Chip, UCl, USl, m![1], m![Hl, H]> {
    let pair = encode_hi_lo(device, x);
    let pair_hbm: HbmTensor<f8e4m3, Chip, m![Hl, H]> = pair.to_hbm(&mut device.tdma);
    let pair_dm: DmTensor<f8e4m3, Chip, UCl, USl, m![Hl, H]> = pair_hbm.to_dm(&mut device.tdma);
    device
        .sub
        .begin(pair_dm.view())
        .fetch::<m![Hl], m![H]>()
        .collect::<m![Hl, H / 32], m![H % 32]>()
        .to_trf()
}

/// Whole feed-forward core (up, gate, GeGLU, down) -> bf16 [H] on cluster 0 for the shared tail.
pub(crate) fn feedforward(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    up_w: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_w: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_w: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_s: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_s: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_s: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_g: &HbmTensor<f32, Chip, m![1]>,
    gate_g: &HbmTensor<f32, Chip, m![1]>,
    down_g: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let x_trf = replicate_pair_trf(device, x);
    let up = project_up_or_gate(device, &x_trf, up_w, up_s);
    let gate = project_up_or_gate(device, &x_trf, gate_w, gate_s);
    let act = geglu(device, &up, &gate, gate_g);
    let part = project_down(device, &act, down_w, down_s);
    finish_down(device, &part, up_g, down_g)
}
