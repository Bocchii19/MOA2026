//! Fast Q/K/V projection: both clusters, f8 weights straight into the contraction, activations as
//! an f8 hi/lo pair. Every slice owns 8 (Q) or 4 (K/V) output rows over the whole H = 3840
//! reduction (contiguous 3840 B per row), so no cross-slice reduction is needed. Results are
//! handed to the shared norm/RoPE code through HBM.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, Ps, Qs};
use crate::device::ffn::Hl;

axes![Cx = 2, Sx = 64, Rq = 32, Rk = 16];

type XCl = m![Cx];
type XSl = m![Sx, 1 # 4];

type QCl = m![Qs / 2048];
type QSl = m![Qs / 32 % 64, 1 # 4];
type PCl = m![Ps / 1024];
type PSl = m![Ps / 16 % 64, 1 # 4];

pub(crate) type XTrf = TrfTensor<f8e4m3, Chip, XCl, XSl, m![1], m![Hl, H]>;

/// x hi/lo pair (from HBM, `[Hl, H]` f8) replicated over every slice, staged into the TRF.
pub(crate) fn x_trf(device: &mut Device, pair_hbm: &HbmTensor<f8e4m3, Chip, m![Hl, H]>) -> XTrf {
    let pair: DmTensor<f8e4m3, Chip, XCl, XSl, m![Hl, H]> = pair_hbm.to_dm(&mut device.tdma);
    device
        .sub
        .begin(pair.view())
        .fetch::<m![Hl], m![H]>()
        .collect::<m![Hl, H / 32], m![H % 32]>()
        .to_trf()
}

/// Q rows: y[Qs] = x . Wq^T * row scale, bf16, in HBM.
pub(crate) fn project_q(
    device: &mut Device,
    x: &XTrf,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> HbmTensor<bf16, Chip, m![Qs]> {
    let w: DmTensor<f8e4m3, Chip, QCl, QSl, m![Qs % 32, H]> = weight.to_dm(&mut device.tdma);
    let w: DmTensor<f8e4m3, Chip, XCl, XSl, m![Rq, H]> = unsafe { w.reshape() };
    let s: DmTensor<bf16, Chip, QCl, QSl, m![Qs % 32]> = scale.to_dm(&mut device.tdma);
    let s: DmTensor<bf16, Chip, XCl, XSl, m![Rq]> = unsafe { s.reshape() };

    let s_vrf: VrfTensor<f32, Chip, XCl, XSl, m![Rq]> = device
        .sub
        .begin(s.view())
        .fetch::<m![1], m![Rq]>()
        .fetch_cast::<f32>()
        .collect::<m![Rq / 8], m![Rq % 8]>()
        .to_vrf();

    let y: DmTensor<bf16, Chip, XCl, XSl, m![Rq]> = device
        .main
        .begin(w.view())
        .fetch::<m![Rq], m![H]>()
        .collect::<m![Rq, H / 32], m![H % 32]>()
        .contract_outer::<m![Rq, H / 32, Hl], m![H % 32], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Rq]>()
        .contract_lane::<m![Rq], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &s_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Rq / 4], m![Rq % 4 # 16]>()
        .commit_trim::<m![Rq % 4]>()
        .commit();
    let y: DmTensor<bf16, Chip, QCl, QSl, m![Qs % 32]> = unsafe { y.reshape() };
    y.to_hbm(&mut device.tdma)
}

/// K or V rows: y[Ps] = x . W^T * row scale, bf16, in HBM.
pub(crate) fn project_p(
    device: &mut Device,
    x: &XTrf,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> HbmTensor<bf16, Chip, m![Ps]> {
    let w: DmTensor<f8e4m3, Chip, PCl, PSl, m![Ps % 16, H]> = weight.to_dm(&mut device.tdma);
    let w: DmTensor<f8e4m3, Chip, XCl, XSl, m![Rk, H]> = unsafe { w.reshape() };
    let s: DmTensor<bf16, Chip, PCl, PSl, m![Ps % 16]> = scale.to_dm(&mut device.tdma);
    let s: DmTensor<bf16, Chip, XCl, XSl, m![Rk]> = unsafe { s.reshape() };

    let s_vrf: VrfTensor<f32, Chip, XCl, XSl, m![Rk]> = device
        .sub
        .begin(s.view())
        .fetch::<m![1], m![Rk]>()
        .fetch_cast::<f32>()
        .collect::<m![Rk / 8], m![Rk % 8]>()
        .to_vrf();

    let y: DmTensor<bf16, Chip, XCl, XSl, m![Rk]> = device
        .main
        .begin(w.view())
        .fetch::<m![Rk], m![H]>()
        .collect::<m![Rk, H / 32], m![H % 32]>()
        .contract_outer::<m![Rk, H / 32, Hl], m![H % 32], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Rk]>()
        .contract_lane::<m![Rk], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &s_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Rk / 4], m![Rk % 4 # 16]>()
        .commit_trim::<m![Rk % 4]>()
        .commit();
    let y: DmTensor<bf16, Chip, PCl, PSl, m![Ps % 16]> = unsafe { y.reshape() };
    y.to_hbm(&mut device.tdma)
}
