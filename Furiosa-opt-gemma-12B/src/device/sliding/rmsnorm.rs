
use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, Gs, Ns};
use crate::device::layout::{Cluster, Slice};
use crate::{Chip, EPS};

const DS_F32: f32 = Ds::SIZE as f32;

fn normalize_query_serial<Cluster: M, Slice: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> {
    let mean_square: DmTensor<f32, Chip, Cluster, Slice, m![Ns, Gs]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ns, Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Gs, Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![Ns, Gs], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![Ns], m![Gs % 2 # 8]>()
        .commit_trim::<m![Gs % 2]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, Slice, m![Ns, Gs]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
        .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns / 2], m![Ns % 2, Gs]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_concat::<m![Ns / 4], m![Ns % 4, Gs]>()
        .vector_final()
        .commit_trim::<m![Ns % 4, Gs]>()
        .commit();

    let weight_vrf = load_norm_weight::<Cluster, Slice>(device, rms_weight);

    let rms_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Ns, Gs]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
        .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
        .to_vrf();

    device.main
        .begin(x.view())
        .fetch::<m![Ns, Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn load_norm_weight<Cluster: M, Slice: M>(
    device: &mut Device,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![Ds]> {
    let weight_dm: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = rms_weight.to_dm(&mut device.tdma);

    device.sub
        .begin(weight_dm.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

fn root_mean_square<Cluster: M, Slice: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![Ns]> {
    let mean_square: DmTensor<f32, Chip, Cluster, Slice, m![Ns]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ns, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![Ns], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![Ns / 2], m![Ns % 2 # 8]>()
        .commit_trim::<m![Ns % 2]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, Slice, m![Ns]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![Ns]>()
        .collect::<m![1], m![Ns]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns / 4], m![Ns % 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_concat::<m![1], m![Ns]>()
        .vector_final()
        .commit_trim::<m![Ns]>()
        .commit();

    device.sub
        .begin(rms.view())
        .fetch::<m![1], m![Ns]>()
        .collect::<m![1], m![Ns]>()
        .to_vrf()
}

fn scale_by_rms_and_weight<Cluster: M, Slice: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    rms_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Ns]>,
    weight_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    device.main
        .begin(x.view())
        .fetch::<m![Ns, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
        .vector_widen_concat::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn scale_by_rms<Cluster: M, Slice: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    rms_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Ns]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    device.main
        .begin(x.view())
        .fetch::<m![Ns, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Ds / 4], m![Ds % 4]>()
        .vector_fp_div(rms_vrf)
        .vector_widen_concat::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

// Per-head normalization runs on the single live slice that already holds the projection
// output. The head-per-slice variant (`normalize_head*` below) spent 6 DMA relayouts
// (~9.5k cycles on the serial DMA queue, QKV schedule 23:37) to save ~3k of Main work on a
// kernel whose DMA queue is 90% busy and Main 38% busy, so the Main-only path is faster.

pub(crate) fn normalize_key(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    let rms_vrf = root_mean_square(device, x);
    let weight_vrf = load_norm_weight::<Cluster, Slice>(device, rms_weight);
    scale_by_rms_and_weight(device, x, &rms_vrf, &weight_vrf)
}

pub(crate) fn normalize_value(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    let rms_vrf = root_mean_square(device, x);
    scale_by_rms(device, x, &rms_vrf)
}

pub(crate) fn normalize_query<Cluster: M, Slice: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> {
    normalize_query_serial(device, x, rms_weight)
}

#[allow(dead_code)]
fn normalize_key_head_per_slice(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    let x:DmTensor<bf16,Chip,Cluster,m![Ns,1 # 32],m![Ds]> = x.to_dm(&mut device.tdma);
    let x=normalize_head(device,&x,rms_weight);
    x.to_dm(&mut device.tdma)
}

#[allow(dead_code)]
fn normalize_value_head_per_slice(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    let x:DmTensor<bf16,Chip,Cluster,m![Ns,1 # 32],m![Ds]> = x.to_dm(&mut device.tdma);
    let x=normalize_head_bare(device,&x);
    x.to_dm(&mut device.tdma)
}

#[allow(dead_code)]
fn normalize_query_head_per_slice<Cluster: M, Slice: M>(device:&mut Device,x:&DmTensor<bf16,Chip,Cluster,Slice,m![Ns,Gs,Ds]>,rms_weight:&HbmTensor<bf16,Chip,m![Ds]>) -> DmTensor<bf16,Chip,Cluster,Slice,m![Ns,Gs,Ds]> {
    type Heads = m![Ns,Gs,1 # 16];
    let x: DmTensor<bf16,Chip,Cluster,Heads,m![Ds]> = x.to_dm(&mut device.tdma);
    let x = normalize_head(device,&x,rms_weight);
    x.to_dm(&mut device.tdma)
}

fn normalize_head<C:M,S:M>(device:&mut Device,x:&DmTensor<bf16,Chip,C,S,m![Ds]>,weight:&HbmTensor<bf16,Chip,m![Ds]>) -> DmTensor<bf16,Chip,C,S,m![Ds]> {
    let ms:DmTensor<f32,Chip,C,S,m![1 # 8]> = device.main.begin(x.view())
        .fetch::<m![Ds / 16],m![Ds % 16]>().fetch_cast::<f32>()
        .collect::<m![Ds / 8],m![Ds % 8]>().vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4],m![Ds % 4]>().vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),Stash)
        .vector_intra_slice_reduce::<Ds,m![1],m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32).vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add,EPS).vector_final().commit_trim::<m![1 # 8]>().commit();
    let rms:DmTensor<f32,Chip,C,S,m![1 # 8]> = device.main.begin(ms.view())
        .fetch::<m![1],m![1 # 8]>().collect::<m![1],m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt).vector_widen_pad::<m![1 # 8]>()
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let rms:VrfTensor<f32,Chip,C,S,m![1 # 8]> = device.sub.begin(rms.view())
        .fetch::<m![1],m![1 # 8]>().collect::<m![1],m![1 # 8]>().to_vrf();
        let weight=load_norm_weight::<C,S>(device,weight);
        device.main.begin(x.view()).fetch::<m![Ds / 16],m![Ds % 16]>().fetch_cast::<f32>()
            .collect::<m![Ds / 8],m![Ds % 8]>().vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Ds / 4],m![Ds % 4]>()
            .vector_fp_binary(FpBinaryOp::DivF,&rms).vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&weight)
            .vector_widen_concat::<m![Ds / 8],m![Ds % 8]>().vector_final()
            .cast::<bf16,m![Ds % 8 # 16]>().commit_trim::<m![Ds % 8]>().commit()
}
fn normalize_head_bare<C:M,S:M>(device:&mut Device,x:&DmTensor<bf16,Chip,C,S,m![Ds]>) -> DmTensor<bf16,Chip,C,S,m![Ds]> {
    let ms:DmTensor<f32,Chip,C,S,m![1 # 8]> = device.main.begin(x.view())
        .fetch::<m![Ds / 16],m![Ds % 16]>().fetch_cast::<f32>()
        .collect::<m![Ds / 8],m![Ds % 8]>().vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4],m![Ds % 4]>().vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),Stash)
        .vector_intra_slice_reduce::<Ds,m![1],m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32).vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add,EPS).vector_final().commit_trim::<m![1 # 8]>().commit();
    let rms:DmTensor<f32,Chip,C,S,m![1 # 8]> = device.main.begin(ms.view())
        .fetch::<m![1],m![1 # 8]>().collect::<m![1],m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt).vector_widen_pad::<m![1 # 8]>()
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let rms:VrfTensor<f32,Chip,C,S,m![1 # 8]> = device.sub.begin(rms.view())
        .fetch::<m![1],m![1 # 8]>().collect::<m![1],m![1 # 8]>().to_vrf();

        device.main.begin(x.view()).fetch::<m![Ds / 16],m![Ds % 16]>().fetch_cast::<f32>()
            .collect::<m![Ds / 8],m![Ds % 8]>().vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Ds / 4],m![Ds % 4]>()
            .vector_fp_binary(FpBinaryOp::DivF,&rms)
            .vector_widen_concat::<m![Ds / 8],m![Ds % 8]>().vector_final()
            .cast::<bf16,m![Ds % 8 # 16]>().commit_trim::<m![Ds % 8]>().commit()
}
