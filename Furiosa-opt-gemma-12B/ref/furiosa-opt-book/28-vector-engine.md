# Vector Engine

The Vector Engine performs elementwise computation and reduction.
Examples include activations (GELU, SiLU), normalizations (softmax, layer norm), binary operations, and intra- and inter-slice reductions.

The engine accepts only 32-bit types, `i32` and `f32`.
An upstream Contraction Engine widens types automatically (`bf16` products accumulate in `f32`, `i8` products in `i32`).
When that engine is bypassed, the Fetch Adapter must widen the input via its type-cast adapter.

## Interface

In a single Tensor Unit invocation, the Vector Engine portion is the method chain from `vector_init()` to `vector_final()`.
The engine has two sub-pieces: the Intra-Slice Chain (elementwise / binary / per-slice reduce stages) and the Inter-Slice Reducer (reduces across the 256 slices in a cluster).
Between `vector_init()` and `vector_final()`, the chain alone, the reducer alone, or both can run.
When both run, the order is `IntraFirst` (chain then reducer) or `InterFirst` (reducer then chain).

The intra-slice chain is entered via `vector_intra_slice_tag()`, or `vector_intra_slice_unzip()` when the input carries a 2-way grouping axis to split into two parallel streams (see Pair Mode).
Either entry point fires right after `vector_init()` or on the inter-slice reducer’s output.
The inter-slice reducer is entered via `vector_inter_slice_reduce()`, either right after `vector_init()` or from a compatible intra-slice stage.
For stage-by-stage API coverage, see Intra-Slice Chain and Inter-Slice Reducer.

The signatures below cover the `vector_init()`-side entry methods only.
The same method names (`vector_intra_slice_tag`, `vector_inter_slice_reduce`) also exist on chain and reducer tensors for the chain↔reducer transitions; those are documented in the child pages.

```rust
impl<'l, const T: Tu, P: CanApplyVectorInit, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Initializes Vector Engine processing for this tensor.
    #[primitive(TuTensor::vector_init)]
    pub fn vector_init(self) -> VectorInitTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet> {
        VectorInitTensor::new(self.device, self.inner)
    }
}

impl<'l, const T: Tu, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorInitTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Enters VE intra-slice pipeline (single stream).
    #[primitive(VectorInitTensor::vector_intra_slice_tag)]
    pub fn vector_intra_slice_tag(
        self,
        branch: TagMode<D>,
    ) -> VectorBranchTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet, Fresh, { VeOrder::IntraFirst }> {
        VectorBranchTensor::new(self.device, self.inner, branch)
    }

    /// Enters VE intra-slice pipeline (two-group / unzip).
    #[primitive(VectorInitTensor::vector_intra_slice_unzip)]
    pub fn vector_intra_slice_unzip<I: AxisName, SplitTime: M>(
        self,
    ) -> VectorTensorPair<'l, T, D, stage::Tag, Chip, Cluster, Slice, SplitTime, Packet> {
        verify_vector_intra_slice_unzip::<I, Time, SplitTime, Packet>();
        VectorTensorPair::new::<I, Time>(self.device, self.inner)
    }
}

impl<'l, const T: Tu, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M>
    VectorInitTensor<'l, T, i32, Chip, Cluster, Slice, Time, Packet>
{
    /// Performs inter-slice reduce for i32 as the first VE operation.
    #[primitive(VectorInitTensor::vector_inter_slice_reduce)]
    pub fn vector_inter_slice_reduce<OutSlice: M, OutTime: M>(
        self,
        op: InterSliceReduceOpI32,
    ) -> VectorInterSliceReduceTensor<'l, T, i32, Chip, Cluster, OutSlice, OutTime, Packet, { VeOrder::InterFirst }>
    {
        let reduced = self.inner.reduce(op.reduce_fn(), op.identity(), true);
        create_inter_slice_reduce_tensor(self.device, reduced)
    }
}
```

## Examples

A few representative examples follow to give a feel for what a Vector Engine call looks like, with the full API tour deferred to the child pages Intra-Slice Chain and Inter-Slice Reducer.

### ReLU Activation

This pass applies ReLU elementwise, computing \(output[b, k, m, n] = \max(input[b, k, m, n], 0)\).

```rust
#![allow(unused)]
fn main() {
#![feature(adt_const_params)]
extern crate furiosa_opt_std;
use furiosa_opt_std::prelude::*;
axes![B = 2, K = 256, M = 16, N = 16];

// ReLU activation after batched matrix multiplication.
// Chain-only pass, so the reducer is skipped and the path trivially resolves to IntraFirst.
// Both clusters (B = 2) and all 256 slices (K) carry real data, no padding.
fn relu<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![B], m![K], m![M, N / 8], m![N % 8]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![K], m![M, N / 8], m![N % 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // max(x, 0), the ReLU itself
        .vector_clip(ClipBinaryOpF32::Max, 0.0f32)
        .vector_final()
}

let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();

let c: ContractTensor<'_, _, f32, m![1], m![B], m![K], m![M, N / 8], m![N % 8]> = ContractTensor::new(&mut device.main, Tensor::zero());
let _o = relu(c);
}
```

### ReLU Then Reduce

This pass applies ReLU per slice and then reduces across `R`, giving \(output[a, b] = \sum_{r \in R} \max(input[a, b, r], 0)\).

```rust
#![allow(unused)]
fn main() {
#![feature(adt_const_params)]
extern crate furiosa_opt_std;
use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, R = 4];

// Chain applies ReLU, then reducer sums across slices.
// IntraFirst shape (chain runs first, then reducer).
// Both clusters (B = 2), all 256 slices (A / 8 * R), and full Way8 packet (A % 8) carry real data.
fn relu_then_reduce<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8, 1 # 4], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // max(x, 0), the ReLU
        .vector_clip(ClipBinaryOpI32::Max, 0)
        // sum across R slices
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
}

let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();

let c: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]> = CollectTensor::new(&mut device.main, Tensor::zero());
let _o = relu_then_reduce(c);
}
```

### Reduce Then Bias

This pass reduces across `R` and then adds a constant bias, giving \(output[a, b] = \left(\sum_{r \in R} input[a, b, r]\right) + 100\).

```rust
#![allow(unused)]
fn main() {
#![feature(adt_const_params)]
extern crate furiosa_opt_std;
use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, R = 4];

// Reducer sums across slices, then chain adds a bias to the reduced result.
// InterFirst shape (reducer runs first, then chain).
// Both clusters (B = 2), all 256 slices, and full Way8 packet carry real data.
fn reduce_then_add<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8, 1 # 4], m![1], m![A % 8]> {
    input
        .vector_init()
        // sum across R slices
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_intra_slice_tag(TagMode::Zero)
        // add bias 100
        .vector_fxp(FxpBinaryOp::AddFxp, 100)
        .vector_final()
}

let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();

let c: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]> = CollectTensor::new(&mut device.main, Tensor::zero());
let _o = reduce_then_add(c);
}
```

### Intra- and Inter-Slice Reduction

This pass reduces `R` entirely by combining the intra-slice reducer (over `R`’s Time and Packet portions) and the inter-slice reducer (over `R`’s Slice portion).
The einsum form is `BR -> B`, with saturating addition.

```rust
#![allow(unused)]
fn main() {
#![feature(adt_const_params)]
extern crate furiosa_opt_std;
use furiosa_opt_std::prelude::*;
axes![B = 2, R = 8192];

// R splits across Slice (R / 32 = 256), Time (R % 32 / 4 = 8), and Packet (R % 4, padded to 8 in Way8).
// Chain runs intra-slice reduce over R's Time and Packet portions, then the reducer collapses the Slice portion.
// IntraFirst shape (chain runs first, then reducer).
fn full_sum<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![R / 32], m![R % 32 / 4], m![R % 4 # 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![1 # 256], m![1], m![1 # 8]> {
    input
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // Way8 → Way4 (back 4 packet positions were padding)
        .vector_narrow_trim::<m![R % 4]>()
        // sum over R's Time and Packet portions
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(IntraSliceReduceOpI32::AddSat)
        // Way4 → Way8
        .vector_widen_pad::<m![1 # 8]>()
        // sum over R's Slice portion across all 256 slices
        .vector_inter_slice_reduce::<m![1 # 256], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
}

let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();

let c: CollectTensor<'_, _, i32, m![1], m![B], m![R / 32], m![R % 32 / 4], m![R % 4 # 8]> = CollectTensor::new(&mut device.main, Tensor::zero());
let _o = full_sum(c);
}
```

### Pair Add

This pass unzips two interleaved groups along `I` and adds them pair-wise.
The einsum form is `ABI -> AB`.

```rust
#![allow(unused)]
fn main() {
#![feature(adt_const_params)]
extern crate furiosa_opt_std;
use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 2, I = 2];

// Pair-mode entry via unzip, then a zip op fuses the two streams with an add.
// Both clusters (B = 2), all 256 slices (A / 8), and full Way8 packet (A % 8) carry real data.
fn pair_add<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8], m![I], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8], m![1], m![A % 8]> {
    input
        .vector_init()
        // split into group 0 and group 1 along I
        .vector_intra_slice_unzip::<I, m![1]>()
        // group0 + group1
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
}

let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();

let c: CollectTensor<'_, _, i32, m![1], m![B], m![A / 8], m![I], m![A % 8]> = CollectTensor::new(&mut device.main, Tensor::zero());
let _o = pair_add(c);
}
```
