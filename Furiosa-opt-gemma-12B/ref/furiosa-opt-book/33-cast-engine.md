# Cast Engine

The Cast Engine narrows `f32`/`i32` pipeline results to lower-precision types (e.g., `bf16`) before the Commit Engine writes them to DM, reducing storage cost.

## Interface

`CollectTensor`, `ContractTensor`, and `VectorFinalTensor` all expose `.cast()` with the same semantics.

```rust
//
// The Cast Engine accepts only `VeScalar` inputs (hardware constraint), so the
// bound lives on the impl rather than on a wider trait.
impl<'l, const T: Tu, P: CanApplyCast, D: VeScalar, Chip: M, Cluster: M, Slice: M, Time: M, Packet: M, B: Backend>
    TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Casts each element to type `OutD` and pads the output packet back to one
    /// 32-byte flit.
    #[primitive(TuTensor::cast)]
    pub fn cast<OutD: Scalar, OutPacket: M>(self) -> CastTensor<'l, T, OutD, Chip, Cluster, Slice, Time, OutPacket, B>
    where
        D: CastEngineCast<OutD>,
    {
        verify_cast::<D, OutD, Packet, OutPacket>();
        CastTensor::new(self.device, self.inner.map(|v| v.cast()).transpose(false))
    }
}
```

`.cast::<OutD, OutPacket>()` converts each element to type `OutD` and pads the output back to one 32-byte flit.
The kernel writer chooses `OutD` (the target type) and `OutPacket` (the output element layout).
The compiler derives the rest.

Although the Cast Engine is not a mathematical tensor move, it preserves the tensor’s shape and only changes the element type.
All dimensions pass through unchanged, except for the `Packet` layout, which repads so the output still fits one 32-byte flit.

The example below casts an 8-element `i32` packet (8 × 4 = 32 bytes) to `i8`.
After the cast, the 8 elements occupy 8 bytes, so `A # 32` pads the output back to 32 bytes:

```rust
#![allow(unused)]
fn main() {
#![feature(adt_const_params)]
extern crate furiosa_opt_std;
use furiosa_opt_std::prelude::*;
axes![B = 4, A = 8];

fn cast_i32_to_i8<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1 # 2], m![1 # 256], m![B], m![A]>,
) -> CastTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![B], m![A # 32]> {
    input.cast()
}

let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();

let c: CollectTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 256], m![B], m![A]> = CollectTensor::new(&mut device.main, Tensor::zero());
let _o = cast_i32_to_i8(c);
}
```

The input data may not fill 32 bytes.
The example below casts an `i32` input where 4 data elements are padded to 8 (`A # 8`, 32 bytes) into an `i8` output where the same 4 elements are padded to 32 (`A # 32`, also 32 bytes):

```rust
#![allow(unused)]
fn main() {
#![feature(adt_const_params)]
extern crate furiosa_opt_std;
use furiosa_opt_std::prelude::*;
axes![A = 4];

fn cast_padded<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1 # 2], m![1 # 256], m![1], m![A # 8]>,
) -> CastTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![1], m![A # 32]> {
    input.cast()
}

let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();

let c: CollectTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 256], m![1], m![A # 8]> = CollectTensor::new(&mut device.main, Tensor::zero());
let _o = cast_padded(c);
}
```

## Supported Casts

Each input is a 32-byte flit.
The supported source types are `f32` and `i32`, each with specific target types:

| Input Type (`D`) | Supported Output Types (`OutD`) |
|---|---|
| `i32` | `i4`, `i8`, `i16` |
| `f32` | `f8e5m2`, `f8e4m3`, `bf16` |

## Performance

The Cast Engine is never the pipeline bottleneck: it processes one flit per cycle regardless of how much of the flit carries valid data.
The downstream Commit Engine aggregates under-utilized flits into dense DM writes, so no DM bandwidth is wasted.
