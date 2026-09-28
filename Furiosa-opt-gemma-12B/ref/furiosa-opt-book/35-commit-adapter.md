# Commit Adapter

The Commit Adapter applies element-wise transformations to the packet stream before the Commit Engine writes it to DM.
It mirrors the Fetch Adapter on the output side of the Tensor Unit.

The adapter’s stages chain as dedicated `.commit_xxx(...)` methods on the upstream tensor, and the chain always ends in `.commit(...)` for the actual DM write.
Trimming is the mandatory first stage: `.commit()` / `.commit_view()` are reachable only after `.commit_trim(...)`, so every commit is trimmed first (it is how flit padding is dropped).
The other public stage is optional type casting, and main and sub contexts then diverge.
Generate Mode is a separate sub-context path used internally by `memset` for immediate fills; it does not chain off a `TuTensor`.

- Main pipeline: Trimming → Type Casting (optionally fusing ReLU) → `.commit()`.
- Sub bypass: Generate Mode writes a constant directly for `memset`, without consuming an upstream Tensor Unit stream.

| Operation | Main | Sub |
|---|---|---|
| Trimming | ✅ | ✅ |
| Type Casting (optional fused ReLU) | ✅ | ❌ |
| Generate Mode (internal `memset` path) | ❌ | ✅ |

## Trimming

Stream packets in the Tensor Unit pipeline are always 32-byte *flits* (see Collect Engine), but a flit may carry fewer valid elements than its capacity, with trailing elements filled by padding.
Writing the full flit verbatim would clobber DM bytes beyond the valid region with the flit’s padding values.

Trimming solves this by writing only the leading `valid_size` elements of each flit to DM, discarding the trailing padding.
The compiler derives `valid_size` from the output tensor mapping.
Users do not set it directly.
`D[valid_size]` must be 8, 16, 24, or 32 bytes (where 32 means no trim).
Trimming adds nearly zero latency.

Trimming is the mandatory first stage of the Commit Adapter, even though not every commit has padding to drop: when `valid_size` is already 32 bytes the flit is fully valid and the trim is a no-op.
It is mandatory because `.commit()` is reachable only after `.commit_trim(...)`, so it anchors the chain and runs ahead of Type Casting (main).

```rust
// `D: MaterializableScalar` here (trim is the commit path's mandatory first stage) keeps i5/i9 uncommittable.
impl<
    'l,
    const T: Tu,
    P: CanApplyCommitTrim,
    D: MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Commit Adapter's trimming stage.
    ///
    /// Drops the trailing padding from each flit so DM stores only valid
    /// elements. `OutPacket` is the post-trim layout the kernel
    /// promises; the compiler derives the trim count from the input and
    /// output mappings.
    #[primitive(TuTensor::commit_trim)]
    pub fn commit_trim<OutPacket: M>(self) -> CommitTrimTensor<'l, T, D, Chip, Cluster, Slice, Time, OutPacket, B> {
        verify_commit_trim::<D, Packet, OutPacket>();
        // `transpose(false)` is type-system filler; real trim lowering lands with the backend wiring.
        CommitTrimTensor::new(self.device, self.inner.transpose(false))
    }
}
```

`.commit_trim::<OutPacket>()` declares the post-trim packet, and the chained `.commit(...)` then performs the DM write on the trimmed stream.
The two are fully separate.

```rust
#![feature(adt_const_params)]
extern crate furiosa_opt_std;
use furiosa_opt_std::prelude::*;
axes![M = 4, K = 2, W = 8, N = 16, J = 64];

fn commit_trim_i8_padding<'l, const T: Tu>(
    input: CastTensor<'l, T, i8, m![1], m![1], m![1], m![M, K], m![W # 32]>,
) -> CommitTrimTensor<'l, T, i8, m![1], m![1], m![1], m![M, K], m![W]> {
    // 8 valid i8 out of 32 padded; OutPacket drops the `# 32` padding.
    input.commit_trim::<m![W]>()
}

fn commit_trim_f32_non_padding<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1], m![1], m![M, K], m![W]>,
) -> CommitTrimTensor<'l, T, f32, m![1], m![1], m![1], m![M, K], m![W = 4]> {
    // 4 valid f32 out of 8; OutPacket resizes `W` to 4.
    input.commit_trim::<m![W = 4]>()
}

fn commit_trim_bf16_with_transpose<'l, const T: Tu>(
    input: CastTensor<'l, T, bf16, m![1], m![1], m![1], m![M, K], m![N]>,
) -> CommitTrimTensor<'l, T, bf16, m![1], m![1], m![1], m![M, K], m![N = 8]> {
    // 8 valid bf16 out of 16; OutPacket resizes `N` to 8.
    input.commit_trim::<m![N = 8]>()
}

fn commit_trim_i4_no_trim<'l, const T: Tu>(
    input: CastTensor<'l, T, i4, m![1], m![1], m![1], m![M, K], m![J]>,
) -> CommitTrimTensor<'l, T, i4, m![1], m![1], m![1], m![M, K], m![J]> {
    // No trimming; `OutPacket == Packet`.
    input.commit_trim::<m![J]>()
}

let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
let a: CastTensor<'_, _, i8, m![1], m![1], m![1], m![M, K], m![W # 32]> = CastTensor::new(&mut device.main, Tensor::zero());
let _o = commit_trim_i8_padding(a);
let b: ContractTensor<'_, _, f32, m![1], m![1], m![1], m![M, K], m![W]> = ContractTensor::new(&mut device.main, Tensor::zero());
let _o = commit_trim_f32_non_padding(b);
let c: CastTensor<'_, _, bf16, m![1], m![1], m![1], m![M, K], m![N]> = CastTensor::new(&mut device.main, Tensor::zero());
let _o = commit_trim_bf16_with_transpose(c);
let d: CastTensor<'_, _, i4, m![1], m![1], m![1], m![M, K], m![J]> = CastTensor::new(&mut device.main, Tensor::zero());
let _o = commit_trim_i4_no_trim(d);
```

## Type Casting

Type casting converts `f32` data to `bf16` format on the commit path, optionally fusing a ReLU activation into the same pass.
The Cast Engine handles most type conversions in the Tensor Unit pipeline.
Commit Adapter type casting exists for one specific case, running main-context contraction in parallel with sub-context Vector Engine work.
The Cast Engine sits on top of the Vector Engine and so occupies it during a conversion.
If the main-context performed its `f32` → `bf16` conversion through the Cast Engine, the Vector Engine would be busy and the sub-context could not run in parallel.
Routing the conversion through the Commit Adapter instead leaves the Vector Engine free for the sub-context.
Sub-context itself does not support type casting (consistent with the support matrix above).

`commit_cast` and `commit_cast_relu` are separate methods because they are separate hardware conversions, not one conversion with a mode: ReLU has no standalone stage to select at run time, and exists only fused with the narrowing cast.

```rust
impl<
    'l,
    const T: Tu,
    P: CanApplyCommitCast,
    D: MaterializableScalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
    B: Backend,
> TuTensor<'l, T, P, D, Chip, Cluster, Slice, Time, Packet, B>
{
    /// Runs the Commit Adapter's type-casting stage.
    ///
    /// Folds the `f32` → `bf16` cast into the commit path, leaving the
    /// [Cast Engine](crate::engine::cast) free for sub-context Vector Engine
    /// work. See [`CommitCast`](crate::prelude::CommitCast) for why this is the
    /// only conversion, and [`commit_cast_relu`](Self::commit_cast_relu) for the
    /// variant that fuses a ReLU.
    #[primitive(TuTensor::commit_cast)]
    pub fn commit_cast<OutD: Scalar>(self) -> CommitCastTensor<'l, T, OutD, Chip, Cluster, Slice, Time, Packet, B>
    where
        D: CommitCast<OutD>,
    {
        verify_commit_cast::<Packet>();
        CommitCastTensor::new(self.device, self.inner.map(|v| v.cast()))
    }

    /// The same cast with a ReLU fused in, clamping negative values to zero.
    ///
    /// A separate method rather than an argument because the two are separate
    /// hardware conversions (`CommitF32ToBf16` and `CommitF32ToBf16Relu`), and
    /// ReLU has no standalone stage to select at run time.
    #[primitive(TuTensor::commit_cast_relu)]
    pub fn commit_cast_relu<OutD: Scalar>(self) -> CommitCastTensor<'l, T, OutD, Chip, Cluster, Slice, Time, Packet, B>
    where
        D: CommitCast<OutD>,
    {
        verify_commit_cast::<Packet>();
        CommitCastTensor::new(self.device, self.inner.map(|v| v.cast_relu()))
    }
}
```

```rust
#![feature(adt_const_params)]
extern crate furiosa_opt_std;
use furiosa_opt_std::prelude::*;
axes![N = 4, C = 3, H = 4, W = 8];

fn commit_cast_example<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1], m![1], m![N, C, H], m![W]>,
) -> CommitCastTensor<'l, T, bf16, m![1], m![1], m![1], m![N, C, H], m![W]> {
    // Cast f32 to bf16 (values preserved), no activation. A real main
    // commit runs `.commit_trim()` first, then `.commit(...)` after.
    // W = 8 f32 elements (32 bytes) stays 8 bf16 elements (16 bytes).
    input.commit_cast::<bf16>()
}

fn commit_cast_relu_example<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1], m![1], m![N, C, H], m![W]>,
) -> CommitCastTensor<'l, T, bf16, m![1], m![1], m![1], m![N, C, H], m![W]> {
    // f32 -> bf16 with a fused ReLU: negative values clamped to zero.
    // e.g. [-5.0, -0.1, 0.0, 3.7] -> [0.0, 0.0, 0.0, 3.7]
    input.commit_cast_relu::<bf16>()
}
```
