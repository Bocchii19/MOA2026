# Lane Folder

The Lane Folder is the Contraction Engine’s final stage.
It eliminates the `Lane` dimension by relocating its 8 values into either `OutPacket` (Interleaved) or `OutTime` (Sequential).
No values are summed: the stage folds `Lane` into another axis rather than reducing it.

## Interface

`.contract_lane(mode)` invokes the Lane Folder.
The stage drains the upstream Time Reducer’s buffer through an 8-element-wide output bus one cycle at a time, and `LaneMode` selects what each cycle’s flit carries.

```rust
impl<'l, const T: Tu, D: ContractionAccumulator, Chip: M, Cluster: M, Slice: M, Lane: M, Time: M, Packet: M, B: Backend>
    ContractTimeTensor<'l, T, D, Chip, Cluster, Slice, Lane, Time, Packet, B>
{
    /// Folds the `Lane` dimension into the output stream.
    /// `LaneMode::Interleaved` relocates `Lane` into `OutPacket`;
    /// `LaneMode::Sequential` relocates `Lane` into `OutTime`.
    #[primitive(ContractTimeTensor::contract_lane)]
    pub fn contract_lane<OutTime: M, OutPacket: M>(
        self,
        mode: LaneMode,
    ) -> ContractTensor<'l, T, D, Chip, Cluster, Slice, OutTime, OutPacket, B> {
        verify_contract_lane(ContractLaneInput {
            in_lane: Lane::to_value(),
            in_time: Time::to_value(),
            in_packet: Packet::to_value(),
            out_time: OutTime::to_value(),
            out_packet: OutPacket::to_value(),
            pre_reduce_time: self.pre_reduce_time,
            mode: mode.into(),
        });
        // Earlier stages only retype the deferred operands. Perform their fused contraction here,
        // then apply the lane-fold relayout.
        // `contraction_prewidened` validates `out` through mapping carve. Symbol-wise comparison is
        // invalid because one contracted symbol may span spatial and reduced slots.
        let contraction = self.inner;
        let out = <m![{ Chip }, { Cluster }, { Slice }, { Lane }, { Time }, { Packet }]>::to_value();
        let reduced: Tensor<D, m![{ Chip }, { Cluster }, { Slice }, { Lane }, { Time }, { Packet }], B> =
            Tensor::from_inner(B::contraction_prewidened(
                &contraction.lhs,
                &contraction.rhs,
                &contraction.lhs_map,
                &contraction.rhs_map,
                &contraction.pre_reduce,
                &out,
            ));
        ContractTensor::new(self.device, reduced.transpose(false))
    }
}
```

The minimal examples below take a `ContractTimeTensor` (the output of the upstream Time Reducer) and call only `.contract_lane(...)`, so each example shows the Lane Folder in isolation.
The input `Packet` carries the size that survived the Packet Reducer, one of `{1, 2, 4, 8, 16, 32}` elements per lane.

### Interleaved

The `Lane` dimension folds into `OutPacket`: each cycle reads one column position across all 8 lanes (one value per lane, 8 values per flit), with `Lane` materialized as the innermost `OutPacket`.

```rust
OutTime   = [Time, Packet]
OutPacket = [Lane # 8]
```

```rust
#![allow(unused)]
fn main() {
#![feature(adt_const_params)]
extern crate furiosa_opt_std;
use furiosa_opt_std::prelude::*;
axes![N = 8, M = 4, P = 16];

/// Lane folds into OutPacket.
fn lane_interleaved<'l, const T: Tu>(
    // Input from upstream Time Reducer: Lane = m![N], Time = m![M], Packet = m![P].
    input: ContractTimeTensor<'l, T, f32, m![1], m![1 # 2], m![1 # 256], m![N], m![M], m![P]>,
    // Output: OutTime = m![M, P] = [Time, Packet], OutPacket = m![N] = [Lane].
) -> ContractTensor<'l, T, f32, m![1], m![1 # 2], m![1 # 256], m![M, P], m![N]> {
    input.contract_lane::<m![M, P], m![N]>(LaneMode::Interleaved)
}

let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();

let a: CollectTensor<'_, _, bf16, m![1], m![1 # 2], m![1 # 256], m![M], m![P]> = CollectTensor::new(&mut device.main, Tensor::zero());
let b: TrfTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![N], m![P]> = TrfTensor::zero();

let i: ContractTimeTensor<'_, _, f32, m![1], m![1 # 2], m![1 # 256], m![N], m![M], m![P]> = a 
    .contract_outer::<m![M], m![P], m![N], m![P], _>(&b)
    .contract_packet::<m![P]>()
    .contract_time::<m![M]>();

let _o = lane_interleaved(i);
}
```

### Sequential

The `Lane` dimension folds into `OutTime`: each cycle reads 8 column positions from one lane’s `Packet` (8 values per flit), with `Lane` iterating across successive cycles.
Since each cycle is 8 elements wide, `Packet` is first padded up to a multiple of 8 (the bus width), then split into `PadPacket / 8` cycles per lane and `PadPacket % 8` elements per cycle.

```rust
PadPacket = Packet # align_up(Packet::SIZE, 8)   (pad Packet up to the next multiple of 8)
OutTime   = [Time, Lane, PadPacket / 8]
OutPacket = [PadPacket % 8]
```

For `Packet::SIZE < 32`, `[PadPacket / 8]::SIZE = ceil(Packet::SIZE / 8)` is the number of cycles per packet (e.g., 1 cycle for `Packet::SIZE = 4`, 2 cycles for `Packet::SIZE = 16`).

```rust
#![allow(unused)]
fn main() {
#![feature(adt_const_params)]
extern crate furiosa_opt_std;
use furiosa_opt_std::prelude::*;
axes![N = 8, M = 4, P = 16];

/// Lane folds into OutTime.
fn lane_sequential<'l, const T: Tu>(
    // Input from upstream Time Reducer: Lane = m![N], Time = m![M], Packet = m![P].
    input: ContractTimeTensor<'l, T, f32, m![1], m![1 # 2], m![1 # 256], m![N], m![M], m![P]>,
    // Output: OutTime = m![M, N, P / 8] = [Time, Lane, Packet / 8], OutPacket = m![P % 8] = [Packet % 8].
) -> ContractTensor<'l, T, f32, m![1], m![1 # 2], m![1 # 256], m![M, N, P / 8], m![P % 8]> {
    input.contract_lane::<m![M, N, P / 8], m![P % 8]>(LaneMode::Sequential)
}

let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();

let a: CollectTensor<'_, _, bf16, m![1], m![1 # 2], m![1 # 256], m![M], m![P]> = CollectTensor::new(&mut device.main, Tensor::zero());
let b: TrfTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![N], m![P]> = TrfTensor::zero();

let i: ContractTimeTensor<'_, _, f32, m![1], m![1 # 2], m![1 # 256], m![N], m![M], m![P]> = a 
    .contract_outer::<m![M], m![P], m![N], m![P], _>(&b)
    .contract_packet::<m![P]>()
    .contract_time::<m![M]>();

let _o = lane_sequential(i);
}
```

## Constraints

The Lane Folder has no constraints of its own.
The `LaneMode` selected here determines the slot capacity bound that the upstream Time Reducer enforces (see Time Reducer Constraints).

## Performance

In Interleaved mode, throughput drops by `Lane::SIZE / 8` when `Lane < 8` (inactive lanes leave bus positions empty).

In Sequential mode, when `Packet::SIZE < 8` (e.g., `Packet::SIZE = 4` after the Packet Reducer collapses half of an `bf16` packet), each cycle carries exactly `Packet::SIZE` elements rather than the full 8-element bus width: the output is narrower per cycle, but there is no padding and no wasted bus slots.

Latency is negligible: the Lane Folder reshapes per-lane outputs and does not add cycles beyond the buffer’s drain time.
