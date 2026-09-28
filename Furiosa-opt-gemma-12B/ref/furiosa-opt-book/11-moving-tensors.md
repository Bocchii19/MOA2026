# Moving Tensors

Furiosa-opt kernels are Rust functions compiled for the Furiosa NPU.
This chapter explains how a kernel transfers tensor data between host memory, HBM, DM, and the Tensor Unit.
The Quick Start kernel design introduces these tiers in a complete kernel.
Choose the source and destination in the movement table before reading the worked transfer.
The Sequencer describes the common loop model.
The engine pages document each concrete boundary.

## Choose a movement route

The movement table selects operations from the current value to its destination.
It gives the operation and context for each public boundary.

| Current value | Destination | Operation and context |
|---|---|---|
| `HostTensor` | `HbmTensor` | `HostTensor::to_hbm` with `Device::pdma` (async); see DMA Engine |
| `HbmTensor` | `HostTensor` | `HbmTensor::to_host` with `Device::pdma` (async); see DMA Engine |
| `HbmTensor` | `DmTensor` | `HbmTensor::to_dm` with `Device::tdma`; see DMA Engine |
| `DmTensor` | `HbmTensor` | `DmTensor::to_hbm` with `Device::tdma`; see DMA Engine |
| `HbmTensor` | `HbmTensor` | `HbmTensor::to_hbm` with the DMA context required by the call; see DMA Engine |
| `DmTensor` | `DmTensor` | `DmTensor::to_dm` with `Device::tdma`, or a fetch/commit round trip through the Tensor Unit; see DMA Engine and Commit Engine |
| `DmTensorView` | Tensor Unit stream | `TuContext::begin`, then `fetch` |
| Tensor Unit stream | `DmTensor` | `commit`, or `commit_view` for an existing mutable view |

## Copyable end-to-end transfer

This page defines movement boundaries, context ownership, and lifetime rules.
The worked transfer widens and reorders an HBM tensor before returning it to HBM.
The device function stages HBM into DM, fetches a stream, casts and collects it, commits to DM, and transfers the result back to HBM.
The host test awaits only host-to-HBM and HBM-to-host I/O and checks the reordered result.

Read and change the worked transfer in this order:

| Step | Declared choice |
|---|---|
| Input | `A = 4096`, `B = 8`, input element type `i8`, and input HBM mapping `m![1], m![A, B]`. |
| HBM→DM | The DM `Cluster` is `m![1 # 2]`, `Slice` is `m![A / 16]`, and `Element` is `m![A / 8 % 2, A % 8, B]`. |
| Fetch | `OutTime` is `m![A / 8 % 2]` and `OutPacket` is `m![A % 8, B]`. |
| Compute and Collect | The stream casts `i8` to `i32`, then Collect uses `m![A / 8 % 2, A % 8]` for `Time` and `m![B]` for `Packet`. |
| Commit→DM→HBM | `commit_trim::<m![B]>()` writes the DM tensor in `m![A, B]`; `to_hbm` then returns `HbmTensor<i32, m![1], m![B, A]>` through DMA relayout. |

Change only these declared axes, mappings, or element types while preserving the alignment assertions in the source.
The host test awaits `to_hbm` and `to_host` through `pdma`, then asserts the expected `B`-major result.

```rust
#![expect(clippy::type_complexity)]

use furiosa_opt_std::prelude::*;

axes![A = 4096, B = 8];

type Chip = m![1];
type Cluster = m![1 # 2];

#[device(chip = 1)]
pub fn fetch_commit_simple(
    device: &mut Device,
    input: &HbmTensor<i8, m![1], m![A, B]>,
) -> HbmTensor<i32, m![1], m![B, A]> {
    // Element's innermost axis must match the source's innermost (B, stride 1) so that the
    // DMA tail contains the full B axis (8 × i8 = 8 bytes), satisfying min_align = 8.
    let input_dm = input.to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut device.tdma);

    let fetch_and_commit_tensor: DmTensor<i32, Chip, Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![A / 8 % 2], m![A % 8, B]>()
        .fetch_cast::<i32>()
        .collect::<m![A / 8 % 2, A % 8], m![B]>()
        .commit_trim::<m![B]>()
        .commit();

    fetch_and_commit_tensor.to_hbm(&mut device.tdma)
}
```

```rust
use furiosa_opt_examples::fetch_commit::fetch_commit_simple;
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn test_fetch_commit_simple_host() {
    use furiosa_opt_examples::fetch_commit::{A, B};

    let mut device = Device::new(fetch_commit_simple.topology()).unwrap();

    // Create input tensor with shape (A=4096)(B=8).
    let input = HostTensor::<i8, m![A, B]>::from_vec((0..32768).map(|x| x as i8).collect::<Vec<_>>())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();

    // Call the device function.
    let output = launch(fetch_commit_simple, (&mut device, &input)).await.unwrap();

    let mut expected = vec![0i32; 4096 * 8];
    let mut idx = 0;
    for b in 0..8 {
        for a in 0..4096 {
            expected[idx] = ((a * 8 + b) as i8) as i32;
            idx += 1;
        }
    }

    assert_eq!(
        output.to_host::<m![B, A]>(&mut device.pdma).await.unwrap().into_inner(),
        Tensor::<_, m![B, A], CurrentBackend>::from_vec(expected)
    );
}
```

The included source is the complete example for the ordering and lifetime rules described below.
This chapter keeps that transfer as a readable HBM↔DM and lifetime tutorial; the linked Tensor Unit I/O pattern owns the detailed packet and performance comparisons.

The type-level `Chip`, `Cluster`, `Slice`, `Time`, and `Element` mappings define each value’s storage and stream behavior.
A DMA call creates a destination with the mapping supplied by its type parameters.
It does not mutate the source.
Fetch and commit are the hand-off points into and out of the Tensor Unit pipeline.

For a first pipeline, stage an HBM input into DM, call `begin(...).fetch(...)`, run the compute stages, then commit the stream to DM and transfer the result back to HBM.
The Case Study: Tensor Unit I/O shows this order with concrete mappings and source code.

## Keep transfers ordered

`Device::new(Topology { chips, pes })` opens the chips and is what a launch runs on, carrying independent `main` and `sub` Tensor Unit contexts plus `tdma` and `pdma` DMA contexts.
Pass the matching mutable context to each operation.
Rust borrows enforce that a Tensor Unit stream keeps its source view alive.
`begin` borrows a `DmTensorView`, and the stream lifetime cannot outlive that view.
Fetch consumes the `BeginTensor` and produces a stream.
Adapters such as `fetch_cast` and `collect` consume and return the next stage.
Commit consumes the final stream, or consumes it while writing to a supplied `DmTensorViewMut`.

Host transfers `to_hbm` and `to_host` are async I/O and must be awaited.
Kernel-side `to_dm`, `to_hbm`, fetch, and commit operations enqueue device commands on their selected contexts.
Keep source and destination handles alive until the enclosing kernel or backend submission has completed.
Use a mutable view only when the destination region is intentionally overwritten.
Do not read a destination before its producing transfer or commit has completed, and do not overlap writes to aliased views without an ordering edge.

## Engine boundaries

- **Fetch** reads DM with one per-slice sequencer and emits a packet stream for the Tensor Unit.
Its output mapping chooses the stream `Time` and `Packet` axes.
Axis lifting can also move an axis from `Time` to `Chip`, `Cluster`, or `Slice` when the selected dimension contains a broadcast.
- **Commit** writes a Tensor Unit stream to DM.
Its `Element` mapping chooses the destination layout and can transpose stream axes during the write.
- **DMA** pairs read and write sequencers for memory-to-memory movement.
The supported public paths are HBM to HBM, HBM to DM, DM to HBM, and DM to DM.
Indirect gather/scatter APIs have separate index-unit contracts: unscaled gather uses a `DmTensor` index with raw row positions, while unscaled scatter is currently unimplemented.
SPM has no public tensor type.

The Tensor Unit pipeline is therefore:

```rust
flowchart LR
    H[Host] <-->|PCIe DMA| B[HBM]
    B <-->|Tensor DMA| D[DM]
    D -->|begin + fetch| F[Tensor Unit stream]
    F -->|commit| D
    F --> A[Adapters / switch / collect]
    A --> C[Compute]
    C -->|commit| D
```

Collect can place a stream operand in the TRF or VRF register files for Contraction or Vector work.
Those compute boundaries are covered in Computing Tensors.

## Tune movement

Every movement is a mathematical tensor move: the logical values are preserved while the destination mapping may change.
The compiler derives sequencer strides from both mappings.
Respect the engine constraints before tuning performance: DM `Cluster` must be 1 or 2, DM `Slice` must be 64, 128, or 256, and DMA packet tails must satisfy the element-size alignment checked by the API.
Fetch and commit also validate their mapping-specific sequencer rules.
Invalid mappings fail during compilation or command verification.

Prefer contiguous packet layouts and mappings that distribute traffic across clusters, slices, DM banks, and HBM channels.
Small or strided packets increase the number of sequencer accesses.
Conflicting DM-bank patterns can starve DMA.
See Memory Performance for the bank-starvation rule, HBM interleaving, and packet-size trade-offs.

For a complete end-to-end movement pattern, see Case Study: Tensor Unit I/O.
For indirect HBM row movement, see DMA gather and scatter.

## See also

The furiosa-opt-std rustdoc documents the release API.
Use the engine pages above for operation-specific contracts and Scheduling and Tuning when a valid movement plan needs measured overlap or hazard diagnosis.
