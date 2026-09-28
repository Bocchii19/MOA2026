# Kernel Design

## First Vector Kernel

### Constant Addition

#### Goal

Compute \(\text{out}[i] = \text{in}[i] + 1\) for every element of a vector.
This pattern covers residual and bias addition.

#### Data Movement

Move the vector from HBM to DM, stream each DM slice through Fetch, Collect, the Vector Engine, and Commit, and move the result from DM back to HBM.

```rust
flowchart TB
    HOST[Host] <-->|PCIe DMA| HBM[(HBM)]
    HBM <-->|Tensor DMA| DM[(DM)]

    subgraph TU[Tensor Unit]
        direction TB
        FE[Fetch] --> CO[Collect] --> VE["Vector AddFxp +1"] --> CM[Commit]
    end

    DM -->|stream| FE
    CM -->|stream| DM
```

#### Device Source

The kernel uses one chip, one of two clusters, and all 256 slices in that cluster.
Each slice receives eight `i32` values.
`TagMode::Zero` executes the Vector Engine on every cycle.
`to_dm` distributes the vector across slices, and the `begin → fetch → collect → vector_init → vector_intra_slice_tag → vector_fxp → vector_final → commit` chain processes each slice in one pass.

Device source: `src/kernel/constant_add_kernel.rs`.

```rust
use furiosa_opt_std::prelude::*;

axes![A = 2048];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![A / 8 # 256];

#[device(chip = 1)]
pub fn constant_add_kernel(device: &mut Device, input: &HbmTensor<i32, Chip, m![A]>) -> HbmTensor<i32, Chip, m![A]> {
    // HBM → DM: split 2048 elements across 256 slices (8 elements per slice)
    let dm = input.to_dm::<Cluster, Slice, m![A % 8]>(&mut device.tdma);

    let result = device
        .main
        .begin(dm.view())
        // Fetch: stream 8-element packets from DM into the pipeline
        .fetch::<m![1], m![A % 8]>()
        // Collect: normalize the stream into 32-byte flits (8 × i32)
        .collect::<m![1], m![A % 8]>()
        // Vector Engine: enter pipeline and arm unconditionally
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // Add the scalar constant 1 to every element
        .vector_fxp(FxpBinaryOp::AddFxp, 1)
        // Exit VE and commit: trim the packet to the commit width, then write
        // results back to DM
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>();

    // DM → HBM
    result.to_hbm(&mut device.tdma)
}
```

#### Host Program and Oracle

The host program creates input, uploads it to HBM, launches the device function, and compares the result with the host equation in its test.

Base-template host program and oracle: `src/constant_add.rs`.

```rust
use furiosa_opt_std::prelude::*;
use {{ crate_name }}::kernel::constant_add_kernel::{A, constant_add_kernel};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut device = Device::new(constant_add_kernel.topology())?;
    let mut rng = SmallRng::seed_from_u64(42);
    let input = HostTensor::<i32, m![A]>::rand(&mut rng);
    let in_hbm = input.to_hbm(&mut device.pdma).await?;
    let _out_hbm = launch(constant_add_kernel, (&mut device, &in_hbm)).await?;
    println!("Constant Add: kernel ran");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_reference() {
        let mut device = Device::new(constant_add_kernel.topology()).unwrap();

        let mut rng = SmallRng::seed_from_u64(42);
        let input = HostTensor::<i32, m![A]>::rand(&mut rng);
        let in_hbm = input.to_hbm(&mut device.pdma).await.unwrap();

        // Reference: out[i] = in[i] + 1.
        let in_buf: Vec<i32> = input.into_vec();
        let expected: Vec<i32> = in_buf.iter().map(|&x| x.wrapping_add(1)).collect();

        let out_hbm = launch(constant_add_kernel, (&mut device, &in_hbm)).await.unwrap();

        let actual: Vec<i32> = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap().into_vec();
        for (i, (&e, &a)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(e, a, "constant_add mismatch at i={i}: expected {e}, actual {a}");
        }
    }
}
```

#### Pattern-Specific Mapping and Memory

TCP has four nested hardware levels.

| Level | Count (RNGD) | Role |
|---|---|---|
| `Chip` | System-dependent | Top-level unit that holds HBM. |
| `Cluster` | Two per chip | Group of 256 slices. |
| `Slice` | 256 per cluster | One Tensor Unit. |
| `Lane` | Eight per slice | One row of the Contraction Engine MAC array. |

A tensor type encodes its element type and the distribution of each logical axis across this hierarchy.
For example, `DmTensor<bf16, m![1], m![1 # 2], m![A / 8 # 256], m![A % 8]>` represents a `bf16` vector with axis `A` split across 256 slices with eight elements per slice.
Each element of `A` has one well-defined position in one slice.

Mapping operators describe where an axis goes.

- `/` splits an axis by stride.
- `%` gives the in-unit count.
- `#` pads a mapping to the hardware unit count.
- `=` reduces how much of an existing term a view covers, which is how a tile selects one piece of a tensor.

`%` and `=` both narrow an axis, so it is worth stating what separates them.
`%` introduces a new inner term: in `m![B / 2048, B % 2048]` the first term counts the 2048-element blocks of `B` and the second counts the positions inside one block.
`=` leaves the terms alone and shrinks the extent the view reads, so `m![B / 2048 = 1 # 2]` selects one of the two blocks, where `# 2` keeps the stride of the full axis rather than of the selection.
A tiled kernel uses them together, as in `m![B / 2048 = 1 # 2, B % 2048]`, which picks one block and covers every position within it.
See Tiling for the general rule.

Tensor Unit tensors also use `Time` for pipeline iterations and `Packet` for elements within an iteration.
The Tensor Unit pipeline is Fetch → Switch → Collect → Contraction → Vector → Cast → Transpose → Commit.
The Switch Engine connects slices while most stages operate independently within a slice.

| Type | Location | Capacity (RNGD) | Role |
|---|---|---|---|
| `HbmTensor` | On-package | 48 GB and 1.5 TB/s | Long-term weight and activation storage. |
| `DmTensor` | On-chip SRAM | 256 MB total and 512 KB per slice | Primary working memory. |
| `TrfTensor` | On-chip SRAM | 8 KB per lane and eight lanes per slice | Contraction Engine register file. |
| `VrfTensor` | On-chip SRAM | 8 KB per slice | Vector Engine operand register file. |

These choices specialize the constant-add pattern.
See Mapping Tensors for the complete mapping model, Moving Tensors for memory movement, and Computing Tensors for the pipeline APIs.

## Stationary Vector Operand

### Elementwise Multiplication

#### Goal

Compute \(\text{out}[i] = \text{lhs}[i] \times \text{rhs}[i]\).
This pattern covers gate scaling, GLU, and attention scaling.

#### Mapping and Data Movement

The `sub` context preloads `rhs` into the VRF while the `main` context streams `lhs` and reads the VRF on every Vector Engine cycle.
The two DM regions must not overlap.

```rust
flowchart TB
    LHS_HBM[(lhs HBM)] -->|Tensor DMA| LHS_DM[(lhs DM)]
    RHS_HBM[(rhs HBM)] -->|Tensor DMA| RHS_DM[(rhs DM)]

    subgraph sub[sub context]
        direction LR
        sFE[Fetch] --> sCO[Collect] --> VRF[(VRF)]
    end

    subgraph main[main context]
        direction LR
        mFE[Fetch] --> mCO[Collect] --> VE["Vector MulInt"] --> CM[Commit]
    end

    RHS_DM --> sFE
    LHS_DM --> mFE
    VRF --> VE
    CM --> OUT_DM[(result DM)]
    OUT_DM -->|Tensor DMA| OUT_HBM[(HBM)]
```

Every device kernel has `device.main` and `device.sub` execution contexts on separate hardware resources.
`main` runs the primary computation while `sub` commonly prefetches operands.
`main` waits when it needs data that `sub` has not produced, and both contexts share the flat on-chip SRAM.

Omit DM addresses for automatic placement.
Use `_at` APIs only when an algorithm needs explicit non-overlapping addresses.
The `Tu` const generic identifies whether a tensor flows through `{ Tu::Main }` or `{ Tu::Sub }`.
The `sub` context loads `rhs_dm` through Fetch, Collect, and `.to_vrf()`, while `main` streams `lhs_dm` through `MulInt` with that stationary VRF operand.
The two contexts run concurrently when their resource dependencies allow it.

#### Device Source

Device source: `src/kernel/elementwise_mul_kernel.rs`.

```rust
use furiosa_opt_std::prelude::*;

axes![A = 2048];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![A / 8 # 256];

#[device(chip = 1)]
pub fn elementwise_mul_kernel(
    device: &mut Device,
    lhs: &HbmTensor<i32, Chip, m![A]>,
    rhs: &HbmTensor<i32, Chip, m![A]>,
) -> HbmTensor<i32, Chip, m![A]> {
    // Move both operands from HBM to DM (DM placement is assigned automatically).
    let lhs_dm = lhs.to_dm::<Cluster, Slice, m![A % 8]>(&mut device.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, Slice, m![A % 8]>(&mut device.tdma);

    // Sub context: load rhs into VRF (runs concurrently with the main context below).
    // VRF holds a per-slice operand that the Vector Engine reads every cycle.
    let rhs_vrf: VrfTensor<i32, Chip, Cluster, Slice, m![A % 8]> = device
        .sub
        .begin(rhs_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![A % 8 / 8], m![A % 8 % 8]>()
        .to_vrf();

    // Main context: multiply every lhs element by its rhs counterpart from VRF
    let result = device
        .main
        .begin(lhs_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // Each slice multiplies its 8 lhs elements by the matching 8 rhs elements in VRF
        .vector_fxp(FxpBinaryOp::MulInt, &rhs_vrf)
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>();

    result.to_hbm(&mut device.tdma)
}
```

#### Host Program and Oracle

Base-template host program and oracle: `src/elementwise_mul.rs`.

```rust
use furiosa_opt_std::prelude::*;
use {{ crate_name }}::kernel::elementwise_mul_kernel::{A, elementwise_mul_kernel};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut device = Device::new(elementwise_mul_kernel.topology())?;
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await?;
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await?;
    let _out_hbm = launch(elementwise_mul_kernel, (&mut device, &lhs_hbm, &rhs_hbm)).await?;
    println!("Elementwise Mul: kernel ran");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_reference() {
        let mut device = Device::new(elementwise_mul_kernel.topology()).unwrap();

        let mut rng = SmallRng::seed_from_u64(42);
        let lhs = HostTensor::<i32, m![A]>::rand(&mut rng);
        let rhs = HostTensor::<i32, m![A]>::rand(&mut rng);

        let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
        let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

        // Reference: out[i] = lhs[i] * rhs[i].
        let lhs_buf: Vec<i32> = lhs.into_vec();
        let rhs_buf: Vec<i32> = rhs.into_vec();
        let expected: Vec<i32> = lhs_buf.iter().zip(&rhs_buf).map(|(&a, &b)| a.wrapping_mul(b)).collect();

        let out_hbm = launch(elementwise_mul_kernel, (&mut device, &lhs_hbm, &rhs_hbm)).await.unwrap();

        let actual: Vec<i32> = out_hbm.to_host::<m![A]>(&mut device.pdma).await.unwrap().into_vec();
        for (i, (&e, &a)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(e, a, "elementwise_mul mismatch at i={i}: expected {e}, actual {a}");
        }
    }
}
```

## Contraction Patterns

### Dot Product

#### Goal

Compute \(\sum_i x_i y_i\).
This pattern covers attention scores and similarity.

#### Mapping and Data Movement

One operand flows through the pipeline while the `sub` context stores the other operand in the TRF.
The `sub` context loads that operand through Fetch, Collect, and `.to_trf()`.

`.contract_outer()` pairs 32-byte flits into 64-byte packets and reads the stationary TRF operand.
The Stream Adapter performs the pairing and the TRF Sequencer reads the stationary operand.
Both operands feed the per-lane multiplier.
`.contract_packet()` reduces products spatially through the hardware reduction tree.
`.contract_time::<m![1]>()` accumulates across time and produces one scalar per slice.
`.contract_lane()` folds the lanes, which is trivial when `Lane = m![1]`.
`.cast()` converts the `f32` accumulator to `bf16`.

Contraction performs multiply-and-accumulate over many terms and needs a widened type.
Contraction widens `i4` and `i8` to `i32`, and it widens `fp8` and `bf16` to `f32`.
The annotations on `.align()` and `.contract()` select that widened type.

#### Device Source

Device source: `src/kernel/dot_product_kernel.rs`.

```rust
use furiosa_opt_std::prelude::*;

axes![A = 2048];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![1 # 256]; // 1 active slice; m![A / 8 # 256] would distribute across all 256
pub type Time = m![1]; // No temporal iteration
pub type Lane = m![1]; // No lane parallelism

#[device(chip = 1)]
pub fn dot_product_kernel(
    device: &mut Device,
    lhs: &HbmTensor<bf16, Chip, m![A]>,
    rhs: &HbmTensor<bf16, Chip, m![A]>,
) -> HbmTensor<bf16, Chip, m![1]> {
    // HBM → DM
    let lhs: DmTensor<bf16, Chip, Cluster, Slice, m![A]> = lhs.to_dm(&mut device.tdma);
    let rhs: DmTensor<bf16, Chip, Cluster, Slice, m![A]> = rhs.to_dm(&mut device.tdma);

    // Sub context: load rhs into TRF
    let rhs: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![A]> = device
        .sub
        .begin(rhs.view())
        .fetch::<Time, m![A]>()
        .collect::<m![{ Time }, A / 16], m![A % 16]>()
        .to_trf();

    // Main context: stream lhs through the Contraction Engine, reduce along A
    let result: DmTensor<bf16, Chip, Cluster, Slice, m![1 # 8]> = device
        .main
        .begin(lhs.view())
        .fetch::<Time, m![A]>()
        .collect::<m![A / 16], m![A % 16]>()
        // Pair consecutive 32-byte flits into 64-byte packets, halving time steps (A/16 → A/32)
        .contract_outer::<m![A / 32], m![A % 32], _, _, _>(&rhs)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>() // cast f32 accumulator output back to bf16
        .commit_trim::<m![1 # 8]>()
        .commit();

    // DM → HBM
    result.to_hbm(&mut device.tdma)
}
```

#### Host Program and Oracle

Base-template host program and oracle: `src/dot_product.rs`.

```rust
use furiosa_opt_std::prelude::*;
use {{ crate_name }}::kernel::dot_product_kernel::{A, dot_product_kernel};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut device = Device::new(dot_product_kernel.topology())?;
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<bf16, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<bf16, m![A]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await?;
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await?;
    let _out_hbm = device.launch(dot_product_kernel, (&lhs_hbm, &rhs_hbm)).await?;
    println!("Dot Product: kernel ran");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_reference() {
        let mut device = Device::new(dot_product_kernel.topology()).unwrap();

        let mut rng = SmallRng::seed_from_u64(42);
        let lhs = HostTensor::<bf16, m![A]>::rand(&mut rng);
        let rhs = HostTensor::<bf16, m![A]>::rand(&mut rng);

        let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
        let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

        // Reference: sum_i lhs[i] * rhs[i] in f32, then round to bf16.
        let lhs_buf: Vec<bf16> = lhs.into_vec();
        let rhs_buf: Vec<bf16> = rhs.into_vec();
        let expected_f32: f32 = lhs_buf
            .iter()
            .zip(&rhs_buf)
            .map(|(&a, &b)| f32::from(a) * f32::from(b))
            .sum();
        let expected = bf16::from_f32(expected_f32);

        let out_hbm = device.launch(dot_product_kernel, (&lhs_hbm, &rhs_hbm)).await.unwrap();

        let actual_buf: Vec<bf16> = out_hbm.to_host::<m![1]>(&mut device.pdma).await.unwrap().into_vec();
        if let Some(&actual) = actual_buf.first() {
            let diff = (f32::from(actual) - f32::from(expected)).abs();
            let tol = (0.02 * f32::from(expected).abs()).max(0.5);
            assert!(
                diff <= tol,
                "dot_product mismatch: expected {expected:?}, actual {actual:?}, diff {diff} > tol {tol}"
            );
        }
    }
}
```

### GEMV

#### Goal

Compute \(y_i = \sum_j A_{ij} x_j\).
This pattern covers LLM decode.

#### Mapping and Data Movement

Map output dimension `I` across slices so each slice computes one row.
Broadcast the full vector `x` so every slice can contract it with its row.
Unlike dot product, the rows are independent but every row needs the same complete vector operand.

#### Device Source

Device source: `src/kernel/gemv_kernel.rs`.

```rust
use furiosa_opt_std::prelude::*;

axes![I = 256, J = 2048];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![I]; // Distribute output dimension across slices
pub type Time = m![J / 32]; // Temporal iterations for reduction dimension
pub type Packet = m![J % 32]; // Packet size for reduction dimension
pub type Lane = m![1];

#[device(chip = 1)]
pub fn gemv_kernel(
    device: &mut Device,
    matrix: &HbmTensor<bf16, Chip, m![I, J]>,
    vector: &HbmTensor<bf16, Chip, m![J]>,
) -> HbmTensor<bf16, Chip, m![I]> {
    // Move data from HBM to DM
    let matrix: DmTensor<bf16, Chip, Cluster, Slice, m![J]> = matrix.to_dm(&mut device.tdma);
    let vector: DmTensor<bf16, Chip, Cluster, Slice, m![J]> = vector.to_dm(&mut device.tdma);

    // Load vector into TRF
    let vector_trf: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![J]> = device
        .sub
        .begin(vector.view())
        .fetch::<m![1], m![J]>()
        // Collect Engine: split into 32-byte flits.
        .collect::<m![J / 16], m![J % 16]>()
        .to_trf();

    // Compute GEMV: matrix × vector
    // Key difference: `I` maps to slice (preserved), `J` gets reduced
    let result: DmTensor<bf16, Chip, Cluster, Slice, m![1 # 4]> = device
        .main
        .begin(matrix.view())
        .fetch::<m![J / 16], m![J % 16]>()
        .collect::<m![J / 16], m![J % 16]>()
        .contract_outer::<Time, Packet, _, _, _>(&vector_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 4]>()
        .commit();

    // Transfer result to HBM
    result.to_hbm(&mut device.tdma)
}
```

#### Host Program and Oracle

Base-template host program and oracle: `src/gemv.rs`.

```rust
use furiosa_opt_std::prelude::*;
use {{ crate_name }}::kernel::gemv_kernel::{I, J, gemv_kernel};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut device = Device::new(gemv_kernel.topology())?;
    let mut rng = SmallRng::seed_from_u64(42);
    let matrix = HostTensor::<bf16, m![I, J]>::rand(&mut rng);
    let vector = HostTensor::<bf16, m![J]>::rand(&mut rng);
    let matrix_hbm = matrix.to_hbm(&mut device.pdma).await?;
    let vector_hbm = vector.to_hbm(&mut device.pdma).await?;
    let _out_hbm = launch(gemv_kernel, (&mut device, &matrix_hbm, &vector_hbm)).await?;
    println!("GEMV: kernel ran");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_reference() {
        let mut device = Device::new(gemv_kernel.topology()).unwrap();

        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = HostTensor::<bf16, m![I, J]>::rand(&mut rng);
        let vector = HostTensor::<bf16, m![J]>::rand(&mut rng);

        let matrix_hbm = matrix.to_hbm(&mut device.pdma).await.unwrap();
        let vector_hbm = vector.to_hbm(&mut device.pdma).await.unwrap();

        // Reference: y[i] = sum_j matrix[i, j] * vector[j] in f32, rounded to bf16.
        let mat_buf: Vec<bf16> = matrix.into_vec();
        let vec_buf: Vec<bf16> = vector.into_vec();
        let expected: Vec<bf16> = mat_buf
            .chunks(J::SIZE)
            .map(|row| {
                let acc: f32 = row
                    .iter()
                    .zip(&vec_buf)
                    .map(|(&a, &b)| f32::from(a) * f32::from(b))
                    .sum();
                bf16::from_f32(acc)
            })
            .collect();

        let out_hbm = launch(gemv_kernel, (&mut device, &matrix_hbm, &vector_hbm)).await.unwrap();

        let actual: Vec<bf16> = out_hbm.to_host::<m![I]>(&mut device.pdma).await.unwrap().into_vec();
        for (i, (&e, &a)) in expected.iter().zip(&actual).enumerate() {
            let diff = (f32::from(a) - f32::from(e)).abs();
            let tol = (0.02 * f32::from(e).abs()).max(0.5);
            assert!(
                diff <= tol,
                "gemv mismatch at i={i}: expected {e:?}, actual {a:?}, diff {diff} > tol {tol}"
            );
        }
    }
}
```

### GEMM

#### Goal

Compute \(C_{ij} = \sum_k A_{ik} B_{kj}\).
`A` broadcasts across `J`, and `B` broadcasts across `I`.

#### Mapping and Data Movement

Map both output dimensions with `type Slice = m![I / 32, J / 32]` so each slice computes a 16 × 16 output tile.
The Switch Engine routes each `B` tile to its matching slice, so each slice receives only the `J` portion that belongs to its output tile.
`.contract_packet::<m![1]>()` reduces along `K` spatially.
`.contract_time::<m![I]>()` accumulates over time while preserving `I`.
`.contract_lane::<m![I], m![J # 8]>(LaneMode::Interleaved)` preserves `I` and `J` in the output packet.

#### Device Source

Device source: `src/kernel/gemm_kernel.rs`.

```rust
use furiosa_opt_std::prelude::*;

axes![I = 512, J = 512, K = 64];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
// Distribute output dimensions `I` and `J` across slices
pub type Slice = m![I / 32, J / 32]; // Each slice handles a 16 × 16 output tile
pub type Lane = m![J % 8];

#[device(chip = 1)]
pub fn gemm_kernel(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![I, K]>,
    b: &HbmTensor<bf16, Chip, m![J, K]>,
) -> HbmTensor<bf16, Chip, m![I, J]> {
    // Move data from HBM to DM
    let a: DmTensor<bf16, Chip, Cluster, Slice, m![I % 32, K]> = a.to_dm(&mut device.tdma);
    let b: DmTensor<bf16, Chip, Cluster, Slice, m![J % 32, K]> = b.to_dm(&mut device.tdma);

    // Load matrix B into TRF
    // Switch Engine distributes B across 256 slices
    // Each slice gets the full `K` dimension but only its (16 × 16) output tile
    // See: Switch Engine topologies for details on distribution
    let b_trf: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![J / 8 % 4, K]> = device
        .sub
        .begin(b.view())
        .fetch::<m![J % 8, J / 8 % 4], m![K]>()
        .collect::<m![J % 8, J / 8 % 4, K / 16], m![K % 16]>()
        .to_trf();

    // Compute GEMM: A × B
    // Switch Engine ensures matching (`I / 32`, `J / 32`) slice distribution
    // Contraction reduces along `K`, preserves `I` and `J`
    let result: DmTensor<bf16, Chip, Cluster, Slice, m![I % 32, J % 32]> = device
        .main
        .begin(a.view())
        .fetch::<m![I % 32, J / 8 % 4], m![K]>()
        .collect::<m![I % 32, J / 8 % 4, K / 16], m![K % 16]>()
        .contract_outer::<m![I % 32, J / 8 % 4, K / 32], m![K % 32], _, _, _>(&b_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![I % 32, J / 8 % 4]>()
        .contract_lane::<m![I % 32, J / 8 % 4], m![J % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![J % 8 # 16]>()
        .commit_trim::<m![J % 8]>()
        .commit();

    // Transfer result to HBM
    result.to_hbm(&mut device.tdma)
}
```

#### Host Program and Oracle

Base-template host program and oracle: `src/gemm.rs`.

```rust
use furiosa_opt_std::prelude::*;
use {{ crate_name }}::kernel::gemm_kernel::{I, J, K, gemm_kernel};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut device = Device::new(gemm_kernel.topology())?;
    let mut rng = SmallRng::seed_from_u64(42);
    let a = HostTensor::<bf16, m![I, K]>::rand(&mut rng);
    let b = HostTensor::<bf16, m![J, K]>::rand(&mut rng);
    let a_hbm = a.to_hbm(&mut device.pdma).await?;
    let b_hbm = b.to_hbm(&mut device.pdma).await?;
    let _out_hbm = launch(gemm_kernel, (&mut device, &a_hbm, &b_hbm)).await?;
    println!("GEMM: kernel ran");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_reference() {
        let mut device = Device::new(gemm_kernel.topology()).unwrap();

        let mut rng = SmallRng::seed_from_u64(42);
        let a = HostTensor::<bf16, m![I, K]>::rand(&mut rng);
        let b = HostTensor::<bf16, m![J, K]>::rand(&mut rng);

        let a_hbm = a.to_hbm(&mut device.pdma).await.unwrap();
        let b_hbm = b.to_hbm(&mut device.pdma).await.unwrap();

        // Reference: C[i, j] = sum_k A[i, k] * B[j, k] in f32, rounded to bf16.
        let a_buf: Vec<bf16> = a.into_vec();
        let b_buf: Vec<bf16> = b.into_vec();
        let expected: Vec<bf16> = a_buf
            .chunks(K::SIZE)
            .flat_map(|a_row| {
                b_buf.chunks(K::SIZE).map(move |b_row| {
                    let acc: f32 = a_row
                        .iter()
                        .zip(b_row)
                        .map(|(&a, &b)| f32::from(a) * f32::from(b))
                        .sum();
                    bf16::from_f32(acc)
                })
            })
            .collect();

        let out_hbm = launch(gemm_kernel, (&mut device, &a_hbm, &b_hbm)).await.unwrap();

        let actual: Vec<bf16> = out_hbm.to_host::<m![I, J]>(&mut device.pdma).await.unwrap().into_vec();
        for (idx, (&e, &av)) in expected.iter().zip(&actual).enumerate() {
            let diff = (f32::from(av) - f32::from(e)).abs();
            let tol = (0.05 * f32::from(e).abs()).max(1.0);
            assert!(diff <= tol, "gemm mismatch at idx={idx}: expected {e:?}, actual {av:?}");
        }
    }
}
```

Blocked GEMM extends GEMM with tiling for matrices that exceed on-chip DM capacity.
It covers temporal partitioning over `K` and spatial partitioning of `I` and `J` across chips.
Flash Attention combines GEMM, Vector Engine softmax, and main/sub prefetch across a transformer attention head.
