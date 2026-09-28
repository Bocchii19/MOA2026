# furiosa-opt Book — local mirror

Source: <https://developer.furiosa.ai/furiosa-opt/book/> — fetched 2026-09-13 from `print.html`.
Single-file copy: [`_full.md`](_full.md). Chapters below are split from it (code fences preserved).

| # | Chapter | File | Lines | Sections |
|---|---|---|---|---|
| 1 | Introduction | [`01-introduction.md`](01-introduction.md) | 37 | Read This Book by Task; License |
| 2 | Quick Start | [`02-quick-start.md`](02-quick-start.md) | 37 | Read the ordered child pages; Continue to Reference Chapters; Choose the next pattern |
| 3 | Setup and Tooling | [`03-setup-and-tooling.md`](03-setup-and-tooling.md) | 107 | Installation and First Program; The Generated Project |
| 4 | Tensor and Contraction | [`04-tensor-and-contraction.md`](04-tensor-and-contraction.md) | 26 | — |
| 5 | Kernel Design | [`05-kernel-design.md`](05-kernel-design.md) | 736 | First Vector Kernel; Stationary Vector Operand; Contraction Patterns |
| 6 | Kernel Validation | [`06-kernel-validation.md`](06-kernel-validation.md) | 15 | — |
| 7 | Mapping Tensors | [`07-mapping-tensors.md`](07-mapping-tensors.md) | 159 | Place and Reduce Dimensions; Mapping Order and Performance; Position Cell Summary; Representation and Distribution |
| 8 | Mapping Expressions | [`08-mapping-expressions.md`](08-mapping-expressions.md) | 661 | Axis Sizes; Position Results; M Constructor Rules; Constructors; Equivalent Mappings |
| 9 | Spatial and Temporal Dimensions | [`09-spatial-and-temporal-dimensions.md`](09-spatial-and-temporal-dimensions.md) | 183 | Spatial Dimensions; Temporal Dimension; NCHW Representation Trace |
| 10 | Tensor Semantics | [`10-tensor-semantics.md`](10-tensor-semantics.md) | 102 | Tensor Holding Semantics; Linear Combination Semantics; Function Specification |
| 11 | Moving Tensors | [`11-moving-tensors.md`](11-moving-tensors.md) | 188 | Choose a movement route; Copyable end-to-end transfer; Keep transfers ordered; Engine boundaries; Tune movement; See also |
| 12 | Sequencer | [`12-sequencer.md`](12-sequencer.md) | 662 | Interface; Examples; Architecture; Configurations; Constraints |
| 13 | Fetch Engine | [`13-fetch-engine.md`](13-fetch-engine.md) | 429 | Interface; Axis Lifting; Constraints; Multi-Read Packet; Interleaving; Optimizations |
| 14 | Commit Engine | [`14-commit-engine.md`](14-commit-engine.md) | 153 | Interface; Constraints; Multi-Write Packet; Optimizations |
| 15 | Case Study: Tensor Unit I/O | [`15-case-study-tensor-unit-i-o.md`](15-case-study-tensor-unit-i-o.md) | 85 | — |
| 16 | DMA Engine | [`16-dma-engine.md`](16-dma-engine.md) | 920 | Interface; Architecture; Constraints; Optimizations; Detailed Examples; Redistribution Operations; Scatter and Gather; PCIe DMA |
| 17 | Memory Performance | [`17-memory-performance.md`](17-memory-performance.md) | 246 | Data Memory (DM); High-Bandwidth Memory (HBM) |
| 18 | Computing Tensors | [`18-computing-tensors.md`](18-computing-tensors.md) | 101 | Selecting a Compute Route; Tensor Unit; Execution Context |
| 19 | Fetch Adapter | [`19-fetch-adapter.md`](19-fetch-adapter.md) | 253 | Table Lookup; Type Casting; Zero-Point Subtraction |
| 20 | Switch Engine | [`20-switch-engine.md`](20-switch-engine.md) | 656 | Interface; Regular Configurations; Architecture; Performance; Custom Configurations |
| 21 | Collect Engine | [`21-collect-engine.md`](21-collect-engine.md) | 219 | Interface; Examples; Register File Loading |
| 22 | Register Files | [`22-register-files.md`](22-register-files.md) | 309 | Tensor Register File; Vector Register File |
| 23 | Contraction Engine | [`23-contraction-engine.md`](23-contraction-engine.md) | 240 | Architecture; Example: Batched MatMul |
| 24 | Outer | [`24-outer.md`](24-outer.md) | 312 | Interface; Stream Adapter; TRF Sequencer; Multiplier |
| 25 | Packet Reducer | [`25-packet-reducer.md`](25-packet-reducer.md) | 104 | Interface; Architecture; Performance |
| 26 | Time Reducer | [`26-time-reducer.md`](26-time-reducer.md) | 133 | Interface; Architecture; Constraints; Performance |
| 27 | Lane Folder | [`27-lane-folder.md`](27-lane-folder.md) | 152 | Interface; Constraints; Performance |
| 28 | Vector Engine | [`28-vector-engine.md`](28-vector-engine.md) | 254 | Interface; Examples |
| 29 | Intra-Slice Chain | [`29-intra-slice-chain.md`](29-intra-slice-chain.md) | 998 | Interface; Stages; Reinterpret; Examples; Performance |
| 30 | Intra-Slice Reduce | [`30-intra-slice-reduce.md`](30-intra-slice-reduce.md) | 229 | Examples; Architecture; Performance |
| 31 | Valid Count Generator | [`31-valid-count-generator.md`](31-valid-count-generator.md) | 931 | Architecture; Putting It All Together; Inexpressible Patterns; Constraints; Downstream 4-Way Operations |
| 32 | Inter-Slice Reducer | [`32-inter-slice-reducer.md`](32-inter-slice-reducer.md) | 207 | Interface; Constraints; Examples; Performance |
| 33 | Cast Engine | [`33-cast-engine.md`](33-cast-engine.md) | 98 | Interface; Supported Casts; Performance |
| 34 | Transpose Engine | [`34-transpose-engine.md`](34-transpose-engine.md) | 395 | Interface; Architecture; Examples; Performance |
| 35 | Commit Adapter | [`35-commit-adapter.md`](35-commit-adapter.md) | 193 | Trimming; Type Casting |
| 36 | Case Study: Chip and Cluster Reduction | [`36-case-study-chip-and-cluster-reduction.md`](36-case-study-chip-and-cluster-reduction.md) | 360 | Shape Transformations; Redistribution Primitives; Chip Examples; Cluster Examples; DMA Layout Requirements |
| 37 | End-to-End Cases | [`37-end-to-end-cases.md`](37-end-to-end-cases.md) | 17 | Choose a Starting Point |
| 38 | Case Study: Transformer | [`38-case-study-transformer.md`](38-case-study-transformer.md) | 150 | Runnable decoder flow; Workload semantics; Decision trace; Host-prepared embedding; Decoder step; Final layer and padded logits; Verify and inspect |
| 39 | Scheduling and Tuning | [`39-scheduling-and-tuning.md`](39-scheduling-and-tuning.md) | 18 | — |
| 40 | Schedule | [`40-schedule.md`](40-schedule.md) | 102 | Execution contexts; Resource and bank contention; Operation order and dependencies |
| 41 | Diagnosis | [`41-diagnosis.md`](41-diagnosis.md) | 31 | Inspect a dumped schedule; Read the timeline; Keep the comparison valid |
| 42 | Tuning | [`42-tuning.md`](42-tuning.md) | 306 | Control the experiment; Choose one lever; Unroll a loop; Double buffering |
| 43 | Tools | [`43-tools.md`](43-tools.md) | 8 | — |
| 44 | Kernel Optimizer | [`44-kernel-optimizer.md`](44-kernel-optimizer.md) | 160 | First Commands; Usage; What the wrapper adds; Automatic kernel builds; Direct compilation: cargo furiosa-opt compile; Environment variables |
| 45 | Language Server | [`45-language-server.md`](45-language-server.md) | 93 | Installation; Environment variables; Features; Caveats |
| 46 | Schedule Viewer | [`46-schedule-viewer.md`](46-schedule-viewer.md) | 80 | Review a Schedule; Getting Started; Usage |
