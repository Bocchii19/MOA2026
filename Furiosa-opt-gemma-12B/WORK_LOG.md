# ver16 on SDK 0.8.1 — measurement log (2026-09-24)

Baseline = ver16 as received, rebuilt from scratch here. Nine independently-seeded runs per
kernel in one Arena job, median reported (the harness the graders use takes a median too).

| | K1 | K2 | K3 | score |
| :-- | --: | --: | --: | --: |
| **baseline (9-run median)** | **74,463** | **38,682** | **216,871** | **8.4388** |
| baseline (9-run min) | 73,274 | 37,981 | 215,272 | 8.5572 |
| official gradings of this source | | | | 8.3384 – 8.4611 |

Leaderboard for reference (2026-09-24): rank 1 `IA -ARIA ON THE PLANETES` 9.383 with
70,158 / 32,708 / 198,051; rank 2 `PN` 9.152 with 68,088 / 36,114 / 199,156.

## What the hardware actually does

Every kernel is **DMA-bound**: the union of DMA spans is 91–97 % of the window (K1 68,996/74,065,
K2 35,836/38,319, K3 210,189/216,746). Compute never binds — `Renegade::*` and the `Cluster`
spans are sync barriers and hidden passes.

**Big aligned weight streams all run at 575–610 B/cycle, chip-wide, and nothing moves that.**
Measured: K3 down weight 608, K3 up 598, K3 gate 549, K1 q 580, K1 k/v 573, K2 tile24 576. The
compiler's own model assumes ~1,150–1,225 B/cycle, i.e. it is optimistic by almost exactly 2x.

Byte floors at 600 B/cycle: K1 52.4k, K2 26.2k, K3 165.9k (score 11.8 if everything else were
free). Our overheads over that floor: K1 22.1k, K2 12.5k, K3 51.0k. Rank 1's: 17.8k, 6.5k, 32.2k.
**The whole gap to the top is prologue/tail and non-stream DMA, not stream throughput.**

## Tools built (reusable)

- `cargo furiosa-opt compile --exact <fn> --dump-schedule out.json` — per-instruction static
  schedule with lifetimes, engine and byte counts. `scripts/…/sched.py`, `stat.sh`, `try.sh` in the
  session scratchpad turn one idea into a 40-second local check.
- `--dump-summary DIR` — `summary.log` gives io/main/sub occupancy and `dram_usage_io`.
- A 9-run measurement overlay (`const RUNS = 9` + `RUNS = 9` in `generate_references.py`, and a
  `min`/`p25` print) removes the +-4k run-to-run noise that made 3-run comparisons useless.
- `FURIOSA_OPT_PROFILE=trace` plus a `name=` field in the harness's span collector dumps every
  hardware span (`SPAN <test> cl= beg= end= dur= name=`), which is how the DMA timeline above was
  measured.

## Closed — measured worse, do not retry

All against the 9-run baseline unless marked (3-run).

| change | K3 | verdict |
| :-- | --: | :-- |
| Up/Gate weight as one contiguous 57,600 B run per slice (flat axis) | +7.7k (3-run) | worse |
| Down weight + scale merged into one 15-row tile (29 -> 25 DMA commands) | +8,464 | worse |
| Down scale merged 4+8+3 -> one 15-row load | +8.8k (with gate-q4) | worse |
| Gate scale switched to the q4 seed (static makespan -712) | +4,955 | worse |
| q8 seed: 240-row leaders, every piece 256 B aligned, ring-8 InterTranspose | +10,968 | worse |
| q8 with the redistribution on Sub instead of Main | static 243k | worse |
| GeGLU handed to Down by DM-to-DM instead of the HBM round trip | static +7.5k | worse |
| Down regrouped 16 groups x 8 (ring-8 x broadcast, half the activation replication) | — | rejected by the compiler: "large continuous flits (8) which are not broadcasted to next slice will cause TensorUnit hang" |
| K2: Up/Gate weight tiled 24+6, 18+12, 12+18, 6+24 | — | the tile view is not a legal DMA shape in 0.8.1 |
| K2: weight tiles 26+4, 28+2, 20+10, 22+8, 24+4+2, 20+8+2, 16+8+4+2 | — | 24+6 is the best of all of them |
| K2: dropping two of the three epilogue operand DMAs | +219 (noise) | those loads are already hidden; they cost nothing |
| K2: contracting against `hi` only (drops the second weight read) | static +487 | the `Dummy2` double read is free |

**The static makespan is anti-correlated with hardware for K3.** Four changes that lowered it
(gate-q4 -712, down-scale-15 -1,097, q8 -1,318, the combination -819) each cost 5–11k on hardware.
Use it only to reject ideas, never to accept them.

## Why ver16's K3 resists tuning

Each of the failures above moves work in one of two directions the design is already balanced
against: merging DMA commands serializes the first consumer behind the whole load (the Down scale
is split 4+8+3 precisely so group 0 starts after four rows), and any on-chip redistribution that
buys alignment lands on Main, which already carries 63k of the 111k static makespan. The one
direction that would help — removing work — has no candidates left: the 99.53 MB of weights and
scales are all live data, and the only replicated read (the Down activation, 1.97 MB for 30 KB of
GeGLU output) cannot be shrunk without a slice-axis reorder that the transpose engine rejects.

## The decisive measurement: DMA volume is not the lever

An ablation build where Gate borrows Up's already-loaded scale tensor removes **one whole 3.69 MB
scale load** (4,880 static DMA cycles, one fewer command). Hardware: K3 216,871 -> **214,193, only
-2,678**. So 4,880 static cycles of DMA buy 2,678 real ones, and the Up/Gate scales -- which look
like the worst-aligned load in the kernel -- are already mostly overlapped.

That caps what any DMA trimming can return. K3's entire non-weight DMA is 31,538 static cycles, so
removing *all* of it (impossible: it is live data) would return about 17k. The same ratio applied to
the other two:

| | non-weight static DMA | best case if all of it vanished |
| :-- | --: | --: |
| K1 | 7,178 | ~70.5k (from 74.5k) |
| K2 | 3,845 | ~36.7k (from 38.7k) |
| K3 | 31,538 | ~199k — but only ~211k for a realistic third of it |

**A perfectly trimmed version of this architecture lands near 8.8, not 9.4.** Rank 1's 9.383
(70,158 / 32,708 / 198,051) is not reachable by tuning ver16; it needs a different decomposition.

Also measured and worth knowing: the FP4 decode table is stored four times (`Renegade::StoTab`,
5,379 cycles each, 23,132 total) because the table lives in one Main Fetch Unit register and the
four decodes are far apart. They run concurrently with DMA, so they are not on the critical path —
checked against the span timeline, not assumed.

## Marginal costs, measured by ablation

Three deliberately-wrong builds price the two engines against each other. All against the 9-run
baseline (K3 216,871).

| ablation | static change | K3 hardware | real cycles per static cycle |
| :-- | --: | --: | --: |
| Gate borrows Up's loaded **weight** (one 29.49 MB stream gone) | DMA -24,612 | 198,730 (**-18,141**) | 0.74 per DMA cycle |
| Gate borrows Up's loaded **scale** (one 3.69 MB load gone) | DMA -4,880 | 214,193 (**-2,678**) | 0.55 per DMA cycle |
| q8 octet scale seed (ring-8 InterTranspose) | Main +22,823 | 227,839 (**+10,968**) | 0.48 per Main cycle |

So **Main and DMA cost about the same per static cycle** and both are partly on the critical path;
neither engine is free to trade into. That is why moving the ring-256 `x` broadcast off Main and
onto the queue (Main -8,455, DMA +18,653) is a clear loss, and why every command merge that bought
DMA cycles at the price of serialising Main lost as well.

The bound this sets is blunt: deleting an **entire third of K3's weight bytes** only reaches
198,730, which is rank 1's K3. Their 198,051 is therefore not a scheduling win over ver16 — it is a
different amount of work, and no arrangement of these passes will find 18k.

## Two more closed

- **K2's 24-copy `delay_output_input` chain is worth its cost.** It is 7,216 of K2's 14,242 static
  Main cycles and does nothing to any value, but cutting it to four copies measured K2 39,987
  against 38,682 (**+1,305**). The static makespan only moves 79 either way, so the chain's payoff
  is the hardware issue order it buys, exactly as its comment claims.
- **Moving the ring-256 `x` broadcast off Main and through HBM** (Main -8,455, DMA +18,653, static
  makespan +27,722) is a large loss, which settles the direction question: neither engine has slack
  to absorb the other's work.
- The RoPE `dma_gather` alternative is already closed by ver16's own note (6.6k of DMA lane plus
  1-20k of sync against ~22 tiny Main passes that run under the weight loads).

## Where this leaves it

Fifteen structural changes measured, none an improvement. The design is at a local optimum that is
tight in both engines at once. Score stands at **8.4711 official** (rank 10), from 7.496 at the
start of the day. Reaching 9.x needs a K3 that does less work, not a K3 that is scheduled better --
see the ablation table above.

## Paired A/B, with a build that is actually per-worktree (2026-09-24)

Two earlier attempts at this table were wrong. Cross-job drift is real (unchanged K3 read 216.7k in
one job and 223.8k in another), so control and candidate now alternate A B A B inside one Arena job
(`scripts/opt/ab.sh`). But the first paired runs compared byte-identical binaries: every worktree
is the same cargo package, so after one built, cargo reported the others fresh and did not relink,
and the furiosa-opt kernel cache is keyed by kernel name only. `scripts/opt/build.sh` now drops the
package fingerprints, gives each worktree its own kernel out dir and serialises builds with flock;
`ab.sh` prints each arm's md5 and warns if they match. Control is md5 `e736fbc3` throughout.

| candidate | K1 | K2 | K3 | verdict |
| :-- | --: | --: | --: | :-- |
| PROBE (wrong on purpose): drop the norm-weight and residual tail loads | -360 | **-827** | +137 | the K2 tail operand loads are on the critical path |
| K2: norm/residual loads forced before the gather (they move to the kernel start) | +235 | +1,434 | -36 | worse |
| K2: all three operand loads forced before the gather | -246 | +2,330 | +84 | worse |
| K2: delay chain 24 -> 8 copies | +138 | +682 | +34 | worse |
| K2: weight tiles 28+2 | -28 | +2,348 | -360 | worse |
| K3: Down 4+4+4+3 tiles with the decode fused into each group | -390 | +18 | **+13,622** | worse |
| K3: Down as one 15-row load and one decode (one table store fewer) | +240 | +117 | +6,884 | worse |
| K3: Up/Gate weights tiled 18+12 (named `UpR` axis) | -798 | -450 | +10,822 | worse |
| K3: Gate scale on the quad seed | +340 | -210 | +5,586 | worse |
| K3: Down scale in one 15-row DMA | +464 | -423 | +4,354 | worse |
| K3: Up scale on the plain seed | +118 | -503 | +944 | worse |
| K3: Up scale passes 8+8+8+6 | -511 | -90 | +376 | neutral |
| K3: Up+Gate in one pass via `to_dm_view` into two halves of one tensor | | | about +110k | the half-view DMA runs at half rate |

The single-job numbers measured yesterday with a correct per-tag build were mostly right; the
"drift correction" that turned them into neutrals was the error. Every rearrangement of ver16's
K2 and K3 tried so far is worse, and K1/K2 medians still wander by up to ~800 between identical
kernels inside one job, so K2 effects under ~800 need a repeat before they count.

What the timeline says about K3's inter-stream gaps: each weight stream is issued only after the
previous contraction is dispatched (Up weight lands 53.4k -> Up table store 55.0-60.4k -> Up
contraction 60.4k -> Gate weight DMA 60.6k; the same Gate -> Down), because the static schedule lists
the next weight DMA after that contraction. Neither a delay chain on x (20 copies moved nothing) nor
a zero dependency on the scale loads (they then move to the front and drag the weights with them)
changed that order.

The host CPU backend cannot check ver16 numerics: K1/K2 panic in furiosa-mapping, and K3 fails on
the emulator while passing on hardware.

## How each DMA is split across the 8 engines (2026-09-24)

`cargo furiosa-opt compile --exact <fn> --dump-summary DIR` writes
`DIR/text_form/resourcelir.desc.json`, whose `DmaCommand` operators list one `srcN/dstN` descriptor
pair per DMA engine with the loop order and strides (`Axis=count:stride`, outermost first, last entry
= packet bytes) and the destination (cluster, pe, dmn). `scripts/opt/desc.sh <wt> <ops::fn>` prints
this for every DMA. It is the only dump that shows packet sizes and engine use.

The book (ch. 16/17): 8 DMA engines per chip, one per DMN pair, each up to 256 B/cycle; ~500 cycles
startup per command; HBM 1.5 TB/s; unaligned HBM reads 2x, partial writes read-modify-write; missing
HBM stack-bit (address bit 8) interleaving halves bandwidth.

What it shows for ver16:
- All big weight loads (K1 q/k/v, K2 tiles, K3 up/gate/down) already use all 8 engines with 256-B
  packets and DMN interleaving. K2's tile loops never toggle address bit 8 in their inner loops, but
  K1/K3 loads that do are no faster (all ~580-610 B/cycle chip-wide), so the ceiling is the DM write
  path (~256 B/cycle per cluster), not the stack bit.
- K2 small transfers are poor: x load 32-B packets (256 per cluster); the three epilogue operand loads
  1 engine each, 240-B misaligned packets, cluster 0 only; gather 120-B packets; final store 1 engine,
  240-B partial writes.
- `ONLY=<test> RUNS=21 ab.sh ...` measures one kernel with 42 samples per arm; K2 A/A read +26.

## K2: two confirmed wins, stacked (2026-09-24)

Four parallel directions, each measured with paired A/B (`ONLY=sliding_attention_output RUNS=21`,
3 rounds, 63 samples per arm), each claimed win re-measured by a separate agent on fresh worktrees.

| change | K2 B-A | verdict |
| :-- | --: | :-- |
| **D: weight tiles and x shards grouped by (Qs bit 11, Qs bit 8)**, so the tile DMA's inner loop toggles HBM address bit 8 (the stack bit); tile24 DMA 22.7k -> 20.8k on the span trace (554 -> 605 B/cycle) | **-1,497**, verify **-1,380** | adopted |
| **B: norm weight streamed by the last epilogue pass, y staged in VRF, residual VRF pass issued right after the gather** (the norm VRF pass and its DMA wait leave the tail; the DRAM-reuse sync is released at ~34k instead of just before the store) | **-830**, verify **-799** | adopted |
| D + B stacked vs control (job 84160) | **-1,697** (38,527 -> 36,830; rounds -1,517 / -2,005) | committed |
| D + B vs D alone (job 84190) | -404 (rounds -619 / -371 / -399) | B still adds on top of D |
| A: x replicated as aligned 256-B pieces + ring-32 Broadcast1 (x DMA 1.4k, switch 0.9k vs 2.4k) | +258 | neutral: tile24 still starts at ~1.85k |
| A + gather buffer from Sub copies (moves the start-of-kernel cluster_sync) | +335 | worse (Sub copies 0.7-1.3k on hardware) |
| C: epilogue restructures (four variants) | +1.5k to +3.8k | worse |
| D + tail restructure (T1f) | -215 | noise |

What the agents established about K2 (reusable elsewhere):
- The stack bit matters for K2's tiles after all: K1/K3 already toggled it, which is why they did not
  look slower. Check every big stream's inner loop strides with `desc.sh`.
- The prologue floor is the core's instruction stream, not the x load: x DMA issue (~100) + ~19
  setup instructions (~440) + a start-of-kernel `cluster_sync` (~1.34k) caused by the cross-cluster
  gather's scratchpad destination. Tile24 starts at ~1.85k however fast x lands.
- The DMA queue behaves FIFO. Any TU pass listed before a DMA delays its issue by ~450 cycles.
- The tail is bounded by three cross-cluster syncs: gather completion (~1.4-2.3k), DRAM reuse before
  the in-place residual store, and end-of-kernel (~0.7-0.85k). Further tail gains need fewer syncs.
- `weight_scale` cannot be folded into the projection row layout: HBM row-group stride 60 B and a
  30-bf16 DM run are both illegal (8-B alignment).
- Sub VRF passes overlapping a DMA into the same DMN take ~1.4k instead of 0.45-0.7k.

## K1: two small changes that only pay together (2026-09-24)

Four parallel directions (prologue + K->Q gap, Q->V transition, tail, stream decomposition), each
with paired A/B (`ONLY=sliding_project_qkv RUNS=21`) and an independent re-measurement.

| change | K1 B-A | verdict |
| :-- | --: | :-- |
| A: `normalize_key_scaled` streams the k scale against the gathered K instead of loading it straight to VRF (5 passes instead of 6); side effect: rope_offset is issued during the K stream and the k rms weight before Q | -608 / -394 / -715, verify -283 | alone: small |
| C/D: V contraction with full 64-B packets (hi/lo `Gs` as a TRF-side time broadcast, weight fetched once); V contraction 3.2k -> 2.15k on hardware | -440 / -421 / -430, C -644 / -315 / -695, verify -380 | alone: small (the scheduler moves the k rms load in front of Q, delaying Q ~1.35k) |
| **A + V64** (A already puts the k rms load before Q, so V64's side effect costs nothing) | **-1,295**, fresh re-run **-1,314** (6/6 rounds) | **committed** |
| A + K,V 64-B packets | -911 | worse than A + V64 |
| A + Q,K,V 64-B packets | -1,198 | worse than A + V64 |
| small loads forced before Q by zero anchors | +1,385 | K->Q gap grows 4.15k |
| V split into row tiles (2+2, 3+1) | +723 / +1,027 | extra junction per DMA, small tiles stream slower |
| hi/lo in TRF lanes for Q/K/V | -90 | faster contractions, lost to a reshuffled K->Q gap |

Facts established:
- Small DMA commands cost ~0.76-0.89k each when queued behind another, 1.4-1.5k after an idle DMA;
  the time is conserved wherever they sit (FIFO). The only recoverable part is idle + first startup.
- DMAs are placed as late as possible relative to their consumer, and a TU pass tied with a DMA at
  the same static time is listed first; V is always listed right after the Q contraction.
- Contraction time on hardware ~1.4-1.7k fixed + ~1.9 cycles per Outer step: 64-B packets halve the
  Outer steps. Row tiles of 4 rows or fewer are dominated by the fixed part.
- The final 2.5k `Cluster` span is launch skew: cluster 1 starts 2-4k later and cluster 0 waits for it
  at the end-of-kernel sync that the compiler derives from the last output's `cluster_sync` (the v
  `dma_scatter`). `scripts/opt/clusters.py` prints each cluster's window and waits.
- `dma_scatter_unscaled` (SPM index) is `todo!()` in SDK 0.8.1.

## Cross-cluster syncs and the K1 tail (2026-09-24)

Four directions, paired A/B + independent re-measurement; then all three winners stacked and measured
together (job 85246, all three kernels, RUNS=7 x 2 rounds): K1 -526, K2 -549, K3 -914, score
8.6145 -> 8.6904 in the same job, 0 FAIL.

| change | measured | verdict |
| :-- | --: | :-- |
| K1 V5: one-pass ring gather with a 4x4 block transpose (`gather_value_b`) for V and K; V sqrt folded into the final pass (`normalize_value_scaled_iv`) | K1 -841 / -773, verify **-903** | adopted |
| K3 B2: Down result DM-to-DM straight to cluster 0's epilogue slices; residual*gate VRF from one Sub vector pass; norm weight a plain `to_vrf`; layer gate on the FMA multiplier (release2 no longer gated by a 4-pass Sub chain) | K3 -1,628 / -1,160 / -2,112, verify -1,407 (8/8 rounds negative) | adopted |
| K2 J1: the cross-cluster gather writes into a buffer produced by a Sub pass, so its readiness sync leaves the kernel start (tile24 starts at ~1.5k instead of ~1.9-2.0k) | K2 -202 / -178 (6/6 rounds) | adopted (small) |
| K1: remove the end-of-kernel sync | -- | impossible, see below |
| K2: remove the whole cross-cluster exchange (probe, wrong numerics) | +52 | the exchange costs nothing in graded mode |

Rules established:
- Every DMA write into DRAM gets a `cluster_sync` on the written tensor, whatever the API or source
  placement (to_hbm_view, dma_scatter, scratch to_hbm, PCIe copy, single-cluster source); a kernel
  with no DRAM write still ends with a dram_reuse pair. Release goes right after the covered DMA's
  wait, acquire at the kernel end. The early cluster waits max(0, S - (A - R)): skew minus the program
  time between the late cluster's last DRAM access and its end. In K1 the v scatter is last, so the
  reported time is S + the late cluster's work. The driver's verifier requires every output to be
  "main cluster synchronized"; `SKIP_SYNC_CHECK` only skips the check.
- **Span traces overstate cross-cluster sync costs**: the same K2 binary reads 36.8k with
  `FURIOSA_OPT_PROFILE=info` and 38.1k with `trace`. Use traces for structure, info-mode A/B for
  decisions.
- K3's ~169k "sync" is not a cross-cluster wait: the acquire completes 8 cycles after the Down scale0
  DMA lands. The Down phase is DMA-queue-bound from w0 landing to w1 landing: sto_s 2.2k, sto_d 2.0k,
  switch config 1.6k, scale0 3.5k, x_seed 5.8k, scale12+scale3 8.6k, w1 12.45k.
- K1: the Q->V transition is ~1.3k when the K chain finishes >= ~0.9k before the Q stream ends, and
  3.9-5.7k when it does not; removing the K transpose pass keeps the K chain early. The static order
  flips unpredictably when one Main pass is removed (its DMA model is ~2x optimistic).
- A 4x4 block transpose inside a Broadcast1 gather is legal and exact.

## Round 4: non-weight DMA (2026-09-24)

| change | measured | verdict |
| :-- | --: | :-- |
| K3 V1b: the up global scale `su` leaves the 64 GeGLU slices; the GeGLU stays unscaled by su (Down split scale 256 -> 1/16, which is inside the old design's range envelope for every global scale in [2048, 16384]) and su is applied exactly in the Down epilogue's sqrt pass, rms' = sqrt(ms' su^2 + eps) / su | K3 -1,895 / -1,149, verify -745 (8/9 rounds negative) | adopted |
| K3 V2b: GeGLU [hi, lo] gathered onto 4 slices per cluster (one per PE), each a 3,840 B run of whole 256-B units, so the HBM store uses 4 engines with aligned writes (sto_d 1.98k -> 1.23k) | K3 -456 / -901 (6/6 rounds) | adopted |
| K2 J4: x all-gather seed on a quarter of the slices (16 destinations per engine instead of 64, 4-flit packets over the same ring-256 Broadcast1, one compaction pass; delay chain 24 -> 21) | K2 -177 / -148 (6/6 rounds); probe ceiling for any cheaper x is -238 | adopted |
| stack of the three vs 5025214 | K3 -891 (3/3 rounds), K2 -230 | committed |
| K1 late cluster (lane-parallel hi/lo for Q/K/V, Sub half_ms) | +1,683 / -46 | no win: the late cluster has no idle gap |
| K3 x_seed quarters + Broadcast01 (no config DMA); per-PE gather | +1,301 / +3,810 | worse |

Facts:
- K3's DMA queue is saturated from ~100 to w1 landing (~201k): 142.9k weight, 57.6k non-weight service
  time. Only removing or shrinking DMA work pays; order inside the saturated span does not.
- Every `fetch_table_lookup` pass gets its own compiler-generated FP4 table DMA (8 B to all 256 slices,
  1.6-2.0k) plus a 5.4-7k StoTab; there is no API to share or hoist tables.
- The K3 post-FFN mean square is ~5e-7..2e-5, the size of EPS: the RMSNorm is not scale-invariant,
  so global scales must still reach the epilogue.
- A partially-live Broadcast1 ring (1 live slice of 4, padding inside the ring digit) is legal and
  exact on 0.8.1; dead slices' packets land in element padding.
- Up (@104) and Gate (@201) weight DMA descriptors are identical (256-B packets, stack bit toggled);
  the Gate stream's lower rate (549 vs 598 B/cycle) is contention with the Up contraction, not layout.
- K3's Up / Gate / Down scale loads use 960 / 800 / 480-B packets at non-256-aligned offsets (2x read
  cost); aligning them needs an on-chip redistribution the compiler prices at 4-5k per tile.

K2 tile split re-checked after the stack-bit change (jobs t26/t22, ONLY=K2 RUNS=21 x3): 26+4 (with
3 extra delay copies so the second tile's DMA is listed before the first contraction) **+914**,
22+8 **+279**. 24+6 remains the best split.

## Round 5: K3 aligned block scales -- closed (2026-09-24)

| change | K3 B-A | verdict |
| :-- | --: | :-- |
| Up/Gate 32 L-rows per slice, 30 live slices per DMN (aligned 7,680-B scale runs), two independent implementations | +14.1k / +10.5k / +17.6k / +8.9k / +9.2k | worse (all numerically exact) |
| Down scales read aligned (probe, wrong numerics: all three, or scale0 only) | +607 / +473 | no gain even when ideal |

Why (span traces):
- A weight stream's rate follows the number of live destination slices per DMN, not just bytes per
  engine: with 30 live of 32 the Up/Gate streams ran ~6% slower (61,440/57,600 = 32/30), wherever
  the dead slots sit.
- Aligned 7.7 KB per-slice scale runs reach only 432-462 B/cycle (Up -1.5k, Gate -2.5k), not ~600.
- 32 rows per slice adds 6.7% per-slice Main work, which makes the Gate-weight -> GeGLU chain
  critical (it had ~2.3k of slack before w0 lands).
- Any aligned layout needs a multiple of 16 rows per slice at 16-row-aligned positions, hence dead
  slices, hence the stream penalty. Padded L digits on the slice axis only lower if L is relabelled as
  its digits (`axes![UpD, UpJ, UpR]`) for every DMA that touches the layout.
- Down phase after the GeGLU store: the DMA queue (scale0, x_seed, scale12, scale3, table + w1
  ~32.9k) and the core issue chain (scale0 -> VRF -> x_seed -> TRF load 6k -> group0 -> VRF -> group1
  issue -> w1 issue) are tied within ~0.1k; scale0 is the only scale on both chains and gains ~0.4k
  when aligned in place.
- Resizing Up/Gate-phase buffers can move the Down x_seed buffer next to the first FP4 table in SRAM
  page 0, which adds a `raw.wait main_context` that holds scale12's issue (~5k gap). Check
  allocations (`resourcelir.lifetime.json`) after any K3 buffer change.
- A replicated tiny load to 256 slices per cluster costs ~8.6k more than one slice per DMN plus a
  single-source Broadcast1 spread.

## Redesign panel to beat rank 1 (K1 68,496 / K2 31,791 / K3 192,967) (2026-09-24)

Eight independent designs (K2 x3, K3 x3, K1 x2), each with a cost-model estimate, static prototypes
and up to two hardware probes, then one skeptical judge per kernel who re-derived every estimate from
measured costs. **No design credibly reaches rank 1 on any kernel.** Design docs:
`$OPT_WORK/design_*.md`, judge plans `$OPT_WORK/plan_k{1,2,3}.md` (session scratchpad).

| kernel | best judged design | judged cycles | decision |
| :-- | :-- | --: | :-- |
| K2 | k2c: 16 row groups x 16 column strips per cluster (G3), x split into hi/lo in two TRF lanes, 64-B contraction packets, inter-slice reduce over the 16 strips (output = exactly one epilogue slice), tiles 92+28 | ~35.7k | build (in progress) |
| K3 | Down split by K across clusters (GeGLU hand-off on chip), retiled Down | 213.5-215k | none: inside job-to-job spread |
| K1 | ver16 dataflow with a prefetched small-load junction / streamed q scale | 72.0-76.0k | none |

Measured in this round:
- K2 probe P2 (cluster 1 streams 6 of its 30 rows less, cluster 0 unchanged): -779. Everything the
  cluster coupling costs in graded mode is <= ~0.8k, so cluster-split redesigns of K2 are closed.
- K2 probe P1 (remove the late cluster's post-barrier StoVrf): -62 (the trace model predicted -0.7..-1.2k).
- K2 64-B tile6 contraction (k2a T1): +384. Rule found: the scheduler ties the weight_scale DMA with
  the last contraction only if that contraction is statically >= 616 cycles; otherwise weight_scale is
  listed after the gather and residual/norm wait for it. The gather is issued after the weight_scale
  wait, so a last contraction shorter than ~1.6k buys nothing.
- K3 H1 (intra-cluster DM-to-DM broadcast of the GeGLU into 64 seed slices): +23.8k -- a DM-to-DM
  broadcast runs at ~76 B/cycle and slows the concurrent decode. k3a P2 (on-chip Sub hand-off) traced
  -2.1k, inside the control spread.
- K3 floors on the current family (from traces): Main chain after the Gate weight lands >= ~203k;
  w0 landing + Down-phase Main work ~206.6k; DMA queue service + tail ~210k. Reaching 193k needs all
  three to drop 11-14k at once (block scales at stream rate on all 256 live slices, no hand-off DMA,
  ~-6k of Down Main work, Gate table off the chain); no expressible layout gives the first.
- K1 R (v scale before V, VRF built under V, scale-free tail): +4,125. Static reject rule for K1:
  any K-chain pass listed after the Q contraction, < ~1.1k static Main slack before V lands, or added
  static Main work between the Q contraction end and V landing -> worse on hardware.
- DM->HBM APIs take only the Tensor DMA context (no pdma stores); `to_dm` accepts only `Dma::Tensor`.

K2 G3 build (redesign panel winner, `src/device/sliding/projection_g3.rs`): 16 row groups x 16
column strips per cluster, each slice's weight piece 120 runs of exactly 256 B, x strip split into
hi/lo in two TRF lanes, 64-B contraction packets, inter-slice reduce over the 16 strips (the output
is one epilogue slice per row group, so the gather moves 32 x 480 B), tiles 92+28 with a 35-copy
delay chain, ver16's epilogue unchanged.
| variant | K2 B-A | verdict |
| :-- | --: | :-- |
| C1 (replicated x, 40 copies) | -180 | not enough: J ~1k, no stream-rate gain (305-310 B/cycle/cluster) |
| C1 (35 copies) | -138 | not enough |
| **C1 + C2 (quarter-live x seed, ring-16 Broadcast1{16,16} + compaction)** | **-879**, fresh re-run **-815** (6/6 rounds) | **committed** |
| T2 (eps in Clip after the all-gather, sqrt folded into the final pass) on the ver16 epilogue | +332 | worse |
C2's chain window is 31-32 copies (33 puts a gather wait before the residual DMA; >= 34 lists
weight_scale before contraction B). The x seed was the missing piece: replicated x landed at 1.73k
vs 1.33k.
G3 tuning (vs e92ae12, ONLY=K2 RUNS=21 x3): 31 delay copies -31, tiles 90+30 -10 -- both neutral; keep 32 copies and 92+28.

## SDK capability census (2026-09-24)

All 203 `#[primitive]`s of furiosa-opt-std 0.8.1 checked against our code and the remaining cost
items; each candidate compiled in our kernel context (reports: `$OPT_WORK/census_{primitives,schedule,fp4}.md`,
plan `census_plan.md`). **No lever pays on its own on today's kernels.**

Works but off the critical path:
- D1 (K3): Gate block scale applied by a second, diagonal bf16 contraction (partials bf16, scale in
  the TRF). Gate scale work 15.5k -> 8.1k on Main, numerics 9/9 PASS; the Gate-contraction -> GeGLU
  chain shrinks 6.8k but w0/w1 landing do not move (the DMA queue binds). Patch `census_fp4_d1.diff`.
- L1 (all): one-pass lo for the FP8 hi/lo split (begin_interleaved + unzip + BitAnd truncation +
  fp_zip AddF), -1 Main and -1 Sub pass per split; K1 in-job -89 (noise).

Measured worse or impossible:
- Version-chain ordering (K3 Down groups 0,3,1,2 so w1 is listed before group1): +4,830.
- fetch_cluster_lift cannot read the other cluster's live data (lifts need replicated sources);
  cluster_swap is the same DMA + sync as our gather; fetch_slice_lift adds a compiler DMA per lifted
  fetch; memset seeds move the gather readiness sync back to the kernel start; cluster/slice-tiled DMA
  destinations are not lowered; fetch_mask / vector_filter / commit_valid_count_pack are todo!();
  commit_cast no effect; K2 pair-mode residual +1,289 static (DramReuse release moves late).
- FP4: contract_outer rejects f4e2m1; fetch_table_lookup is Main-only; there is one table (4 KB
  constant, 8 B DMA per slice + 16 KB-per-slice StoTab at ~3 B/cycle/slice) per lookup pass and no
  API to share one; integer paths are i4->i5 / i8->i9 only; tiled views cannot be reshaped for
  overlapping aligned scale windows.
- Scheduling: 0.8.1 has no priority/fence/barrier/queue API; statement order, store order, operand
  order and drop() give byte-identical command streams; the only handles are version chains (writes
  into tiles of one DmTensor run in program order), data anchors, and rolled loops (full barriers,
  +2.4-2.9k static in K1). Rule R1: a big weight DMA is released when the contraction consuming the
  previous weight starts.

## Clock domains and the stream ceiling (2026-09-24)

A forum post (forums.furiosa.ai/t/.../469) measured the profile span counter -- what the harness and
grader report -- at ~2.0 GHz, while compiler schedule cycles are 1.0 GHz PE cycles. That is why the
static schedule looked "2x optimistic" all along (static makespan x2 ~= hardware for K1 and K3).
Our ~600 B/cycle streams are ~1.2 TB/s, 80% of the 1.5 TB/s HBM peak (750 B per measured cycle).

Stream-rate lab (three agents, ~20 hardware variants, reports `$OPT_WORK/stream_{bank,order,model}.md`):
- **Ceiling**: span = ~1.08k per command + bytes / ~647 B/cycle steady with 8 engines; best measured
  617-633 B/cycle. One cluster alone streams 359-382, two together ~615-633: the limit is shared on the
  HBM side. Nothing approached 700.
- No effect: bank/row locality (0 same-bank-different-row collisions 616 vs control 614), modelled
  channel balance (predictions made before two runs failed), HBM order (G3, strip-major, per-engine
  sequential, 128-KB co-read all within 1%), live slices per DMN 1-32, destinations per engine 2-64,
  bytes per slice 7.7-245 KB. Commands sharing engines are FIFO and never overlap: splitting a stream
  into 2/4 commands only adds ~1.1k each.
- Rate killers: HBM bit 8 fixed across an engine's innermost requests (-4%, -18% if bits 8-10 fixed),
  one DMN per engine (-23%), one stack per cluster (-37%), one stack per engine (-12..-16%),
  misaligned or 800-1,024-B packets (~350-390), a contraction reading DM during the stream (~-20%).
- K2 G3 and K3 streams already sit at the ceiling (609-628). **K1's whole-row layout runs 5-8% below
  it** (Q 604 / K 579 / V 583 service-based); K2-geometry reads of the same bytes reach ~630 in K1's
  timeline (probes with wrong numerics), worth ~2.5-3.2k of DMA service if K1 is rebuilt in a strip
  geometry.
- So rank 1's edge is lower overhead, not a faster stream (K2: 31.9k - 25.4k single-command stream
  leaves ~6.5k of non-stream time for them vs ~10.4k for us).

## K1 strip geometry (2026-09-24)

Two independent builds of K1 in a 15-strip geometry (each slice holds a 256-column strip of H for a
group of rows; the Inter-Slice Reducer sums the strips; slot 15 of every 16-slot row group is dead,
so 30/32 slices per DMN are live -- no fully-live aligned strip layout exists because 15 is odd).
Each DMA engine's inner loop then walks one whole 3,840-B row across the strip slices (HBM bit 8
toggling every 2 requests), which is what reaches the stream ceiling.

| build | K1 B-A | verdict |
| :-- | --: | :-- |
| a (`qkv_s.rs`, 518 lines) | -2,398 / -2,265, verify **-2,163** | equal to b |
| **b** (`qkv_s.rs`, 366 lines; Q-only variant -1,172 / -1,161) | -2,348 / -2,102 / -2,198, verify **-2,232** | **committed** (smaller) |
| a vs b head-to-head | b - a = -44 | noise |

Trace (b): K service 622 vs 584 B/cycle, Q 621 vs 601; Q stream ends 1.85k earlier; Q contraction
3.5k vs 5.1k; K-chain slack before Qc 3.0k vs 1.4k; tail 6.9k vs 7.4k. The dead slot is excluded by the
compiler's valid-count tagging of the padded strip digit: probes that pre-filled it with NaN passed
all runs. Rules learned: the old "no K-chain pass after the Q contraction" reject rule only held
while the Q stream was slow (3 K RoPE passes now land after Qc and fit); anchoring the K chain before
Qc (build a1) made the K contraction static-critical and cost 1.4k; reducing a padded slice digit is
supported but promoting a time digit into the freed slot is not; memset runs at ~7.5 B/cycle/slice;
hi/lo in TRF lanes vs time broadcast: no difference.

## K3 block scales, round 2: aligned scales with the stack bit kept (2026-09-25)

Corrected diagnosis of round 5: its 32-row relayout lost ~6% stream rate because the 61,440-B
per-slice weight run is an EVEN multiple of 256 (bit 8 never toggles), not because of dead slices.
With R rows per contiguous slice block, an odd-multiple weight run needs R = 2 mod 4 and aligned scale
runs need R = 0 mod 16, so both builds decoupled the row-to-slice maps.

| build | K3 B-A | verdict |
| :-- | --: | :-- |
| a: weights in row-pair-parity slices (3,840-B odd inner stride), scales as aligned 16-row blocks, ring-2 InterTranspose on Main hands each slice its rows' scales; D1 on Gate | -2,393 / -2,240, verify -3,007 | good, superseded by b |
| **b: 30 rows per slice, all 256 live, rows re-assigned (slice 8k+c of a cluster holds row pairs 240k+16g+2c+{0,1}, g=0..14); weight run still 57,600 B with a 3,840-B odd stride just inside the DMN loop; block scales loaded as aligned 16-row groups (3,840 B, 256-B packets, 8 engines) and handed over by ONE Main InterTranspose{16,1} shared by Up and Gate (two DMAs into tiles of one source); D1 (diagonal bf16 contraction) applies the Gate scale; results gathered ring-8 onto 32 slices x 240 rows for the GeGLU / split / store** | **-4,659 / -6,988, verify -4,750** (all rounds negative, 0 FAIL, max abs err 0.0078 vs 0.0117) | **committed** |
| b + D1 on Up too | -5,031 | same as b within noise, slightly worse numerics |

Trace (b): Up/Gate scale service 7.3k each (was 9.8-10.7k), weight streams unchanged (~47.5k);
the shared hand-over moves the Gate scale ahead of the Gate weight, so the Gate hand-over, bf16
conversion and TRF load run in the Up phase where Main has slack; Gate-chain slack before the Down
decode 4.7k. Trap to check on every K3 build: a buffer placed in the first FP4 table's SRAM slot
(address 131072) adds a `raw.wait main_context` on the w0 decode before the Down x TRF load.

## Round 7: K3 Down scales and K1 junctions -- no win (2026-09-25)

| change | B-A | verdict |
| :-- | --: | :-- |
| K3 Down scales aligned (4-row groups, 256-B packets) + ring-8 InterTranspose hand-over on Main | +1,529 | worse: the hand-over passes sit in front of the Down groups on the critical Main chain |
| same, hand-over on Sub | trace +10k | the Sub InterTranspose commits took 12.7k / 7.9k on hardware (4.9k static) |
| probe: aligned Down scales with no hand-over (wrong numerics) | -428 | ceiling for a free hand-over |
| K3 D1 on Down rows 0-11 (4-row filler + merged 11-row scale) | +1,451 | w1 lands 1.67k earlier, but the scale StoTrf (6.9k) runs beside group1/2 and slows them; g3 waits on its StoTab |
| K3 2-row VRF groups | +2,570 | a Down group pass costs ~2k fixed |
| K1 q norm in the normalize_key_scaled form | +129 | noise |
| K1 probe: V normalized with the k scale (no v scale load) | +838 | the extra k-scale consumer reorders the K->Q junction (Q issues 1.8k later) |

Facts:
- The aligned Down scale reads are faster (S0 ~2.8k vs 4.9k, S12 ~3.7k vs 6.3k), but the Down phase
  has THREE tied chains after the GeGLU store: the DMA queue (scale0, x_seed, scale12, scale3, table,
  w1), the core issue chain (scale0 -> VRF -> x_seed -> TRF -> group0 -> VRF -> group1 issue -> w1
  issue), and Sub after group1 (VRF 0.8k -> g3 StoTab 5.4-7.9k on hardware vs 1.3k static -> VRF
  0.7k). Speeding one moves the wait onto another. A structural change (w1 issued independently of
  group1, the GeGLU store earlier) is needed.
- K1/K3 contraction stall: K and V contractions take ~1.70k or ~2.50k, Q ~2.72k or ~3.52k -- always
  +800-806 cycles -- and the slow mode goes with a Sub pass running beside the contraction (control:
  K slow 16/16, V 14/16; with no Sub pass beside them K fast 4/16, V 6/16). No controllable trigger
  found yet; the V contraction on K1's tail is the slow case.
- K1 junction rule: each small load lands exactly when the next begins in the serial static DMA
  channel, and a TU op tied with a DMA is listed first, so the first consumer of each small load is
  always listed before the next load (the k-scale cold start behind pos). One cycle of later Q issue
  costs ~0.46 cycles end to end (the other cluster streams alone at its per-cluster cap meanwhile).

## Round 8: contraction start-up, K3 Down restructure, K2 G3 tail -- no win (2026-09-25)

- **Contraction "stall" is the default**: with nothing beside them, back-to-back strip contractions
  take the slow time 180/180 (V/K ~2,497, Q ~3,525); the ~800-count start-up stays with the Vector
  Engine, MulF, inter-slice reducer, lane mode or time broadcast removed, and does not scale with
  steps. The occasional fast case (~1,700) happens when the two clusters run the same contraction
  500-999 counts apart (fast 1/540 when within 100): a chip-level shared cost, not controllable
  from source. Design rule: a contraction pass costs ~670 fixed + ~800 start-up + ~2 per outer step,
  so fewer, larger contraction passes pay. Dropping the 1/16 MulF (EPS x 256): -52, noise.
- K3 Down restructure: D1 on Up (closes the w0-issue gap but its StoTrfs slow the Gate stream);
  Broadcast01 over 4 aligned quarter seeds instead of the ring-4 CustomBroadcast (no config DMA) +
  11-row scale load (w1 ~1k earlier, but g3 then waits on the Sub chain VRF2 -> table store 7.0k ->
  VRF3); together (dn12) -619 then -309 with mixed-sign rounds -- not accepted. After the GeGLU store
  the DMA queue, the core issue order, Sub and the Main decode (17.1k) are within ~1k of each other.
  Static rule: the w0 DMA always waits for the Gate D1 StoTrf (every DMA after the Gate table DMA ties
  with the table store, and ties list the TU pass first).
- K2 tail on G3: an in-place epilogue on both clusters' G3 output slices (partials only crossing)
  measured +3,839 (trace, in-job): the slices sit at stride 16 so the cross-row-group reduce needs a
  ring-256 Broadcast (2,985 vs 796 for the ring-32), and a both-cluster exchange doubles the DMA
  descriptors. Pair-mode final pass +156. Probe dropping the norm/residual loads -539 (ceiling of the
  operand-chain lever). 120-to-128-row relayouts for aligned stores do not lower. The current gather to
  32 adjacent slices + ring-32 reduce is the cheapest arrangement.


## K3 toward 190k: local GeGLU/Down hand-over (2026-09-26)

**190k not reached.** Final paired hardware measurements (`info` mode):
- K3 only, 31 distinct seeds: **208,552 -> 203,258 (-5,294, -2.54%)**, job 99368, all PASS.
- Full three-kernel harness, 9 seeds x 3 rounds: **208,574 -> 203,651 (-4,923, -2.36%)**,
  job 99310; all three kernels PASS. Final K3 max absolute error across 31 seeds 0.01172,
  every element within the unchanged combined absolute/relative tolerance.

Committed path: Down is split by K across clusters, so GeGLU stays on chip. Main performs
ring-8 replication plus stride-8 all-gather in 24-byte FP8 packets, then Sub loads TRF.
The first 12 Down rows use fused FP4 lookup/contraction -> BF16 block partials -> diagonal
BF16 scale contraction; the final 3-row tile stays separate to overlap its weight DMA.
Both clusters' H-sized partials are transferred to cluster 0 and added by one intra-slice
reduction. Up uses the same BF16 diagonal scale method as Gate. DMA commands 28 -> 23;
static makespan 107,226 -> 103,020. Only `src/device/shared/mlp.rs` changes executable code.

Ablations: local hand-over on Sub +5,163 vs baseline; on Main -4,502 / reverse-order -4,267;
one-pass epilogue reduction another -922. Up diagonal is nearly neutral in isolated K3
(-119, mixed round signs), but helps the full harness -988 / reverse-order -831, 6/6 negative
paired rounds (jobs 99348/99367), so it is retained. Keeping only the reduction variant
measured 203,696 in K3-only/31-seed mode and 205,152 in the full harness. Do not mix contexts.

Rejected: all 15 Down rows together (+18,501 baseline or +5,035 with local hand-over),
combined Up/Gate lookup (+12,708), 120-value hand-over shards (+4,380), direct padded
Broadcast01 (+7,103). Some full-packet alternatives fail SDK layout lowering. The remaining
gap is 13.3-13.7k; no correct measured variant reaches 190k.

Full evidence, all samples, checksums, per-round medians, rejected patches and reproduction:
[2026-09-26 K3 report](reports/2026-09-26-k3-190k/README.md).

## Round 9 (2026-09-26): K1 prologue win; K3 tables / Down prefetch, K2 x-order / per-cluster epilogue closed

| change | B-A | verdict |
| :-- | --: | :-- |
| **K1 prologue: rms weight and x loaded as quarter-strip seeds (60 x 128-B pieces per engine, 15 destinations) and spread by one ring-64 Broadcast1{4,16} Main pass each; the sum(0*w) anchor runs on the weight SEED (Sub) so the x DMA is issued 8 cycles after w lands and the K weight DMA before x lands** | -726 / -670, verify **-459** (9/9 rounds negative) | accepted (`r9_k1t.diff`) |
| K1 prologue, alternative implementation (seeds on row group 0, single padded H/256 # 64 digit; zero anchor removed) | -546 (one run; confirmation not completed) | head-to-head vs the accepted one pending |
| K1 x loaded before the rms weight | +404 | worse |
| K1 quarter seeds with the control head order / f32 no-op spread | +854 / +783 | the K norm/RoPE chain got listed after Qc |
| K3 FP4 tables: rule established -- `sto_tab` count = number of `fetch_table_lookup` pass INSTANCES, always (adjacent, separated by Main/Sub/DMA, same view twice, loops): no reuse exists; best reorder variant | -168 / -384 | not accepted |
| K3 Down scales aligned + handed over in the Gate phase (D1 TRF scale, 8 rows) | +2,120 | the 1.97 MB aligned scale DMA streamed at 433 B/cycle behind the Gate table and pushed w0 7.1k |
| K3 Down scale rows 0-3 aligned, loaded in the prologue, hand-over pre-w0 | -729 (mixed rounds) | pre-w0 DMA costs 1:1 (the FIFO is saturated from 0.1k to w0) |
| K2 tile A's DMA first, x behind it (order verified in the lowered stream) | +2,424 / V5 +750 | a Broadcast1 switch pass running beside a landing weight stream costs ~3.3k instead of ~0.5k |
| K2 per-cluster epilogue (strip-major output slices) | +4,753 | illegal as designed: the Inter-Slice Reducer must reduce the innermost slice digit, so row-group outputs sit at stride 16 in every legal layout; the cheapest cross-group sum at stride 16 is ~3k |
| K2 fused ms + r=32 reducer + folded sqrt | +4,308 | a 1-flit inter-slice reduce costs ~4.4-4.6k for r=32 |

Facts: a switch pass beside a landing weight stream is ~7x slower; a Sub to_vrf whose
`raw.wait sub_context` precedes the gather costs ~1k after its span; the static DMA-channel order is
neither program order nor pure critical path (it flipped x-first whenever the anchor chain ended
before x's static landing); table DMAs are enqueued where the scheduler's DFS from the kernel output
reaches the consuming pass.

External commits on opt-next (another session, `reports/2026-09-26-k3-190k`): 8c63dbd "keep GeGLU on
chip and fuse block scaling" claims K3 208.5k -> 203.3k (31 seeds, jobs 99246-99368); being
re-measured here before use. Competition facts (web research): the final grade is the LAST submission
before 2026-09-30 23:59 AoE; rank 1 is 63,704 / 31,966 / 190,026 (9.8995) as of 09-26; public code is
prohibited for provider and copier alike.

Verified here (2026-09-26): 8c63dbd (K-split Down, GeGLU on chip, diagonal BF16 scales) K3 208,941 -> 203,421,
**-5,520** (rounds -5,663 / -5,225 / -5,601), 0 FAIL, max abs err 0.0117 (job ab_v8k3b). K1 prologue
quarter seeds (r9_k1t) committed on top; head-to-head vs the alternative prologue build: -305 in its favour.
New K3 tail (trace log_v8k3_tr, cl0 run1): w1 lands 190.5k -> fused g3 5.2k (4 small DMAs overlap it) ->
Cluster wait 1.85k (cluster 1's partial transfer + sync) -> 6 epilogue passes 3.2k -> store 0.84k = 13.2k.

## Round 10 on the K-split K3 (2026-09-26): -4.9k from TU-context placement

| change | K3 B-A | verdict |
| :-- | --: | :-- |
| **M5: GeGLU hand-over TRF load on Main (1.3k vs 4.2k on Sub); Down FP8->BF16 scale conversion + scale TRF load on Main right after the decode (the Sub conversion in flight made the lowering emit a source-less `raw.wait sub_context` before the g3 table DMA, holding w1's issue 2.6k); config-free x spread (Broadcast1 ring-32 stride-8 replicate + ring-8 all-gather) removing the 16 KB switch-config DMA** | -4,854 / -4,985, verify **-4,932** (9/9 rounds) | **committed** |
| M2 (the first two changes only) | -4,291 / -3,415 | superseded by M5 |
| C1/X6: 'touch' passes to reorder the Down scale DMA before the decode | +5,538 / +4,545 | +542 static cycles on the w0 chain flipped the global list schedule (Up projected before Gate) |
| tail part 1: smaller last tile (13/2, 14/1, 11/4) | model: 12/3 optimal | the fused lookup+contraction runs at ~1.35k per 3,840-B row per slice (fetch/lookup-bound, same rate as the pure decode); the Sub chain after the scale lands binds |
| tail part 1: fused add / split partial push | -0.2..-0.4k modelled | probe with the whole transfer+sync+add chain removed: -1,658 = hard ceiling |
| tail part 2: 5 -> 4 epilogue passes (sqrt folded into the final pass via q^2 trick, gs^2 on the all-gather's free multipliers) | -209 / -341 (full harness -91) | not accepted; the acquire waits back to the same point |
| Down scales aligned in the K-split layout | +1,020 | geometrically void: a cluster's scale piece per row is a 480-B K-half at stride 960 B; only full rows align and that doubles the bytes |
| K1: V ring gather fused into the two norm passes | +1,411 | a ring-64 switch with 32-flit vector work costs ~1.7k: switch and vector work add, they do not overlap |

Rules: the lowering emits a bare `raw.wait sub_context` before every table DMA when Sub work is
outstanding; every Main op listed after a StoTab waits for it; x TRF load on Main 1.3k vs Sub 4.2k;
a Main pass beside a StoTab slows (Down decode 19.7k vs 17.1k); the K1 K->Q junction order in the
control is an allocation accident (Kc's output reuses the rope_offset buffer), so any change to K1's
tail tensor set re-packs DM and can flip it. K3 after M5: queue continuous from 0.1k to w1's landing
(~186.4k), tail ~12k (g3 5.2k + push/sync + 5 passes + store) -- both near this structure's floor.
Numerics robustness of 0abcde6 (K3): 45 seeded runs on Arena (`scripts/opt/sweep.sh`, job 100869), 45 PASS,
max abs err 0.01172 on every seed that reaches it (a bf16 ulp; inside the combined atol+rtol criterion),
K3 median 198,287 (min 197,184). Submitted as 16fa33ae (final grading uses the last submission).
Official grade of 0abcde6 (submission 16fa33ae, 2026-09-26): K1 68,963 / K2 35,920 / K3 197,954 = **9.1481**
(rank 6; ranks 3-5 at 9.1975 / 9.1695 / 9.1666; rank 1 9.8995).

## Round 12 (2026-09-26)

| change | B-A | verdict |
| :-- | --: | :-- |
| **K2: x seed on PE0 only (1 engine per cluster, 64 x 128 B), the ring-16 switch writes into a buffer cleared by one Sub memset whose static length outlasts the x DMA (so tile A's DMA is listed right behind x with no wait), delay chain 32 -> 24 copies (tile B's DMA listed after the x StoTrf, before the tile A wait)** | -686 / -482, verify **-621** (9/9 rounds) | **committed** |
| K2: x seed + two memsets, 32 copies | -108 | a memset costs ~0.83k on hardware |
| K2: 24 copies alone | -13 | only the combination pays |
| K2: pe4 epilogue layout (stride-8, 4-engine operand loads and store) | +1,866 | closed again |
| K3: destination of cluster 1's partial transfer pre-written by a local Main pass in the prologue (its readiness sync leaves the kernel start; the 1.1-1.24k DMA-queue gap after the first DMA drops to ~0.64k; every later landing ~0.8k earlier) | -610 / -810, third run **-943** (9/9 rounds) | **committed** |
| K3: Down w0/w1 stream -- bit 8 only toggles every 64 requests (all engines of a cluster read the same chunk index and chunk strides are even multiples of 256) | probe (wrong numerics) -2,317 | no correct form under the K-split: the parity of j+c cannot vary inside a cluster without the other cluster's GeGLU half |
| K1: fewer DMA commands | -- | closed: the 11 loads read 11 distinct input tensors once each |
| K1: junction loads on PE0 only | +361 (trace) | each cluster's DMA queue is in order whatever engines a command uses |
| K1: all 5 junction loads removed (probe) | -4,335 | unreachable ceiling (~0.87k per load) |
| K1: q-scale load moved behind K via a zero anchor | +1,666 | the K->Q junction grew to 6.3k (queue idle 2.2k before Q) |

Official grade of d9b447a (e030bc97): 69,838 / 35,514 / 199,212 = 9.1250 (a slow draw: K1 +875 with K1 code
unchanged). 45-seed sweeps of d9b447a on Arena (`scripts/opt/sweep.sh`): K1 135/135 checks PASS (max abs err
0.0625, inside atol 0.04 + rtol 1e-2), median 69,229; K2 45/45 PASS (max 0.03125), median 35,204; K3 45/45
PASS (max 0.01172), median 197,938 -> score from the 45-seed medians ~9.20.

## Round 13 (2026-09-26): pdma closed; K3 small-load placement -1.2k

| change | B-A | verdict |
| :-- | --: | :-- |
| **K3 C2: gate global scale loaded to 1 slice per cluster + ring-32 Broadcast1 (queued seed 0.95k vs 1.80k), memset anchor on the rms-weight chain so the up-scale DMA issues with rms-w at 0.1k (its landing -1.3k), and the g3 Down-scale VRF built on Main** | -1,003 / -1,219, verify **-1,206** (9/9 rounds) | **committed** |
| C1 (C2 without the Main g3 scale VRF) | -1,092 / -342 | superseded |
| pdma merge of small inputs into one scratch HBM tensor + one tdma load (K1 4 inputs / K2 3 operands / K3 x + rms-w) | **+6,977 / +4,000 / +3.5k** | **closed**: in a compiled kernel `pdma` HBM->HBM lowers to a dram->dram dma_command on the same PE tensor-DMA engines and sits in the same in-order per-cluster queue (~1-1.3k per copy, like a small load); each write to the scratch adds a cluster_sync (writes to one tensor are serialized per tensor, not per tile); rope_offset (i32) cannot join a bf16 scratch |
| K2 probe: drop residual + norm_weight tail loads (wrong numerics) | -312 | ceiling of any K2 operand-load lever |
| K2 E1a: eps+sqrt folded out of the rms pass | +43 | the reduce's to_vrf needs Sub and waits for the y StoVrf |
| K2 E1b: final pass streams y with a nw*ws VRF | +792 | a Sub cast-VRF beside a Main vector pass stretches both |

K2 facts: tile B has no contraction-A contention -- it streams at F (~1.08k) + bytes/647; the gap to
620 B/cycle is the second command's fixed cost. The real post-sync chain is sync -> y StoVrf (1.24-1.37k)
-> rms (0.41k) -> final (0.53k) -> store (0.87k); the reduce hides under the StoVrf.
Official grade of 849283b (0ebbd92e): 69,315 / 35,669 / 196,918 = 9.1699; leaderboard shows the team at rank 4
(9.1765) on 2026-09-27, rank 3 EoM 9.2514.

## Round 14 (2026-09-27): three tail/junction designs closed by their probes

| design | B-A | verdict |
| :-- | --: | :-- |
| K2 progressive gather (tile A rows gathered and reduced during tile B) | probe (all the moved work removed, wrong numerics) +99 | closed: the y StoVrf already runs in parallel with ms+reduce; only ~134 of the ms pass would move; a narrowed second gather lowers to the same full-row descriptor |
| K3 early add/ms of Down rows 0-11 during g3 | probe PX -197; Sub variants ICE | closed: Main and Sub share the Vector Engine, so a Sub vector pass cannot overlap g3 (which holds the VE for 5.2k); the 3-row late chain saves <= 0.83k even if the 12-row work were free |
| K1 V split into two row tiles (3/4 + 1/4) | +2.6k (trace, unpaired) | closed: the tail contraction saves ~1.2k but the extra command costs ~1.1k; gathers ~1.05-1.2k regardless of size; ceiling ~-0.3k |
| K1 warm junction loads (q-rms moved ahead of Q behind k scale; q norm variants) | H3 -7 (mixed rounds); B' +2.2k (trace) | closed: a load queued behind another while a contraction runs costs 0.99-1.17k vs 1.21-1.43k cold (the 0.8k figure only holds with nothing beside it), so warm saves <= 0.15-0.3k per load; moving a load between junctions only shifts the cost; the order comes from the LIR list scheduler (lir.lifetime.json), where a landed stream's contraction ties with the next DMA and is listed first |

Tools: `scripts/opt/submit_loop.sh` keeps submitting the current `opt-next` whenever the grader is idle
(so the last submission is always the newest verified tree).
