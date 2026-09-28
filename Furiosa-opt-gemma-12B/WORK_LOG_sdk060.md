# Work log — Stage 1 kernel optimization

Team EDABK Spidey, MICRO 2026 MOA round 1. Last updated 2026-09-22.

This is the main working log; raw measurements and reproduction details also live in `reports/`.
It replaces nine documents that grew during the project
(`OPTIMIZATION_LOG.md`, `HIEN_TRANG.md`, `CHANGES.md`, `KIEN_TRUC.md`, `TONG_KET.md`,
`FORUM_QUESTION.md`, `OPTIMIZATION_STATE.md`, `FAILED_EXPERIMENTS.md`, `BEST_CONFIG.md`).
Their full text survives in git — `git show bb6b08c:HIEN_TRANG.md` and so on — but everything
worth acting on is below. `README.md`, `ARCHITECTURE.md`, `OPTIMIZATION.md` and `SERVING.md`
are the repo's own documents and are untouched.

---

## 1. Where things stand

### 2026-09-22 follow-up — paired RoPE, measurement correction

The latest official best verified at the start of this follow-up is **7.4924**
(`4912db55`), superseding the historical 7.2050 entry below. The historical public
Arena best remains 7.9614. **An 8–9× result has not been demonstrated.** These are
Stage 1 kernel scores, not end-to-end model-serving speedups.

The working tree adds a paired RoPE implementation to `35b7838`: two multiplier
ALUs compute the cosine and sine products, and Clip adds their groups. This removes
the `x*cos` DM temporary and its VRF reload for Q and K, retaining the existing bf16
rounding boundaries. All 15 kernels compile. Both diagnostic Arena runs pass all
three tests with unchanged Q/K/V maximum errors. K1 static makespan is
**42,388 → 42,246**, with **175 → 171** instructions.

Two alternating trace pairs (74377/74397 and 74421/74438) give K1 hull-minus-wait
medians **87,428 → 85,681 cycles**, a 2.00% reduction in that diagnostic. This small
sample does not establish an overall score gain. Raw cycles, the source patch,
hashes, schedules, rejected experiments, and reproduction instructions are in
[`reports/2026-09-22-codex/README.md`](reports/2026-09-22-codex/README.md).

Two further alternating pairs with the **unchanged test harness** and ordinary
`info` profiling (74453/74463 and 74477/74482) also pass. Estimated score medians
are **6.9969 / 7.2506**, candidate best **7.4386**. K2/K3 binaries are byte-identical
between variants but their times vary, so the total-score difference is not a
clean estimate of the patch's effect. The final root build's graded kernels match
the tested candidate byte for byte. Official candidate submission: **`dd0580b2`**.

The official result is **7.1154**, with K1/K2/K3 median cycles
**93,249 / 49,221 / 227,044**. All three tests pass on **seven independently
seeded inputs each**. The official best remains **7.4924**: this patch has not
demonstrated an official score gain. Both official logs (`dd0580b2` and
`4912db55`) explicitly report seven runs and medians. This supersedes the old
one-run-grader assumption and the recommendations to optimize for isolated
best draws below. The repository's public Arena test still runs once per case.

**Correction to the older measurement rule:** use the **union**, never the sum,
of overlapping `Cluster` spans. Also, `total - Cluster_union` is a diagnostic
residual, not proven useful-work time: engine spans can overlap waits and contain
their own stalls. The new `scripts/analyze_trace.py` reports that overlap explicitly
and has regression checks against the overlapping waits in Arena 72093.

### 2026-09-22 — branch `opt-next`, four accepted changes on top of the 7.2050 tree

| commit | kernel | what | measured |
|---|---|---|---|
| `32ebe25` | K3 | lean tail: pair reduce broadcast, ring-64 gather, 4 contiguous scratch pieces, fused epilogue | Arena 237-248k vs 253-255k |
| `b7dd091` | K3 | down block scale as 480-byte cluster rows + ring-2 InterTranspose | Arena 233-244k |
| `ab0d8f5` | K3 | block-scale multiply-accumulate moved from the Vector Engine to the **Contraction Engine** (up, gate, down) | static 122,167 -> 112,962; TU instructions 249 -> 150; max\|delta\| 0.0078 -> 0.0039 |
| `a7613ca` | K1+K2 | odd-stride query weight layout; per-cluster half norm with a scalar exchange | K1 work -5k, K2 work -9k (traced) |
| `544d1db` | K3 | down weight as 15 whole cluster rows per slice (3,840-byte runs), scale still on the CE | K3 DMA busy 213.1k -> 202.9k, work -10.5k |
| `e08affc` | K1 | odd-stride k/v pair layout, permute chains dated behind the RoPE broadcast | K1 work 89.5k -> 85.2k |
| `a6645ef` | K3 | the up/gate block-scale decode moved to the sub context (it needed no Vector stage) | K3 work 217,967 -> 217,327 over two traced runs each |
| `538e1bf` | K3 | the down weight as two chunks (12 + 3) instead of three: one DMA command and one LUT table staging fewer, same exposed decode | K3 work 220.7k -> 218,256 |
| `d003b4f` | K2 | one weight transfer again: every DMA command costs a fixed 1.9-2.7k, so k pieces cost k x F + 22,231 whatever the cuts | K2 work 32.9k -> 30.5k |

Traced work (total minus the cluster-skew span), same harness, 2026-09-22:

| | control tree | opt-next @ 544d1db |
|---|---|---|
| K1 | 95,440 / 92,971 / 95,519 | 86,522 / 84,904 |
| K2 | 37,073 / 37,869 / 37,537 | 30,509 / 31,431 |
| K3 | 230,273 / 228,803 / 225,187 | 221,217 / 220,127, then 218,256 with the two-chunk down |

i.e. **K1 -8.9k, K2 -6.5k, K3 -9.5k, 25k of work in total**, each measured across several traced runs.

Arena draws of the whole tree, same afternoon, five runs each (the spread is launch skew, not the tree):
control `ab0d8f5` score median 6.852 (K1 98,390 / K2 50,273 / K3 237,025);
`d003b4f` score median 7.124, best 7.573 (K1 95,562 / K2 48,783 / K3 228,764), and one draw of the
same sources at 7.961. The graded 7.2050 was drawn on 2026-09-14, on a visibly faster machine day.

Recovery of the graded 7.2050 tree is unchanged: `git checkout worktree-as-found`.

### The measurement rule that changed everything

**A kernel's time is (its own work + its own launch skew), and the skew is random per kernel
launch.** In one job the cluster wait was K1 1,559 / K3 3,668 / K2 10,615; in another K2 drew
28,632. A K2 built with the cross-cluster exchange deliberately removed (numerically wrong,
Arena job 72055) still waited 14,301 — so the wait is not the exchange, it is the launch.
Consequences:
 1. Never judge a change by one draw. Use alternating measurements of the official totals.
    For diagnosis, run with `TUC_PROFILE_LEVEL=trace` and inspect **total minus the union of
    Cluster spans** and **DMA span union**. These include the limitations described in the
    follow-up above; neither is a replacement for the official metric.
 2. The official grader takes seven runs per kernel and reports their median
    (verified 2026-09-22). Historical single-run Arena minima do not predict that
    score. Compare repeated measurements and retain the raw samples.


| | |
|---|---|
| Best graded score | **7.2050** — MOA submission `cf6978f5`, 2026-09-14 17:04 |
| That submission | K1 94,120 · K2 42,679 · K3 249,860 |
| Tree that produced it | commit **`bb6b08c`**, tag **`worktree-as-found`** |
| Recovery | `git checkout worktree-as-found` |
| Speedup over baseline `ed1db81` | K1 2.92× · K2 2.99× · K3 44.4× |

**The implementation was uncommitted until 2026-09-16.** `HEAD` was still `8504287` and the
whole ~2,000-line result — `f8split.rs`, `mlp13.rs`, `sliding/tail.rs` plus eight modified
files — existed only as working-tree state. It is now committed and tagged. Verified: checking
`src/` out of the tag and rebuilding produces a **byte-identical** test binary, so a rebuild is
a free control-identity check before spending a job.

### Scoring

`score = (B1/K1 · B2/K2 · B3/K3)^(1/3)` — the geometric mean of the three speedups. Fitted
against all ten leaderboard teams: `3·ln(score) + ln K1 + ln K2 + ln K3 = 40.4667 ± 1e-4`,
equivalently `score = (3.754e17 / (K1·K2·K3))^(1/3)`.

**A 1% cut in any kernel is worth the same 0.33% of score, regardless of that kernel's size.**
Most early effort went into K3 because it is the biggest in cycles; that was the wrong
allocation. Headroom against the best value *any* team has posted per kernel:
**K1 23.4% · K2 18.0% · K3 4.1%.**

---

## 2. Read this before optimizing anything

**The same binary scores between 6.4 and 7.2.** Six Arena runs of `bb6b08c` (jobs 42133, 42135,
42150, 42204, 42211, 42221): K1 98,985–113,581 · K2 42,779–48,430 · K3 246,891–255,673 · score
6.5728–7.0284. Per-kernel swing on identical code: **K2 ±24%, K1 ±6%, K3 ±3%.**

The public test harness runs each kernel **once** per job with no warm-up, and its
reported number is a convex hull — `max(end) − min(begin)` over every `span::npu`
span (`tests/test_kernels.rs:612`). Delayed cluster starts affect that measurement;
K2 is particularly sensitive because it is the shortest kernel. **The official
grader differs: seven independently seeded runs, then the per-kernel median.**
The older attribution of all variability to random launch skew is a hypothesis,
not a proof that kernel scheduling or shared-machine effects cannot matter.

The noise is **per-kernel, not machine-wide**: between two back-to-back runs K3 got 2.6% faster
while K2 got 11.5% slower.

> ⚠ **Unresolved, and the resubmission strategy depends on it.** The per-kernel decorrelation
> above says there is no "fast window" and only the number of attempts matters. But an earlier
> session recorded the opposite: identical code scoring 7.10 at 11:25 and 6.30 at 12:00 with K3
> alone drifting 254k → 270k, and 7.0446 → 6.2695 with K3 250k → 277k — and concluded "never
> compare two submissions taken at different times; submit when the machine is idle". Both
> readings are in the git history; neither has been tested against the other. If the drift
> reading is right, resubmissions should be clustered in a quiet window rather than spread out.

Historical conclusions from the single-run measurements follow. Their minimum-
and resubmission-based recommendations are **not established for the current
official median metric**; use alternating measurements and report distributions.

1. **A single run decides nothing.** Compare using alternating paired runs, and judge on the
   **pooled minimum over every sample ever taken of each tree**, never on the minimum inside one
   pairing run. E20 was accepted on 7 control samples and reverted once 18 were pooled — it lost
   on both statistics.
2. **A verdict is only valid against the exact baseline it was measured on.** The same K2 tail
   patch was a wash before E13 and +41% after it.
3. **Resubmission is the largest single lever in the project.** The tree's median is ≈6.75 and
   its best draw is 7.2050; all three per-kernel records in one run would be ≈7.35. Median → best
   is worth **~7% of score**, more than any code change ever accepted here except E13. Scoring
   keeps the best result (confirmed with the organisers 2026-09-14), so resubmitting unchanged
   code is a free lottery ticket.

---

## 3. How this machine actually behaves

Everything here was measured, mostly at the cost of an Arena job each. None of it is in the
vendor manual in this form.

### The binding tier is DM write, not HBM

The manual's HBM cost model predicts 15.7 MB in ~10,200 cycles (≈1,540 B/cycle). Nothing ever
came close. The working ceiling is the **DM write stage at ≈683 B/cycle** (4 DMNs × 128 B/cycle
at 0.75 GHz, expressed in 1 GHz cycles).

**Unresolved contradiction, and it must travel with the number:** one trace of K3's up stream
measured **709 B/cycle**, above that ceiling, while the table below lists the same stream at 623.
Either the ceiling is higher than 683 or one of the two span measurements is wrong. The *rank
order* of the streams does not depend on resolving it.

| stream | geometry | cycles | B/cycle | % of 683 |
|---|---|---:|---:|---:|
| K2 o_weight | 120 runs × 256 B, 30,720 B/slice | 24,142 | **652** | 95% |
| K3 up (f4) | 1 run × 57,600 B/slice | 47,312 | **623** | 91% |
| K1 k+v | 1 run × 15,360 B/slice | 32,432 | 485 | 71% |
| K1 q | 1 run × 30,720 B/slice | 34,460 | 456 | 67% |
| K3 down (f4) | 30 runs × 1,920 B | 65,700 | 449 | 66% |
| K3 block scales | 1 run × 7,200 B/slice | 10,093 | 365 | 54% |

### What actually predicts the rate

**Bytes per slice**, almost monotonically, within the one-run-per-slice family:
7,200 B → 54% · 15,360 B → 71% · 30,720 B → 67% · 57,600 B → 91% · 61,440 B → 88%.

This is the one rule that has held across every stream, including the pathological case: K3's
tail scratch store moves 30,720 B at **6.8 B/cycle** because each of its 128 live slices holds
only 120 bytes.

Run length matters too, and **not monotonically** — 256 B is a sharp isolated spike:

| run | 128 B | 240 B | **256 B** | 480 B | 1,920 B | 3,840 B | 15,360 B | 30,720 B | 57,600 B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B/cycle | 308 | 293 | **635** | 377 | 449 | ~637 | 480 | 456 | 623 |

The arithmetic behind it: a packet costs `ceil(n/256)` AXI requests. 480 B is 1.875 requests so
it costs 2 — a 7% request inflation. K3's 1,920 B runs are 7.5 requests each.

**Fan-out is not the mechanism.** E11 was built to discriminate: doubling K2's fan-out from 16 to
32 slices while halving the run moved the identical bytes from 24,753 to 51,012 cycles
(635 → 308 B/cycle). Run length and per-slice bytes decide; the number of live slices does not.

### What is *not* the problem (each closed with evidence, do not re-audit)

- **HBM channels.** All 32 channels take an identical number of 256 B requests in every stream.
  A 16-request window touches 15.8/16 distinct channels for contiguous layouts, 14.9 for the
  strided one — and the *better*-spread K1 query runs 30% *slower* than K2's output weight.
  Channel spread is anti-correlated with speed.
- **HBM rows.** 1.0 distinct row per 16-request window everywhere.
- **DM bank starvation.** All 178 `fetch_unit`/`commit_unit` sequencer configs across the three
  kernels were scanned: no stride 0, no stride that is a multiple of 128 B with limit ≥ 64. The
  only non-zero strides present anywhere are 1, 4, 8, 16, 32, 64 B.
- **The compiler stealing DMA slots.** No instruction in any schedule dump carries both a Tensor
  Unit context and `DmaEngine`. DMA spans are pure transfer time.
- **Slice-stride parity.** A model fitting seven points predicted the eighth and ninth
  *backwards* (E27). Odd stride destroys per-slice contiguity, which is what the fast streams
  actually have. For K1's query the two are provably mutually exclusive.

> ⚠ **The static schedule runs ~2–2.2× faster than hardware.** Every `--dump-schedule` number in
> this file (makespan 122,167 for K3 against a real ~253k; `total_execution_cycle`; `io_only`) is
> in static units. Use the dump for *structure and ordering*, never as a cycle prediction, and
> never compare a static number against a measured one.

### The DMA queue

**Strictly serial.** Two transfers never overlap — not for disjoint HBM tensors into disjoint DM
tensors, not in any trace ever taken. K1's k occupies 8,154–24,194 and v 24,196–40,586 with not
one cycle of overlap.

- A single `to_dm` over 256 slices is **already spread across all 8 DMA engines** by the
  compiler; splitting a command only reduces engines per command.
- There is **no API in furiosa-opt-std 0.6.0 to select a DMA engine**, despite
  `ref/furiosa-opt-book/16-dma-engine.md:147` saying the kernel writer can. This is the only
  surviving form of the "run two transfers in parallel" idea.
- Fixed cost per transfer ≈ 1.1k cycles (1.5–3k by another measurement). A transfer cannot span
  two HBM tensors, so **every HBM tensor costs at least one transfer** — which makes K1's ten
  small inputs an 11k-cycle hard floor.
- A replicated (256-slice) load of a tiny vector costs **5–10k cycles** (~7k per scalar in K3).
  The working pattern is: load onto one slice per cluster, then Switch-broadcast. Doing this for
  K3's global constants was worth **−21k cycles**, the largest single step in the K3 series.
  Padding makes it far worse: a replicated `1 # 8` element costs ≈7.4k, a `1 # 2` element ≈**33k**,
  and 192- or 240-slice layouts (`# 256`) **ICE the DMA lowering**.
- **There is no DM→DM path between clusters.** The compiler rejects it outright ("not synchronized
  from other clusters"). `dm.to_hbm(&mut ctx.tdma)` allocates kernel-private HBM and `to_dm()`
  reads it back replicated on both clusters — that round trip is the *only* cross-cluster
  mechanism, which is why K2's and K3's exchanges cannot simply be deleted.
- **A read waiting on the other cluster blocks every stream queued behind it.** This is why K3's
  single exchange sits at the very end. An early "fake sync" to pre-pay skew cost K2 **+9k**.
  Related: the **first** cross-cluster wait pays the entire launch skew; every later one costs
  only ~0.4k — so a proposal that adds a sync *after* the first is much cheaper than one that
  adds the first.
- **The core blocks at `StoTab`** and at every Tensor Unit program until its input DMA returns.
  A 4 KB lookup-table load placed between two weight streams therefore delays the second stream
  (in mlp13, gate16 starts exactly when the second `StoTab` starts). This is a separate mechanism
  from the transfer-count rule and is why 12 LUT programs were merged into 6.

### The scheduler

- **`to_dm` call order does not set the DMA queue order.** The scheduler places every transfer
  itself, ALAP relative to each consumer. Four different source orderings of K3's down chunks and
  block-scale produce a **byte-identical binary** (`md5 106a762937af…`) and an identical schedule.
  **Any candidate whose whole content is "issue this transfer earlier/later" is void** — check a
  `--dump-schedule` before building one.
- Only a **data dependency** changes order. That is what K1's `kv_anchor` is for.
- It **ASAP-fills small transfers into the earliest free gap**, and every transfer placed before
  the first weight stream costs roughly its own span *again* in delay to that stream. E8, E9 and
  E12 each paid 1–2k per added transfer, wiping out predicted savings of 2.4k, 5.3k and 9.7k.
  **Transfer count is the lever; transfer cost is not.**
- K1's 2.8k prologue DMA idle is **core-issue-bound, not queue-bound** — the core has not reached
  the next DMA instruction because it is executing Tensor Unit programs in order. Three separate
  mechanisms (E8, E9, E33) tried to fill it; three nulls. The gap is not a cost.
- `SCHEDULER_MANUAL_ORDERING_PATH` only constrains Tensor Unit operators and only in a local
  build. The grader rebuilds from source, so it is useless there.

### The Switch Engine

**A sub-ring must be either FULLY live or have EXACTLY ONE live source. Anything in between
hangs the device**, with no compile-time diagnostic — HAL error PXI-601, Execution: Timeout.

Density does not rescue it. E2 was 8 live in a ring of 256 (1-in-32) and hung; E38 was 8 live in
a ring of 32 (1-in-4, thirty-two times denser) and hung identically. Bank starvation was
suspected and statically ruled out, so this is its own failure mode.

> ⚠ `normalize_replicated_from_chunks` (`src/device/shared/rmsnorm.rs:180`) has exactly this
> shape, has no callers, and has never been validated. **Do not call it.**

Other Switch facts:

- **Delivery is ring-MINOR.** With one packet per slice, ring-major and ring-minor coincide and
  this is invisible — which is why the shipped ring-32 gathers are correct. At ring 8 with four
  packets per slice the interleave shows: value position `p = 64f + 8j + i` rather than row order
  `32j + 8f + i`. Cost three Arena jobs to establish (42161 / 42215 / 42223).
- **Factoring a time digit does not express delivery order.** `m![Qs % 256 / 32, Qs % 32 / 16]`
  normalizes to `m![Qs % 256 / 16]` — the same mapping written two ways. Jobs 42161 and 42215
  returned identical wrong answers for this reason. Order lives in the `SwitchConfig` only.
- `Broadcast1` takes a **single stride**; it cannot express a source split into two runs 128
  slices apart. Any such gather degrades into a DM relayout at ~4.7× the cost (E23: 700 → 3,301).
- A large-ring scatter transmits to the whole ring, so its cost multiplies by ring size.
- Switch traffic **contends with DMA for the DM port** — mlp20 made K3's down streams 15k faster
  and still netted 258.6k, because its 29.5 MB of Switch traffic took the ports back.
- `Broadcast01` / `CustomBroadcast` produce an EDF but no runnable binary. ⚠ **This is a known
  vendor-side issue and `CustomBroadcast { ring_size: 256 }` is still used by shipped code** —
  `layout.rs:20,36`, `shared/lm_head.rs:25`, `vision/projection.rs:20,228`,
  `audio/projection.rs:89`. Those are not the three graded kernels, but they must still compile.
- The other open vendor issue touching shipped code: **`dma_gather_unscaled` fails the V1
  scheduler on a 64 KiB SPM request**. That call is E1, at `sliding/tail.rs:272-273`, inside K1.
  It has never failed in practice here, but nothing else in the tree records the risk.

### Layout rules, and the wall they build

Three rules must hold **simultaneously** for any weight mapping:

1. the slice mapping must multiply to **exactly 256** — otherwise
   `slice extent N does not match the device config (slice = 256); a DM allocation must span the
   whole device`;
2. a run should be a multiple of **256 B** (`ceil(n/256)` AXI requests per packet);
3. rows per slice must be **even** — the Transpose Engine folds `[hi, lo]` lane pairs and takes
   at most four valid rows per output packet.

**`H = 3840 = 2⁸ × 15` and `L = 15360 = 2¹⁰ × 15`. Fifteen is odd**, so it never divides 256 and
never pairs. Full enumeration:

- K1 query, strips of `w` bytes: 256 B needs 17.07 row groups, 768 B needs 51.20, 1,280 B needs
  85.33 — none integral. Only the 3,840 B strip works, which is the layout already in use.
- K3 down, full row 3,840 B: 15 rows/slice is odd; 16/20/24 rows give 240/192/160 slices which do
  not divide 256; only 30 rows (128 live) and 60 rows (64 live) are legal.

**This is why K1 can never borrow K2's fast 256 B geometry.** K2's contraction axis is
`Qs = 4096 = 2¹²` and divides cleanly; K1's is `H`. It is a proof, not a measurement.

More compiler rules, each found as a build error:

- a `tile` offset must be a multiple of the tile window (`stride must divide the size`);
- `commit_trim` packets may only be **8, 16, 24 or 32 bytes** (3 × f32 = 12 B rejected);
- a DM tensor's in-slice element extent must be a multiple of **8 bytes** (15 × f32 = 60 B rejected);
- a VRF `collect` must land **exactly one 32-byte flit**;
- a **packet is at most 32 bytes** (`ref/furiosa-opt-book/12-sequencer.md`);
- a DM→DM relayout needs contiguous valid bytes — a `1 # 8`-padded source fails
  `tail_size % min_align (4) != 0`;
- HBM scratch does not accept a padded axis;
- `VRU reduce axes must be innermost in Partitioning` — this is what forces a row's two column
  halves onto *adjacent* slices in K3's down projection;
- a reduce axis may not carry padding;
- a VRF operand's packet axis must be innermost-contiguous or absent — a missing middle `Time`
  digit silently broadcasts instead of erroring;
- logic ops are **Way8 only**, and an anchor's legal spelling depends on the Way: `vector_clip`
  anchors work only at Way8 (before Narrow); at Way4 use `AddF` against a zero VRF. This is a
  third, distinct entry in the anchor-hazard family (after the `Max` bug and the stage-ordering
  one);
- a contraction packet is at most **2 flits** (64 f8), hence at most 4 blocks per packet;
  `LaneMode::Sequential` yields `OutTime = [Time, Lane, packet_outer]`,
  `OutPacket = [packet_inner # 8]`;
- the **"owner mismatch" checker** rejects a two-piece output projection — an independent
  compile-time block on the whole "split the weight so the first contraction overlaps the second
  stream" family in K2. `contract_output_96` / `contract_output_24` still sit in `projection.rs`
  as the evidence;
- a DMA **destination** covering a slice sub-range is inexpressible: `DmTensorViewMut` has no
  `slice_tile` (only the immutable view does), its `tile` only rewrites `Element` so a slice digit
  gives `visa: IndexAccess … cannot find tag`, a reshaped destination gives
  `mir: expected a DmTensorViewMut`, and `Resize`/`Padding` pin the live window at index 0 so
  every partially-live mapping contains slice 0. Five spellings, five different errors.
- the device compiler lowers MIR **before** const-folding, so a `match` on a `const` inside a
  `#[device]` call graph fails with `Mapping not found`, and a `const` of a custom enum fails with
  `const fill does not support this scalar type`. Variant selection must be done by rewriting
  source text — that is what `scripts/variant.py` is for.

One inferred rule that turned out to be **wrong**: digits of one axis do *not* have to nest. A
slice mapping may have three digits and its innermost may sit below an element digit. The false
rule would have ruled out valid candidates.

### The Tensor Unit

Stage order is **fixed**: Fetch → Switch → Collect → Contraction → Vector → Cast → Transpose →
Commit. **All bit logic is at the Vector stage, i.e. after the multiply.**

- The Contraction Engine multiplies only **matched types**: i4/i5, i8/i9, bf16×bf16, f8×f8.
  `f4e2m1` has no `ContractionCast` impl and can never be an operand, and `FetchCast` has no f4
  impl — so the NVFP4 table lookup is **architecturally mandatory**, not a style choice.
- Integer contraction is impossible: the table lookup cannot emit integers.
- One **narrow/widen transition per pass**. `vector_fp_unary` needs the narrow state,
  `vector_clip` needs wide, and binary ops are unavailable in the narrow state after a reduce.
  Therefore `sqrt(mean + eps)` cannot be fused — the two-pass RMS is the engine minimum (E14,
  three orderings, three distinct compile errors).
- The Clip cluster (Add) sits **after** the Float cluster (Mul), so `(x + r)·s` cannot be one
  pass.
- **Pass count dominates pass width.** VRF is 8 KB/slice; K3's 8-row × 240-f32 block-scale
  operand is 7.68 KB = 96% of it and does *not* break pipelining. Halving it to get more passes
  lost all three paired runs by ~6,200 cycles (E18). Widening is impossible: 15 rows needs 14.4 KB.
- TRF is 64 KB/slice and only 12% used — but every graded kernel is a matrix-vector product over
  a **single token**, so each weight byte is used exactly once. There is no reuse for a
  weight-stationary scheme to capture.
- `begin_interleaved` takes two `DmTensorView`s — tensors **already in DM**. It fuses two *vector*
  passes, not two DMA transfers. The name is misleading.
- `fetch_table_lookup` costs one 4 KB table DMA plus a `StoTab` of ~5k cycles per program, and
  tables cannot be shared between programs.
- **Tensor Unit work is free.** E34 removed an entire Tensor Unit program and ~86 KB/slice of DM
  traffic from K3 and the kernel changed by **+474 cycles** — nothing. Everything that fits under
  the DMA queue costs nothing.

### The governing rule

> **The bottleneck is the DM side, and everything under the DMA queue is free. A change wins only
> if it removes DM traffic or removes a DMA command without adding one.**

All four accepted changes fit that shape. Every failure paid a DM-side cost to buy an HBM-side
gain and lost by roughly what it paid — E31 +83,155 (+33%), E29 +18,075 (+41%), E30 +9,548,
E22 +7,900.

---

## 4. Why the code looks the way it does

Each item below reads as waste or as a bug in the source. None of it is.

### The core technique: `f8 × f8` with a two-lane activation

`src/device/shared/f8split.rs`. The Contraction Engine needs matched types, so either the weights
go up to bf16 or the activation comes down to f8. Weights are ~146 MB per layer and the
activation is one vector, so the activation moves:

```
x (bf16) ─── ×X_SCALE (a power of two) ──→ x·S (f32)
                ├─ hi = reinterpret i32 → BitAnd 0xFFF0_0000 → reinterpret f32 → cast f8e4m3
                └─ lo = (x·S − hi)                                              → cast f8e4m3
            w·x = w·hi + w·lo,  exact to bf16
```

**This is a throughput device, not a precision trick.** A contraction flit holds 32 bf16 *or*
64 f8, and the f8 path carries two lanes per step — 128 element-multiplies per step against 32,
a factor of 4. The doubled contraction work is free because all three kernels are DMA-bound.
Measured directly: rewriting K1's query as bf16 × bf16 with the weight decoded at fetch gave
K1 = 113,213 against a ~100,000 control (E19).

**The `×16` / `×64` constants are load-bearing.** Without the power-of-two pre-scale the `lo`
lane falls into the f8e4m3 subnormal region (normal range 2⁻⁶ … 448) and loses bits — K3's error
inflates to 1% against a 0.01 tolerance. Power-of-two scaling is exact in binary floating point
and the reciprocal folds into a multiplier that already exists, so it costs **zero passes**.
`X_SCALE = 16` in K1/K2, `64` in K3, `G_SCALE = 1024` for GeGLU output.

> ⚠ **An ordering anchor must be additive to be neutral.** `split_scaled_anchored!` originally
> used `vector_clip(ClipBinaryOpF32::Max, anchor)` — i.e. `hi = max(hi, 0)` — which silently
> destroyed every negative weight: K1 max|Δ| 2.95 against a 0.04 tolerance (jobs 34166, 34183).
> Fixed to `::Add`. The helper has no callers today; the fix is kept as a trap removal.

> ⚠ **Building a zero from few-live-slice data:** mask every bit off
> (`reinterpret i32 → BitAnd 0 → reinterpret f32` gives `+0.0` for any bit pattern). Multiplying
> by zero does **not** work — a tensor live on 4 of 256 slices leaves the other 252 reading
> uninitialised DM, and `NaN × 0 = NaN`.

### K1 — `sliding_project_qkv`

| choice | why |
|---|---|
| query rows `m![Qs % 2048 / 8]` — only 8 rows, 30,720 B per slice, 256 live | the *only* legal layout under the factor-15 wall; looks wasteful, has no alternative |
| k and v as one paired DM tensor `m![Gs, Ps % 4, H]` | one transfer instead of two; E20 split them, won on a small sample, and was **reverted** once samples were pooled |
| `kv_anchor` — an all-zero f32 VRF value derived from the k/v result | pure dead arithmetic whose only job is to give the scheduler a **data dependency** so the query weight streams *after* k/v. Deleting it silently reorders the streams |
| per-cluster RoPE gather with `dma_gather_unscaled` (E1) | `dma_gather_scaled` reads its index from HBM and **does not replicate to cluster 1** — cluster 1 got garbage. The unscaled form takes an index already in SPM, so each cluster computes its own row (`rope_offset >> 9`). Saved 2 transfers and one cross-cluster ordering constraint |

Cycle budget (~100k, trace job 34347): prologue 8.2k · k+v 32.4k · **ten small loads 11.4k** ·
q 34.5k · tail after the last byte 14.8k. The DMA queue is busy ~97% of the kernel.

K1's non-weight DMA, itemised from the static schedule (multiply by ~2.2 for hardware):
x + rms_weight 6.2k · RoPE path 7.9k · row scales 5.4k · norm weights 2.4k · output scatter/store
5k.

**K1 is bimodal** and the mechanism is known: in the slow mode the scheduler drops the small
scale and norm loads *between* the two big k/v transfers, widening the window. Roughly 3 in 16
samples land in the fast mode (below 96k). Pinning it would be worth ~3,000 cycles; three
attempts to anchor the ordering failed with `commit: input does not match pipeline`.

### K2 — `sliding_attention_output`

| choice | why |
|---|---|
| 16 row groups × 16 column strips, 256 live slices, **120 runs of exactly 256 B** | this is the 635 B/cycle spike. A knife-edge optimum — 128 B gives 308, 240 B gives 293. Any relayout destroys it |
| its **own** scratch tensor rather than borrowing `residual_hbm` (E13) | the single largest accepted win: **−11% on K2** (median 43,613 vs 48,988, 4/4 paired). Same bytes, same transfer count — only *which* HBM tensor receives the two result halves changes. The 10,085 cycles in three `Cluster` barrier spans were never the exchange, they were the **aliasing**: a read-after-write on a tensor both clusters touch. That distinction is why E10, attacking the same cycles with a new layout, lost badly (71,450 vs 44,000) |
| an HBM round trip between projection and post-norm | **structurally required** — each cluster owns only half of `H` (1,920 of 3,840 rows) while the post-norm needs all 3,840 to form the mean square. That meeting point is also where cluster skew is paid |

The weight stream is **finished**: 24,142 cycles at 652 B/cycle = 95% of the ceiling, estimated
floor 23k. Everything left in K2 is the cross-cluster barrier and launch skew. This is why K2
optimization attempts keep failing — the stream is done before you start.

### K3 — `decoder_feedforward`

| choice | why |
|---|---|
| NVFP4 decoded by `fetch_table_lookup::<f8e4m3>()` **fused into the contraction's own Fetch stage** | the baseline decoded f4 → bf16 and wrote the decoded matrix *back* to DM — about **1.2 GB of DM writes** for K3 alone. The fused path reads 29.5 MB from DM exactly once |
| block scales applied **after** the contraction, onto 16-column block sums | 240 multiplies per row instead of 3,840 |
| down split **16 + 10 + 4** rows, three tile views | looks like three wasted DMA startups plus three ~8% tile-view penalties. E22 merged them into one plain `to_dm` and **lost all four paired runs by ~7,900**: down's contraction is the last thing K3 does, so one chunk puts all thirty rows of decode *after* the last byte lands. Three chunks let chunk A's decode run under B and C's DMA |
| `LaneMode::Sequential` | keeps four 16-wide blocks inside the packet — 4 useful values per packet instead of 2 |
| GeGLU as two immediate multipliers | via `2·gelu(v) = v·(1 + erf(v/√2))`, every constant pre-merged |
| global constants onto one slice per cluster, then Switch-broadcast | **−21k cycles**; a replicated 256-descriptor load costs ~7k per scalar |
| 8 rows per block-scale pass | its VRF working set is 7.68 KB against an 8 KB VRF — a hard limit, not a style choice |
| exactly **one** f32 exchange through HBM scratch, placed at the very end | for `down` each cluster owns *columns*, so it can only produce a partial sum — the exchange is **mathematically mandatory**. It sits last because the DMA queue is serial and a read waiting on the other cluster blocks everything queued behind it |
| `normalize_replicated_from_hbm_keep` returns both `x_hat·64` and the raw `residual_rep` | so the final residual add never re-reads HBM |

Cycle budget (~253k): prologue 8k · up 41.6k · gate ~42k · **down 65.7k** · three block-scale
tensors 30,279 · tail ~36k. **DMA is busy 211k of the 253k.** DMA commands went from ~198 per
cluster in the baseline (the down path alone was 60 × 3) to **8**.

Static schedule (makespan 122,167) and the exposed tail:

| # | transfer | begin → end | cycles |
|---|---|---|---:|
| 1–2 | up weights · up block-scale | 4,653 → 34,168 | 24,612 · 4,897 |
| 3–4 | gate weights · gate block-scale | 35,533 → 66,410 | 24,612 · 4,897 |
| 5 | down chunk A (16 rows) | 66,410 → 82,360 | 15,950 |
| 6 | **down block-scale** | 82,360 → 88,925 | 6,565 |
| 7–8 | down chunk B (10) · chunk C (4) | 89,763 → 105,180 | 10,175 · 4,398 |
| 9 | scratch store (`ops.rs:297`) | 108,676 → 113,211 | 4,535 |

**Exposed tail after the last weight byte = 16,987 cycles, and it is not the contraction:**
4,535 the scratch HBM store · 2,200 the cross-cluster combine (`ops.rs:306`) · 1,463 the lane sum
· 1,203 the down scale commits · ~7,500 the rmsnorm / residual / layer-gate epilogue.

The scratch store moves 30,720 B in 4,535 cycles = **6.8 B/cycle**, descriptor-bound at 128 live
slices × 120 B each. It is the worst instance anywhere of the bytes-per-slice rule.

---

## 5. Closed — do not retry

| direction | evidence |
|---|---|
| Issue a transfer earlier/later in the source | no-op; the scheduler decides (F001, 0 jobs) |
| Partially-live Switch sub-ring, any density | **hangs the device**: E2 (1-in-32), E38 (1-in-4), and again 2026-09-22 (Arena job 72079, `Broadcast1 { slice1: 2, slice0: 2 }` with 2 live sources per ring of 4, to widen the K3 up/gate scale read to 14.4 KB per slice: `HAL error on ClusterId(npu0pe0-3): Connection timed out`). The sub-ring is `slice1 * slice0` slices wide, not `slice1`: every validated Broadcast1 in this tree is either fully live or has exactly one live slice in the whole ring. |
| A one-live-source `InterTranspose` | **hangs the device**: Arena job 72113, 2026-09-22, ring-2 `InterTranspose { slice1: 2, slice0: 1, time0: 300 }` fed by a scale load live on every other slice, same `HAL error on ClusterId(npu0pe0-3): Connection timed out` as the two-live-sources case an hour earlier. The "fully live **or** exactly one live source" rule appears to hold for `Broadcast1` only: an InterTranspose is an exchange. Under `out(s2,t1,s0)@(t2,t0,s1) = in(s2,s1,s0)@(t2,t1,t0)` every output slice carries an `s1` component that must be SOURCED by the other ring member, so a dead partner stalls the ring; in a one-live-source `Broadcast1` the dead slices are pure sinks and never have to source anything, which is why that shape is safe. The lowering verifier checks shape reassembly and sequenceability only, never ring liveness, so both compile and then time out. The shipped ring-2 InterTranspose of `b7dd091` is FULLY live (both slices hold their own half). Treat every InterTranspose as requiring a fully-live ring. |
| Split the K3 down projection by output ROWS across the clusters (so each slice reads 115 KB of whole rows) | **+33k** (Arena 72154: K3 270,554 against 232-237k, numerics PASS at 0.0078). The stream did not get faster either -- the DMA union rose 200,890 -> 207,095 -- and the cross-cluster exchange of `g` that a row split forces put a **98,175-cycle cluster wait** in the middle of the kernel: the two clusters now depend on each other before the down projection instead of only at the end. Closes the "down at 115 KB per slice" idea. |
| Rearrange K1's slice-to-row mapping to speed the stream up (any variant) | **closed by measurement and by the descriptor.** Two first-principles hypotheses were built and both died. (1) "The rate follows the address stride between consecutive slices": refuted by the lowered descriptor - the engine does NOT walk slices in slice-index order, the innermost slice digit has a stride of 983,040 B and the deltas it actually issues are +983,040 / -979,200 / -925,440, so 3,840 and 57,600 are never strides the hardware sees. (2) "The rate follows how many of the 8 HBM channel residues one 64-request walk covers" (natural 1 -> 516-525, odd-stride 2 -> 567-595, K2 8 -> 663): a layout covering 4 residues instead of 2 was built, verified legal and correct, and measured **571 and 583 B/cycle against the shipped layout's 554-590** - no change at all, and K1 work 85,610 against 85,478. Arena 74184/74185, both PASS. The q descriptor is the only thing that changed: run length, request count, bytes, commands and io_cycle are all identical between the two, so there is nothing left in the mapping to tune. |
| Uneven two-piece split of the K1 q weight to hide the tail (7/8 + 1/8, unlike E30's even split) | the window does not exist: the scheduler serialises the first piece's tail chain after the second piece's DMA instead of overlapping it, so the split buys nothing and costs one command. Static total 42,388 -> 44,411, io_cycle 38,557 -> 39,106. |
| Port K2's 663 B/cycle geometry to K1's weights | **impossible by arithmetic, not by effort.** K2 is fast because it splits its contraction axis into 16 strips of 256 elements and 4,096 = 16 x 256. K1's contraction axis is H = 3,840 = 15 x 256, so a power-of-two strip count gives strips of 1,920 / 960 / 480 / 240 bytes, none a multiple of 256, all of them landing in the measured 280-394 B/cycle misaligned band, i.e. **slower than the 590 the shipped odd-stride layout already gets**. |
| K1 q weight at 64 live slices, stride 4, 32 rows (probe 71073 measured 653 B/cycle) | **unreachable**: the mapping `m![Qs % 2048 / 32, 1 # 4]` is partially live at every ring width, so every Switch over it - and the head gather needs one - is the hang shape. Cross-slice movement has no other path than a Switch or an HBM round trip. The arithmetic also loses before that: the per-slice Tensor-Unit work grows faster than the stream saving, which is what the same trade did to K3 (+16k, Arena 72141). |
| Fuse K1's tail lane sum | static makespan 42,388 -> 43,777 (+1,389) with DMA busy identical at 38,557 over 18 commands: it saves no byte and no command, and the candidate's tail is 187 cycles LONGER than the control's including the program it removes. |
| A zero anchor to fill K1's 2,510-cycle prologue idle window | **+1,626 of measured work** (Arena 73098/73099: 85,679 / 88,529 against 84,020-86,726 over six runs of the shipped tree). It adds 1,168 static cycles of in-order-core stall and makes `input_rms_weight` the first transfer, delaying the whole norm chain. Fourth mechanism against that window, fourth null after E8/E9/E33. |
| The up/gate f4 weight rate as a function of the per-slice piece | **flat at 626-631 B/cycle above ~57 KB per slice**, measured three times independently (probes Q5/Q8 at 128 stride-2 x 60 rows and 256 x 30; p90/p91 on 128 and 64 slices; and Arena 72141 in the kernel itself). The "115 KB -> 680" row of the rate table is the 3.7 MB block SCALE on 32 slices, not a 29.5 MB f4 weight - do not re-derive a weight saving from it. |
| 128 live slices of 60 rows for the K3 up/gate path (to double the per-slice piece) | **+16k of work** (Arena 72141: K3 work 237,068 against 220,127-221,217). The DMA union moved only -2.3k -- **the up stream took 46,934 against 46,934 before, i.e. a 115 KB per-slice piece on 128 stride-2 slices streams no faster than a 57.6 KB piece on 256** -- while TuExec grew 119k -> 175k and StoTab 31k -> 48k, because the LUT decode and the scale contraction now run on half the slices and no longer fit under the stream. The "bigger per-slice piece is faster" reading of the rate table does not hold across a change in the live slice count. (The build also failed numerics, max\|delta\| 0.94, but the timing already settles it.) |
| Broadcast K1's two norm weights from one slice instead of loading them onto their 8 and 4 live slices | **+2k** (Arena 72140, K1 work 87,659 against 84,904-86,522). `q_rms_weight` and `k_rms_weight` are NOT replicated loads: `QVecSlices` has 8 live slices and `KVecSlices` 4, so each is already 16 or 8 pieces of 256 B, within 40 static cycles of the two-descriptor floor. Reaching those slices from one source needs a ring of 256, and a ring-256 broadcast of a 512-byte row costs 256 x 16 = 4,096 Switch cycles. |
| Chunk-slice broadcast of K1's `x` and `input_rms_weight` (the genuinely 256-descriptor loads) | that is `normalize_replicated_from_chunks`, i.e. `Broadcast1 { slice1: 32, slice0: 8 }` = ring 256 with 8 live sources = the hang shape. Do not build it. |
| Intra-slice reduce whose reduce axis shares the Packet with a non-reduce axis | silently wrong, no diagnostic: the packet clipper's valid_size is a contiguous prefix, so the other axis is dropped (ch. 31 "Inexpressible Patterns", worked counter-example `reduce_wrong_mixed_packet`). The reduce axis must OWN the Packet. |
| A partial packet reduce (4 lanes -> 2) | does not exist on the Vector Engine: the packet reducer is either pass-through (`OutPacket = Packet`) or a full collapse to `m![1 # 4]` (ch. 30). All ~30 shipped reduces are one of those two. |
| `vector_widen_concat` with `Time2 != Time / 2` | the contract is two 4-way flits in, one 8-way flit out (ch. 31 "Downstream 4-Way Operations"); declaring half the time steps writes twice the declared bytes, i.e. past the tensor extent. |
| Fold the K3 lane-sum pass into 8-row flits (`narrow_split` -> reduce over `Dummy2` -> `widen_concat`, no Transpose stage) | compiles and the index algebra reads correct, but Arena jobs 72116 and 72128 both returned K3 max\|delta\| **1.174** against a 0.01 gate (mean 0.178, 9.8% within tolerance) while K1 and K2 passed unchanged: after a reduce consumes the split digit the pipeline re-digits to a dense shape, and the committed order is not the H order the declared type claims. The shipped one-value-per-packet form with the Transpose stage costs 1,240 static cycles and is correct. |
| Port K2's 256 B geometry to K1 or K3 | factor 15 — a proof, not a measurement |
| Split one DMA command to engage more engines | inexpressible (E28); and the compiler already uses all 8 (E32) |
| Overlap two big transfers | the queue is strictly serial; measured three times (K1 T4, p97/p98, mlp19e) |
| Fill K1's prologue idle window | E8, E9, E33 — three mechanisms, three nulls; it is core-issue-bound |
| Add a mid-kernel cross-cluster sync to absorb skew | void by construction: kernel time = skew + work either way (E7) |
| Uneven split between the two clusters | the mapping algebra expresses only even partitions |
| Merge small HBM tensors into one transfer | a transfer cannot span two HBM tensors — K1's 11k small-load floor |
| Stage the block scales as one contiguous read | E31: **+83,155 cycles (+33%)** |
| Shorten the K2 tail with a new sync point | E29: **+18,075 (+41%)** |
| Overlap the K1 q tail with the stream | E30: +9,548; per-slice piece fell 30.7 → 15.4 KB |
| K3 down as one transfer instead of three | E22: +7,900, lost 4/4 pairs |
| Fewer live slices for a fatter piece (K1 16 rows/128 slices) | E15/E16: stream −8,286, contraction +3,137, gather +1,016 — a wash |
| Halve the K3 VRF scale operand | E18: +6,200, lost 3/3 pairs |
| bf16 × bf16 instead of the hi/lo split | E19: K1 113,213 vs ~100,000 |
| Split k and v into separate tensors | E20: accepted, then **reverted** on pooled samples |
| DMN permutation (K1 query, K3 down) | E23–E26: stream +15% but gather 700 → 3,301; the family is empty |
| Odd slice stride for channel spread | E27 **falsified**: −16% where +36% was predicted |
| Fuse the RMS sqrt into the mean-square pass | E14: one narrow/widen transition per pass |
| Decode NVFP4 by bit manipulation | `f4e2m1` has no `ContractionCast`; bit logic is post-contraction |
| Weight-stationary / use the spare TRF | matvec over a single token — no reuse exists |
| `begin_interleaved` to pair two HBM transfers | both arguments are already-in-DM views |
| HBM channel / bank / row tuning | all three audited clean; channel spread is anti-correlated with rate |
| K3 down as 15 rows × 7,680 columns | E3: four compiler rules force both lanes into the output, doubling the exchange |
| Splitting K1's q into 480 B strips | E12: 377 B/cycle, K1 110,008 (but it **halves q error** — see below) |

---

## 6. Still open

1. **Align evaluation with the official metric.** Seven independently seeded runs
   and a per-kernel median are now verified in official logs. The historical ~7%
   single-draw advantage does not establish an engineering improvement. Revisit
   decisions made chiefly for lower minimum cycles rather than better medians.
2. **K3's block scales on fewer live slices — the best-priced open item.** Probes p92/p93/p94
   moved the same 3.7 MB onto 512 / 128 / 32 slices for **9,400 / 8,400 / 5,400 cycles**, and a
   115 KB per-slice piece reached **680 B/cycle**. Against today's 10,093 cycles per tensor ×3,
   that is a real target. It is *not* contradicted by E31 (+83,155): E31 bought the geometry by
   adding ~11 MB of DM writes plus an ~11 MB DM→DM relayout, which is exactly the DM-side price
   the governing rule forbids. The open question is whether the same geometry can be reached
   without that staging pass. ⚠ Redistributing makes the Switch contend with DMA for the DM port
   (see mlp20), so it must be measured, not modelled.
3. **K3's down with 3,840 B runs.** Probes p95/p96 measured **46.3k / 49.5k against 65.6k today**
   — a −16k…−19k prize, the largest single measured opportunity left. E3 documents what blocks
   it: four compiler rules force both `[hi, lo]` lanes into the output, doubling the cross-cluster
   exchange (ring-128 → ring-256 gather +4.5k, doubled split +2k, doubled reorder +1.7k), netting
   about −4k, inside the noise. **Worth revisiting only if the gather cost can be removed.**
4. **K3's scratch store descriptor count** (`ops.rs:297`, 4,535 exposed cycles at 6.8 B/cycle).
   The only legal consolidation is a fully-live ring gather, which needs `DownRowsHalves`
   (`mlp13.rs:33`) changed from `m![H / 30, L % 7680 / 3840]` to `m![L % 7680 / 3840, H / 30]` so
   the inter-slice reduce leaves 128 *contiguous* live slices instead of stride-2.
   ⚠ **E25 already tried exactly that digit order and was rejected by the hardware:**
   `VRU reduce axes must be innermost in Partitioning`. Check that before building anything.

   Do **not** extend the same idea to up/gate: probes p90/p91 put `up` onto 128 and 64 slices per
   cluster for 47.9k and 47.5k against 512 slices — the f4 rate is **flat above 57 KB/slice**, so
   there is nothing there regardless of ordering.
5. **Cover `H` with four powers of two** — `3840 = 2048 + 1024 + 512 + 256`. Each block *is* a
   power of two so each individually satisfies all three layout rules; the factor-15 wall only
   bites when demanding one uniform layout across the axis. Upside on the query alone:
   15.7 MB at 635 instead of 456 B/cycle = 34,460 → 24,769, about −9.7k. Cost: three extra
   transfers (1–2k each after scheduler hoisting) plus a real combining problem — the four blocks
   reduce onto 32/64/128/256 live slices, so their partial sums must meet before the head gather.
   Net plausibly −5k, could easily be zero. **Never measured.** Two cost models exist and both
   should travel: the per-transfer one above, and a second that prices it as "four contraction
   programs instead of one". Block geometry: A 2048 cols / 8 strips / 32 row groups / 64 rows per
   slice / 16,384 B / 64 runs; B 1024 / 4 / 64 / 32 / 8,192 / 32; C 512 / 2 / 128 / 16 / 4,096 /
   16; D 256 / 1 / 256 / 8 / 2,048 / 8. They sum to the same 30,720 B per slice K1 reads today.
6. **E35 via a custom `SwitchConfig`.** The fat-slice layout (32 contiguous rows, 122,880 B/slice,
   64 live) is validated — k, v, K2 and K3 passed on every run. Only the gather's ring-minor
   delivery blocks it. Matching ring-minor by reshaping would require four blocks of 8 rows
   strided by 64, destroying the contiguous run that is the whole point. A custom Switch config is
   the remaining legal route. E35's own pooled verdict prices it at **+0.04 score under median
   scoring, ≈0 under best-of** — cheap-only-if-easy.
7. **A numerics reserve, if a later round tightens tolerance.** Splitting K1's query into eight
   480-column chunks halves q error (0.03125 → 0.01562) because eight f32 partials accumulate
   more accurately than one 3,840-term contraction. It costs speed (E12) but the accuracy is free
   if the geometry is ever wanted. Same for bf16 contraction (E19): exact, halves the error,
   costs ~13k cycles. Also flagged for Stage 2: **E34** — it was cycle-neutral (+474) but halves
   the partials' DM footprint and has fewer rounding events at identical measured error.

8. **`FORUM_QUESTION.md` was never answered, and nothing records whether it was even posted.**
   Its five questions are still live and each one gates an open item above: (a) which tier binds,
   given 709 B/cycle measured above the supposed 683 ceiling; (b) the API behind "the kernel
   writer can also specify an engine explicitly", and whether a DMA destination covering a slice
   *sub-range* is expressible at all; (c) what actually separates fast from slow streams — the
   question's own framing notes that run length does *not* order them (a 57,600 B run and a 256 B
   run are both at the ceiling), which is the tension that produced the bytes-per-slice rule;
   (d) whether a published cycle-estimation model exists; (e) whether DM read+write should be
   treated as *strictly* more expensive than any HBM-side geometry it can buy. The text is in
   `git show bb6b08c:FORUM_QUESTION.md`. Asking is cheap and would close several items at once.

### Estimated floors, and what blocks each

| stage | now | floor | what blocks it |
|---|---:|---:|---|
| K1 weight stream | 62.9k | ~53k | needs 61 KB/slice; the contraction then gives it all back (E15) |
| K1 ten small loads | 11k | **11k** | one transfer per HBM tensor — a hard floor |
| K2 weight stream | 24k | 23k | already 95% of the ceiling |
| K3 up/gate | 94k | **94k** | already 700+ B/cycle |
| K3 down | 65.7k | ~50k | needs a 3,840 B run; packet/tile rules (see open item 3) |
| K3 block scales | 30.3k | ~18k | needs fewer slices; Switch then contends for the DM port |
| cluster skew | 0.6k–18k | — | a skewed partition is inexpressible |

Two independent whole-kernel floor estimates disagree slightly and both are recorded:
K1 ~86k (from the stream model) against 89.5k / K2 33.4k / K3 237k (from the stage table).
8× requires `K1·K2·K3 ≤ 7.3e14`, e.g. 90k · 38k · 214k.

### The 8.0 question

7.2050 → 8.00 needs 11.1% overall, ~3.6% off every kernel. Against that: K2's stream is at 95% of
the DM write ceiling, K3's up stream at 91%, and the byte counts are fixed by the model — every
weight byte is read exactly once, so only *rate* and *transfer count* are optimizable.

Two independent ceiling estimates agree: **8.09× if all three kernels hit their estimated floors
and cluster skew were zero; 7.30–7.31× using the smallest values ever actually measured
together.** The gap between them is entirely skew and kernel tails.

38 experiments produced four accepted changes. The honest expectation from further code work is
single-digit tenths of a point, and the distribution of grader draws is worth more than any of
them.

---

## 7. How to measure

```sh
python3 scripts/bench_pair.py CTRL <binA> CAND <binB> -n 8   # alternating pairs — the real test
python3 scripts/bench_arena.py <tag> -n 6                    # pooled runs → OPTIMIZATION_RESULTS.csv
cargo furiosa-opt compile <kernel> --dump-schedule s.json \
  && python3 scripts/analyze_schedule.py s.json              # free static pre-check
python3 scripts/variant.py show                              # source-rewriting knobs
```

Build and submit:

```sh
cargo furiosa-opt test --release --test test_kernels --no-run   # 15 kernel .bin, include_bytes!'d
```

Three files are uploaded and **the controller keeps each file's basename**, so they must be
staged as exactly `remote_entrypoint.sh`, `test_runtime`, `fixtures.safetensors` — submitting
`target/release/deps/test_kernels-<hash>` directly fails with exit 126.

**Arena.** URL `https://arena.furiosa.ai` is compiled into the binary as the default, so bare
`furiosa-arena health/list/submit` works and `$FURIOSA_ARENA_URL` is only an override.
(`ARCHITECTURE.md:201` still implies the variable is required; it is not.) Limits: **at most 2
active jobs**, `--timeout` capped at **70**. The queue is shared with other teams.
**`furiosa-arena rerun <job>` re-runs without rebuilding** — the cheapest way to draw samples.

> ⚠ **`scripts/cpu_verify.sh` does not work on this tree.** The shipped control fails identically
> to every variant: `panicked at npu-mapping-impl/src/sequencer/smapping.rs:627: modulo must
> divide the size`, then SIGABRT. The CPU backend's sequencer cannot express the mappings
> `mlp13.rs` / `f8split.rs` / `sliding/tail.rs` use. **A cpu_verify failure is not evidence
> against a candidate**, and there is no local numerics gate — every numerics-affecting candidate
> costs at least one Arena job, which reports max|Δ| and PASS/FAIL per kernel alongside cycles.

**Tracing.** `TUC_PROFILE_LEVEL=trace` alone prints only `Task` spans. Named per-engine spans
(`DMA`, `Renegade::TuExec`, `StoVrf`, `StoTrf`, `Core`, `Cluster`) need `record_str`/`record_debug`
capture added to `Collector` in `tests/test_kernels.rs` — and `tests/` is **frozen**, so that
patch lives only in a measurement-only copy. Trace level inflates the kernel ~12%: read
**structure** from a trace, never totals. Profiling is active only for
`TUC_PROFILE_LEVEL ∈ {info, debug, trace}`; unset gives no `cycles` line at all, while
set-but-no-spans prints `cycles=none observed` — a different failure mode worth distinguishing.

**Grading tolerances.** PASS requires `|Δ| ≤ atol + rtol·|expected|` for **every element** — not a
percentage — and any non-finite value fails immediately.

| kernel | atol | measured max\|Δ\| | margin used |
|---|---:|---:|---:|
| K1 | 0.04 | 0.03125 | 78% |
| K2 | 0.05 | 0.03125 | 63% |
| K3 | **0.01** | **0.00781** | **78%** |

K3's is the thinnest. Do not spend it for cycles before Stage 2 runs.

**Edit surface.** Only the function bodies in `src/ops.rs` and code under `src/device/` may be
changed. `#[device]` signatures, `axes.rs`, `lib.rs`, `tests/` and `Cargo.toml` are frozen, and
shared code must still compile **all 15 kernels**, not just the three graded ones.

**Toolchain traps.** A `cp -a` of the tree breaks cargo's fingerprint, so the first build is cold
and the build script re-downloads the prebuilt `.a`; running several copies in parallel corrupts
the download (set `FURIOSA_MAPPING_IMPL_LOCAL_PREBUILT` and `FURIOSA_OPT_LOWER_IMPL_LOCAL_PREBUILT`
at an existing copy). Each run pushes 188.8 MB host→HBM, entirely **outside** the measurement
window. `scripts/rngd/remote_entrypoint.sh:34` is the only place the test binary's real exit code
reaches the Arena job. The grader occasionally returns "INFRASTRUCTURE ERROR — this is not your
code"; that is their side.

---

## 8. Traps in the tree

**Dead code that still compiles** and would mislead a reader:

- `src/device/shared/mlp.rs` (504 lines, the old feed-forward path — K3 calls `mlp13`; only still
  declared at `shared/mod.rs:3`)
- `src/device/sliding/rmsnorm.rs`, `src/device/sliding/rope.rs` — unchanged from baseline, but K1
  now calls `sliding::tail::*`
- in `tail.rs`: `park_rope_tables`, `rope_tables_from_hbm`, `reload_rope_tables*`, `table_anchor`
  — only `kv_anchor` is live
- in `projection.rs`: `contract_output_96` / `contract_output_24`, a two-piece output projection
  **rejected by the "owner mismatch" checker**, never called
- in `mlp13.rs`: the old 16+14 gate path (`load_gate16`/`load_gate14`, `GateChunk16/14`,
  `GateParts16/14`), left over after E5

**Stale comments in live files:** `projection.rs:306-307` describes a two-piece weight layout that
does not match the code below it; `f8split.rs:128` says "three passes" where there are two Tensor
Unit programs; `mlp13.rs:12-13` says the lane sum lives inside the scale pass, whereas there are
four scale passes *plus* a separate lane-sum pass (`mlp13.rs:20-21` is the correct comment).

**Numbers that disagreed across the old documents**, resolved here:

- K3 down is **449 B/cycle**, not 462; the three block-scale tensors are **30,279 cycles**
  (3 × 10,093), not 36k.
- The DM ceiling of 683 B/cycle is a **working hypothesis**, contradicted by a 709 B/cycle trace
  of the same K3-up stream that the table lists at 623.
- **K3's best is quoted as both 249,215 and 250,215**, and separately three different "best K3
  Arena" values circulated unreconciled (246,637 / 246,677 / 248,565). Treat **250,215** as the
  grader record and the rest as unverified.
- `OPTIMIZATION_LOG.md` recorded **E35 as ADOPTED**, but it was never in the tree —
  `QueryRows` is still `m![Qs % 2048 / 8]`.
- **E20 was accepted and then reverted** (2026-09-15) once samples were pooled. The old log's E20
  heading still read ACCEPTED. The shipped tree uses the **paired** k/v tensor. Stated once here
  so it is unambiguous.
- **Is K3 Tensor-Unit-bound after `up` lands?** One session measured ~190k of TU work and used
  that to justify merging 12 LUT programs into 6. E34 later removed an entire TU program for
  +474 cycles — i.e. TU work is free under a DMA queue busy 211k of 253k. Both cannot hold. The
  later measurement should win, but since the earlier one justifies the current LUT-program
  count, **any change to that count must be re-measured, not argued from either.**
- **E1 was kept for its best draws**, not a faster mean: historical standard deviation
  3.3k → 6.4k (the proposed explanation was removal of a mid-kernel sync).
  The official logs checked on 2026-09-22 confirm **median-of-seven** kernel timings,
  so that original justification needs re-examination. E1 has not been reverted:
  no new controlled measurement establishes that its predecessor is better.
- A stale section in the old `CHANGES.md` described K1's row scales as loaded *after* the gather
  on the head layout. No such code exists — that design **is E9's proposal, which was measured and
  rejected**. The shipped code loads them on the weight layout (`projection.rs:152,180,230-236`)
  and applies them inside the contraction's Vector stage.
- Old per-kernel result tables quoted the board best as 7.1023; the standing entry is **7.2050**.

**Also worth knowing:** `cpu_verify.sh` has never been tried on baseline `ed1db81`, so
"pre-existing breakage, not a regression" is plausible but **unproven**. And the old
`FORUM_QUESTION.md` stream table listed K3's down and block scales at 256 live slices per
cluster; they are **128 × 2**.

---

## 9. The 2026-09-22 measurement round — where every cycle of each kernel goes

Source: Arena jobs 72029 (control tree), 72057 (down-15 merge), 72066/72067 (integrated), all with
`TUC_PROFILE_LEVEL=trace` and the patched harness (`scratchpad/patch_trace_harness.py`). A DMA span
begins at ISSUE time and includes queue wait; the queue is FIFO and shared, and a transfer that only
touches one cluster's slices can overlap one that touches the other's.

### K3 `decoder_feedforward` — 219.8k of work, 91% DMA-bound
up 46.2k + gate 48.2k for 59 MB (616 B/cycle, near the 650-709 ceiling) · down 3 chunks ~54k for
29.5 MB (564 after the 15-row layout, was 458) · **three block scales 32.0k for 11.1 MB (355 B/cycle,
the worst point of the rate curve: 7.2 KB per slice)** · four LUT table loads ~7.6k plus 31k of StoTab
Tensor-Unit time · three global scales ~3.0k · x + rms 5.4k · **DMA idle 28.6k**, of which a 4.3k gap
in the middle (the core issues in static index order; the 21k gate LUT-decode program holds the Tensor
Unit, the next TU program cannot be issued, and the first down-chunk DMA sits after it in that order)
and the rest is the ~21k epilogue.

### K1 `sliding_project_qkv` — 89-90k of work, 89% DMA-bound
prologue x 2.7k + rms 2.7k (both replicated, ~1.1k of fixed cost each) · k+v 15.7 MB in 35.4k
(443 B/cycle; the V2 odd-stride layout measured 27.9k but re-ordered the small loads and lost more than
it gained) · **ten small loads 10.3k** (2 norm weights, rope_offset, 4 rope cos/sin gathers — the gather
lowers to one instruction per cluster — and the q/k/v row scales; ~1k of fixed cost each, 2-3 overlap) ·
q 15.7 MB in 26.4k with the odd-stride layout (595 B/cycle, was 485) · k/v cache scatters 4.8k ·
tail after the last q byte 11.2k.

### K2 `sliding_attention_output` — 21-28k of work
x 1.6k · o_weight 15.7 MB at **663 B/cycle, the fastest stream measured on this chip** · the weight is
split into two plain `to_dm` pieces so only the small piece's contraction is exposed; the 96+24 split
streams the 24-row piece at 468 B/cycle, the 112+8 split streams the 8-row piece at 308, and the
exposed contraction shrinks with the piece — the optimum is a small piece of 8-24 rows, worth ~1k ·
the three per-row loads (channel scale, norm weight, residual) overlap the contraction and are free ·
per-cluster half norm: 32 sums of squares through an 8 KB scratch, then each cluster writes its own
1,920-row half of the residual stream.

### The cost model that now explains every transfer

`T = F + bytes / R` with **R ~ 707 B/cycle marginal** and **F ~ 1.9-2.7k per DMA command**, F tracking the
descriptor count (a 1-2 descriptor gather costs 500-960; a 256-descriptor load of a few KB costs 1.5-2.0k).
Fitted on five K2 weight pieces of 120, 112, 96, 24 and 8 rows per slice, then checked against K1 and K3.
Two consequences that were both measured: splitting a stream into k pieces costs `k x F` and buys nothing
(K2, -2.4k by going back to one piece), and a geometry that leaves the per-slice contiguous piece small
pays twice (K3's block scales: 7.2 KB per slice, 10.3k for 3.7 MB against 7.1k predicted).

### A `Broadcast1` delivers exactly ONE input time step

Discovered 2026-09-23 after two silent wrong-answer runs (Arena 80505, 80563: k max|delta| 3.29688,
v exactly 0.0 on every dimension outside input time step 0). **The ring digit must be the only Time
digit the switch introduces; any input `Time > 1` is silently truncated to index 0, with no
compile-time diagnostic.** The payload belongs in the **Packet**: the shipped
`rmsnorm::gather_replicated` pushes a 1,920-byte packet through a ring-8 `Broadcast1` and lets
`collect` cut it into flits afterwards. Every Switch that has ever produced correct results in this
tree has `Time = 1` at the switch; the one that does not (`normalize_replicated_from_chunks`,
`Time = 30`) is the function already flagged "has no callers, do not call".

Related, from the same investigation: the commit stage computes addresses from the **element
mapping**, not by writing time steps sequentially - rewriting a gather's element mapping into stream
order and moving the reordering into the consumer's fetch returned a bit-identical device result.

And a latent hang already in the tree, not introduced by any change: `full/projection.rs:153` has
`Broadcast1 { slice1: 64, slice0: 4 }` over a 64-live-of-256 input, which is the partially-live shape.
`full_project_qkv` is not one of the three graded kernels, so it has never run.

### The rule that explains every DMA rate we have measured

**A transfer runs at roughly half speed when a slice's piece does not start on a 256-byte boundary**,
because the lowering cuts every per-slice run into 256-byte requests and a misaligned slice makes each
one straddle two blocks. Sorted by that test, all eighteen measured rates fall into two clean groups:

| transfer | bytes per slice | multiple of 256? | B/cycle |
|---|---|---|---|
| K3 up/gate weight | 57,600 | yes | 626 |
| K1 q weight, odd-stride | 8 runs of 3,840 | yes | 590 |
| K2 o_weight | 120 runs of 256 | yes | 663 |
| K3 down weight, 15-row | 15 runs of 3,840 | yes | 546-564 |
| **K3 up/gate block scale** | **7,200** | **no, 32 over** | **355** |
| **K3 down block scale, 240-B rows** | **7,200** | **no** | **280-293** |
| **K3 down block scale, 480-B rows** | **7,200** | **no** | **360-377** |

Every aligned case is 449 or better; every misaligned case is 280-394. The three K3 block scales are the
only misaligned transfers left in the tree and they cost about **15k cycles of K3**.

**And they cannot be fixed.** The live slice count must be divisible by 64 - the lowering splits the
slice axis into per-PE groups of 64, and a 240-slice layout dies with `visa: internal compiler error:
split (axis_index: 0, inner_size: 64) is not valid on shape([L_32=240])` (tested 2026-09-22, which also
explains the older "padded 240-slice layouts ICE" note). Each cluster's scale is 1,843,200 B = 7,200
blocks of 256, and 7,200 divided by 64, 128, 192 or 256 gives 112.5, 56.25, 37.5 and 28.125 - never an
integer. So no legal slice count makes the per-slice piece 256-aligned, for the up/gate scale or the
down scale. This is a property of the tensor shapes in the problem statement, not of our layout.

### A limit that was never real

The working note "TRF 64 KB per slice, 8 KB per lane" is wrong twice. Manual ch. 22: the TRF is
8 lanes x 2 banks x 128 rows x 320 bits = **80 KB per slice**, and fewer active lanes give each lane
proportionally more rows, so there is no per-lane byte cap to design around. The shipped
`UpGateScaleTrf` is 14,400 B in a single lane today, and 28,800 B in one lane compiles. Any design
that was scaled down to fit "8 KB per lane" can be reconsidered.

### What is left, in size order
1. **K3 block scales, 32.0k at 355 B/cycle.** Every route that keeps 256 slices of 30 rows and widens the
   read with a Switch is now closed by two device hangs (section 5). The remaining route is to change the
   slice mapping itself: 128 live slices at stride 2 owning 60 rows each, which makes the weight piece
   115 KB (~680 B/cycle instead of 625) and the scale piece 14.4 KB (~481 instead of 355) with **no Switch
   at all**. Cost: the LUT decode and the scale contraction run on half the slices, so their per-slice time
   doubles and must still fit under the stream. Design in `scratchpad/DESIGN_K3_60rows.md`. **-12k.**
2. **K3 down weight, ~54k at 564 B/cycle**, because each slice reads 15 runs of 3,840 B at a 7,680-B stride
   (its cluster's column half of each row). Splitting the down projection by output ROWS instead would make
   each slice read 115 KB contiguous, at the price of a cross-cluster exchange of `g`. Design in
   `scratchpad/DESIGN_K3_downrows.md`. **-8k.**
3. K3's mid-stream idle gap, still 3,954 cycles in the current tree (trace 72147, 126,094 -> 130,048). The core
   is blocked at the block-scale decode program because the Tensor Unit is busy with the 21k gate LUT decode, and
   the first down-chunk DMA sits after that program in static index order. Moving the load earlier in the SOURCE
   changes nothing (verified again on 2026-09-22: byte-identical static schedule). The one untried lever is to
   split the gate LUT decode into two programs so the scale decode can be issued between them, freeing the core
   ~9k earlier; whether the scheduler interleaves them that way is a coin flip. **-4k if it lands.**
4. K3 epilogue: 18.5k of DMA idle between the last down byte and the scratch store, and 6.9k more after it. Most
   of it is the cross-cluster wait; the rest is the eight-program epilogue chain. **-3k.**
5. K1's ten small loads, 10.3k of pure fixed cost. Only fewer descriptors help: load onto few slices and
   spread with a one-live-source `Broadcast1` (the validated `broadcast_table` shape), never an
   InterTranspose. **-4k.**
6. K1 k/v as two plain `to_dm` tensors instead of two `to_dm_view` tiles into a pair: probe 71073 measured
   591 B/cycle against 463-521. **-2k.**
