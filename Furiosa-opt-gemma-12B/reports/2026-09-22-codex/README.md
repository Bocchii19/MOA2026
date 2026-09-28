# Optimization measurements, 2026-09-22

The requested **8–9× Stage 1 score has not been established**. The most recent
official best verified when this round started was **7.4924**, submission
`4912db55`. The historical public Arena best in the repository is **7.9614**;
neither number is a new result from this round. Arena score estimates use the
previously inferred baseline product `3.754e17` and are not official grades.

## Official result

Submission **`dd0580b2`** completed at **2026-09-22 13:08 UTC** with score
**7.1154**. The skeleton contract, remote build, and all three correctness tests
passed. Each test used **seven independently seeded inputs**, and the grader
reported the **median** cycle count for each kernel:

| Kernel | Median cycles | Seven measured cycle counts |
|---|---:|---|
| K1 | 93,249 | 91,451; 93,131; 92,077; 93,983; 94,141; 93,249; 94,957 |
| K2 | 49,221 | 47,675; 50,481; 51,577; 53,345; 46,519; 47,663; 49,221 |
| K3 | 227,044 | 230,705; 227,044; 231,233; 225,017; 229,812; 225,041; 218,675 |

Every output element passed tolerance. Maximum absolute errors across the seven
inputs were **0.03125** for each of Q/K/V, **0.03125** for K2, and **0.01172** for
K3 (values printed by the grader). These are distinct from the single-fixture
Arena errors below. Full evidence is in `official-candidate.log`,
`official-candidate-status.txt`, and `official-results.json`.

This submission does **not** improve the official best of **7.4924**. That older
submission also used seven runs and medians, as verified in
`official-best-at-start.log`; the older work log's description of a one-run
official grader was incorrect. Ordinary public Arena measurements still use the
repository's single-run test harness and cannot be treated as equivalent grades.
The retained change has a smaller static schedule and passes correctness, but
an official score improvement has **not** been established. The official jobs
were not an alternating experiment, so their difference alone also cannot
isolate the effect of this patch.

## Retained change: paired RoPE arithmetic

`src/device/sliding/tail.rs::rope` interleaves the original and rotated vectors,
uses separate multiplier ALUs for `x*cos` and `rotate_half(x)*sin`, and adds the
two groups at the Clip stage. This removes the f32 `x*cos` DM temporary and its
VRF staging program for both Q and K. The normalization's bf16 boundary and the
final bf16 cast remain in place. The two half-swap programs remain unchanged.

The two VRF operands must use **different** multiplier ALUs. Passing both to one
paired Mul0 operation exceeds the Vector Engine operand slots and fails to
compile. A separate attempt to combine the half swaps also failed compiler
validation and is not included in the retained patch.

| K1 static schedule | Control | Paired RoPE |
|---|---:|---:|
| Makespan, cycles | 42,388 | 42,246 |
| Instructions | 175 | 171 |

Static makespan falls **0.34%**. The exact JSON schedules and source patch are
included here. All **15 device kernels** compile with the candidate.

| Trace pair | Control job | Candidate job | Control K1 hull minus wait union | Candidate K1 hull minus wait union |
|---|---:|---:|---:|---:|
| 1 | 74377 | 74397 | 86,798 | 84,633 |
| 2 | 74421 | 74438 | 88,058 | 86,729 |

Both candidates passed all three kernel tests. Q/K/V maximum absolute errors
were **0.03125 / 0.015625 / 0.03125**, unchanged from control. The median K1
diagnostic drops from **87,428 to 85,681 cycles (2.00%)**, but two pairs are
insufficient to establish a reliable overall score improvement. Raw K1 cycles
were `93915 / 99584` in pair 1 and `91931 / 89322` in pair 2; launch variability
can reverse the apparent result. Trace and ordinary-profile runs must be kept
separate. The machine was shared and jobs queued for several minutes.

## Original harness, ordinary profiling

These runs use the unchanged `tests/test_kernels.rs` and `TUC_PROFILE_LEVEL=info`.

| Pair | Variant | Job | K1 | K2 | K3 | Estimated score | Correctness |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | Control | 74453 | 93,462 | 47,188 | 232,279 | 7.1560 | PASS |
| 1 | Paired RoPE | 74463 | 91,453 | 42,420 | 235,099 | 7.4386 | PASS |
| 2 | Control | 74477 | 112,638 | 45,136 | 230,963 | 6.8378 | PASS |
| 2 | Paired RoPE | 74482 | 97,315 | 46,604 | 234,971 | 7.0625 | PASS |

Median estimated scores are **6.9969 / 7.2506** (control/candidate). The candidate
wins both pairs on K1 cycles and total score, but K2 and K3 also vary despite
their device binaries being byte-identical. Thus these score differences cannot
all be attributed to the source change. The candidate's best ordinary run in
this round is **7.4386**, below the requested target. All four candidate runs,
including the two diagnostic runs, pass correctness.

The final root build's three graded kernel binaries are byte-identical to those
tested here. Its host executable is preserved as `final.bin`; its SHA-256 can
differ because it was built in another checkout. Official submission `dd0580b2`
contains the exact candidate sources; its result is recorded separately.

## Other experiments

| Experiment | Result | Decision |
|---|---|---|
| K2 broadcast each leader within ring 16, reduce all 256 slices, exchange two aligned blocks | Static 26,375 cycles versus control 24,036 | Reject: extra reduction dominates |
| K2 read all partials on one slice per cluster, then broadcast RMS into VRF | Arena 74429: K2 output non-finite; K1 and K3 pass | Reject; no performance credit |
| K3 down-scale decode on sub context | Static 110,260 versus 108,999 | Reject |
| K3 LUT directly into scale TRF | Compiler rejects operand type; explicit identity cast then hits unsupported StoTrf lowering | Reject |
| K3 gate LUT split into 16+14 rows while keeping one weight transfer | Static 110,239; Arena 74399 passes, diagnostic K3 218,314 versus control 218,614/218,878 | Do not retain: small difference does not establish a win |
| K3 full-row packet gather | Compiler rejects padded commit or Collect packet mapping | Reject |
| K1 gather RoPE rows directly onto head slices, broadcast only the zero scheduling anchor | Static 43,627; Arena 74446 passes, K1 diagnostic 90,451 versus paired-RoPE 84,633/86,729 | Reject: slower |

## Reading traces correctly

`scripts/analyze_trace.py` reports the **union** of intervals. Summing overlapping
Cluster spans double-counts waits: historical job 72093 has a K2 wait union of
21,461 cycles, so `51970 - 21461 = 30509`; summing waits instead produces the
incorrect 14,740. The script has regression tests for that exact case.

Even the corrected subtraction is only a diagnostic. Engine spans can overlap
Cluster spans and include dependency waits themselves. `dma_union_cycles` is
elapsed DMA span coverage, not a direct measurement of bytes moving. Neither
quantity replaces the official total cycle metric. The analyzer reports
`cluster_engine_overlap_cycles` to make that limitation visible.

```sh
python3 -m unittest discover -s scripts -p test_analyze_trace.py
python3 scripts/analyze_trace.py reports/2026-09-22-codex/*.spans.txt
```

## Reproduction

The baseline is commit `35b7838`; `paired-rope.patch` contains the candidate's
complete kernel-source change. `metadata.json` records fixture, binary, and
patch SHA-256 hashes. Preserved executables are under
`target/experiments/codex-20260922/`. Files ending in `-trace.bin` use the
diagnostic harness; `control.bin` and `k1-pairedrope.bin` use the unchanged public
test source. The official grader supplies its own seven-run test harness.

Build isolated baseline and candidate worktrees with the repository toolchain:

```sh
git worktree add --detach /tmp/gemma4-control 35b7838
git worktree add --detach /tmp/gemma4-candidate 35b7838
git -C /tmp/gemma4-candidate apply "$PWD/reports/2026-09-22-codex/paired-rope.patch"
# In each worktree, compile all kernels and build the original test harness:
cargo furiosa-opt compile
cargo furiosa-opt test --release --test test_kernels --no-run --message-format=json
# Preserve each reported executable before another build, then alternate them:
python3 scripts/bench_pair.py CONTROL /path/control.bin PAIRED_ROPE /path/candidate.bin -n 8
```

For diagnostic traces, apply `scripts/diagnostics/test_kernels_trace.patch`
**only to an isolated worktree**, rebuild its test binary, and submit the staged
entrypoint/binary/fixture with `furiosa-arena submit --env TUC_PROFILE_LEVEL=trace`.
Keep the staged filenames `remote_entrypoint.sh`, `test_runtime`, and
`fixtures.safetensors`, and use `--entrypoint remote_entrypoint.sh --timeout 70`.
The root repository's grader tests and all public device signatures are unchanged.
