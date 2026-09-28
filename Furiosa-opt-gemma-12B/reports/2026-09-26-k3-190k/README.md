# K3 optimization toward 190,000 cycles — 2026-09-26

**190,000 cycles was not reached.** The final implementation reduces K3 from
208,552 to **203,258 cycles** in a paired K3-only run over 31 distinct seeds
(-5,294, **-2.54%**). In the complete three-kernel harness, it reduces K3 from
208,574 to **203,651 cycles** (-4,923, **-2.36%**). These are separate execution
contexts, measured in `info` mode, not compiler estimates or trace-mode minima.

Branch: `opt/k3-190k`. Control: `7f19a0a`. The implementation change is confined to
`src/device/shared/mlp.rs`; public kernel signatures, tolerances, input generation,
and the repository's harness/fixtures are unchanged.

## Final implementation

- Partition Down by its 7,680-element K-halves across clusters. Each cluster consumes
  the GeGLU values it already computed. Two Main switch passes replicate/all-gather
  the FP8 hi/lo activations, followed by a Sub TRF load. The GeGLU HBM store/reload
  and its synchronization disappear.
- Fuse FP4 lookup and contraction for the first 12 Down rows per slice, save the
  block-16 partials as BF16, and apply exact BF16-converted block scales with a
  diagonal contraction. Retain the separate three-row final tile to overlap its
  weight DMA with the first tile's work.
- Gather both clusters' H-sized partials to cluster 0 and add the two halves with
  one intra-slice reduction, avoiding a VRF staging pass on the epilogue path.
- Apply Up's block scales through the same BF16 diagonal method already used for
  Gate. This is nearly neutral in isolated K3 comparisons, but improves the full
  three-kernel harness by 988 and 831 cycles in separate paired jobs, with all six
  round deltas negative. The final selection uses that full-harness evidence.

BF16 partials introduce rounding in Up and Down. Validation uses the existing
reference and unchanged `atol=0.01`, `rtol=0.01` criterion. The FP8 hi/lo activation
representation and scaling are preserved. No computation is skipped based on
fixture values. Maximum reported K3 absolute error in the final 31-seed sweep:
**0.01172**, with every element passing the combined absolute/relative criterion.

Static K3 makespan: **107,226 → 103,020**, DMA commands **28 → 23**.
The repository source matches the measured combined variant and the extended
31-seed build after newline normalization; repository CRLF is preserved.

## Hardware validation

All comparisons below use `FURIOSA_OPT_PROFILE=info`, fresh processes, matching
fixtures for both arms, and the existing maximum per-cluster profile window metric.

| Job | Comparison/context | Control K3 | Candidate K3 | Delta | Checks per arm |
|---:|---|---:|---:|---:|---|
| 99246 | Local hand-over + Down diagonal, K3 only | 208,660 | 204,158 | -4,502 | 9 seeds × 3 rounds |
| 99265 | Same comparison, reversed B/A order | 208,585 | 204,318 | -4,267 | 9 seeds × 3 rounds |
| 99284 | Add the one-pass epilogue reduction | 204,413 | 203,491 | -922 | 9 seeds × 3 rounds |
| **99310** | **Final combined implementation vs baseline, all kernels** | **208,574** | **203,651** | **-4,923** | **9 seeds × 3 rounds × 3 kernels** |
| 99314 | Add Up diagonal to the reduction variant, K3 only | 203,549 | 203,430 | -119 | 9 seeds × 3 rounds; mixed round signs |
| 99329 | Variant without Up diagonal vs baseline, all kernels | 208,507 | 205,152 | -3,355 | 9 seeds × 3 rounds × 3 kernels |
| 99339 | Original vs cleaned variant without Up diagonal, all kernels | 205,274 | 205,125 | -149 | 9 seeds × 3 kernels |
| 99341 | Variant without Up diagonal vs baseline, K3 only | 208,984 | 203,696 | -5,288 | 31 distinct seeds |
| 99348 | Add Up diagonal, all kernels | 204,781 | 203,793 | -988 | 9 seeds × 3 rounds × 3 kernels |
| 99367 | Same full-harness comparison, reversed B/A order | 204,707 | 203,876 | -831 | 9 seeds × 3 rounds × 3 kernels |
| **99368** | **Final combined implementation vs baseline, K3 only** | **208,552** | **203,258** | **-5,294** | **31 distinct seeds** |

All listed correctness checks passed. In final-vs-baseline job 99310, candidate K1/K2
medians are 69,599/35,988 versus control 69,629/35,807. Their implementations are
unchanged; those small timing differences are not claimed as gains or regressions.

The nine-seed rounds repeat nine configurations; they are not 27 distinct seeds.
The 31-seed sweep adds 22 configurations. This is local hardware validation, not a
hidden-grader result. The cleaned intermediate variant's schedule was also compared
with `localreduce`: it is identical after diagnostic descriptions/debug names are
removed, and its hardware cleanup check is job 99339.

## Other experiments

All numerical experiments below passed the tested reference checks; slower variants
were rejected. Deltas use the arm explicitly named here, not always the original baseline.

| Variant / job | Result |
|---|---|
| All 15 Down rows in one diagonal path, 99219 | +18,501 vs baseline |
| 12 Down rows diagonal, scale conversion on Main, 99224 | +4,905 vs baseline |
| Same, conversion on Sub, 99223 | +3,825 vs baseline |
| K-split local hand-over on Sub, 99231 | +5,163 vs baseline |
| Shared Up/Gate weight buffer and one lookup/contraction, 99249 | +12,708 vs baseline |
| Local hand-over plus all 15 Down rows, 99268 | +5,035 vs the Main hand-over variant |
| Redistribute GeGLU into 120-value shards first, 99273 | +4,380 vs the Main hand-over variant |
| Single padded Broadcast01 hand-over, 99292 | +7,103 vs the Main hand-over variant |
| Add diagonal scaling to Up alone, 99294 | -694, one of three round deltas positive; followed by the full-harness checks above |

Additional full-packet packing attempts either failed SDK commit/transpose/TRF
layout checks (`compile/`) or did not improve the static schedule. Patches are
retained. Job 99221 is invalid: its entrypoint tried to copy executables into the
worker's read-only `/tmp`; no kernel result from it is used. The runner was fixed
to stage executables in the job directory, then the experiment was rerun as 99223.

Trace jobs 99230 and 99270 diagnose scheduling only. Down still waits on weight/scale
traffic and FP4 lookup/contraction. Shortening the hand-over, merging all Down rows,
or moving work between Main/Sub often moved the limiting wait elsewhere. None of
the measured correct variants reached 190k. The final gap is **13,258 cycles** in the
isolated check, or **13,651** in the full-harness check. This observed gap does not
prove that 190k is impossible.

## Evidence and reproduction

- `results.json`: medians, all samples, validity, per-round medians, job IDs, profile
  mode, execution order, binary/fixture hashes.
- `raw/<tag>/`: executed entrypoints, metadata, logs and status. Trace logs are
  separate from info-mode comparisons.
- `patches/`: gzip-compressed patches against `7f19a0a`. `finalcandidate.patch.gz` and
  `combined_original.patch.gz` describe the final source; `localreduce.patch.gz` is the
  intermediate version without Up diagonal. Use `gzip -dc <patch.gz> | git apply` in a baseline worktree.
- `compile/`: compressed baseline/final schedules and rejected lowering diagnostics.
- `source-hashes.json`: final source hashes in repository CRLF and normalized LF form.
- `run_pair.py`: the exact runner, including this session's scratch paths. Binaries
  and generated fixtures are not committed.

Fresh comparison using the existing tooling (SDK 0.8.1, `nightly-2026-05-01`,
configured Arena access):

```sh
git worktree add --detach /tmp/k3-control 7f19a0a
git worktree add --detach /tmp/k3-candidate opt/k3-190k
ONLY=decoder_feedforward RUNS=31 scripts/opt/ab.sh /tmp/k3-control /tmp/k3-candidate k3-recheck 1
RUNS=9 scripts/opt/ab.sh /tmp/k3-control /tmp/k3-candidate k3-all-recheck 3
```

Apply measurement overlays only in the detached worktrees. Do not commit their
modified fixtures or harness files to the implementation branch.
