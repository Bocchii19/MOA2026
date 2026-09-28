# Follow-up toward the 8.9–9.0 official score target

The user target remains **8.9–9.0**, verified by the official grader with correct
outputs. It has **not** been achieved. The official best verified before this
round is **7.4924**; the previous paired-RoPE submission scored **7.1154**.
This report describes ongoing experiments, not a completed optimization claim.

## Measurement method

`scripts/bench_batch.py` runs seven alternating A/B pairs inside one Arena job.
Each invocation is a fresh process, using the unchanged public test executable
and the same public fixture. All samples, including the first and outliers, are
retained. Failed/incomplete runs invalidate the performance summary. This reduces
allocation differences and queue overhead, but is **not equivalent** to the
seven independently seeded inputs of the official grader. Estimated scores use
baseline product `3.754e17` and are labelled accordingly. `--resume DIR` observes
the same recorded job; it does not resubmit after an observation failure.

## Hardware evidence so far

| Variant | Arena job | Pairs | Relevant median cycles, control → candidate | Paired candidate wins | Decision |
|---|---:|---:|---:|---:|---|
| K2 gather scalar partials locally before HBM exchange | 76120 | 7 | 49,063 → 51,311 | 1/7 | Reject; slower, all correctness checks pass |
| K3 fuse four partial sums in the epilogue | 76121 | 7 | 232,615 → 232,577 | 2/7 | No demonstrated gain; paired median difference is +4,018 cycles |
| K2 gather final output before HBM write | 76135 | 7 | 50,504 → 47,015 | 6/7 | Replication needed |
| K2 output store, reversed variant order | 76145 | 7 | 47,272 → 48,669 | 2/7 | Mixed evidence; not adopted; trace diagnosis pending |

K1 and K3 device binaries are byte-identical between the K2-store variant and its
control (`k2-store-kernel-hashes.json`). They still show substantial timing
variation, so the entire estimated score difference cannot be credited to K2.
Raw logs, hashes, run order, all samples, and summaries are in the job folders.

## Candidate: aligned K2 output store

The final K2 output originally writes sixteen 240-byte pieces per cluster.
One same-cluster DM gather makes a contiguous 3,840-byte half, followed by one
aligned HBM write per cluster. Arithmetic is unchanged. The static schedule
changes from **24,036 to 23,336 cycles**: an 882-cycle DM gather replaces a
2,040-cycle HBM store with a 458-cycle HBM store. `store-gather.patch` records
the exact two-line source change. Hardware validation and replication govern
adoption; the static model alone is not the score.

## Rejected or diagnostic compiler probes

- **Direct DM cluster exchange:** SDK method exists, but a fresh compiler probe
  fails with `TensorDmaClusterSwap lowering is not yet implemented`. This is a
  current toolchain limitation, not an architectural impossibility proof.
- **K2 bf16 contraction:** decoding f8 weights through a Fetch lookup and keeping
  the input in bf16 removes the two-lane split, but raises the static schedule
  from 24,036 to **28,552**. The exposed projection grows from 1,385 to 5,063
  cycles. No hardware measurement warranted for this implementation.
- **K2 inverse RMS through log/exp:** static schedule remains **24,036**, so no
  demonstrated benefit; numerical behavior is unverified and this is not adopted.
- **Existing K2 switch RMS exchange:** adds 387 static cycles to the output-store
  candidate; retain the original local exchange instead.

K1 alternatives and K3 local split/gather experiments are still running in
isolated worktrees. Frozen root tests, dependencies, and public signatures remain
unchanged. Experiments are not automatically merged into the root worktree.
