# scripts/opt — measurement tools

All tools read `env.sh`; work products go to `$OPT_WORK` (session scratchpad by default).

| tool | what it does |
| :-- | :-- |
| `worktree.sh <name>` | fresh worktree of `main` on branch `exp/<name>` |
| `static.sh <wt> [ops::fn…]` | static makespan / DMA / Main / Sub from `--dump-schedule` (~40 s) |
| `sched.py <json>` | every DMA and every ≥300-cycle pass of a schedule, in time order |
| `measure.sh <wt> <tag> [--spans]` | 9 seeded runs per kernel on Arena; median, min, p25 and scores |
| `crit.py <log> <test/runN>` | from a `--spans` log: time split into DMA-only / compute-only / both / idle |
| `harness_overlay.py` | the measurement-only harness patch `measure.sh` applies |

Read `WORK_LOG.md` before trusting `static.sh`: for K3 the static makespan is anti-correlated with
hardware, so it can reject an idea but never accept one.
