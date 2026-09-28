#!/bin/bash
# cpu_check.sh <worktree> <test-name>  -- numerics of one kernel on the host CPU backend (no NPU,
# no cycles). ver16's K1 uses a mapping the emulator rejects, so run K2 and K3 only.
set -u
. "$(dirname "$0")/env.sh"
WT="$1"; ONLY="$2"
python3 "$(dirname "$0")/harness_overlay.py" "$WT" --runs 3 >/dev/null || exit 1
cd "$WT" || exit 1
export CARGO_TARGET_DIR="$OPT_WORK/target_host"
GEMMA4_ONLY="$ONLY" cargo run --release --bin test_kernels 2>&1 | grep -E "^\[|PASS|FAIL|panick|error:" | tail -12
