#!/bin/bash
# build.sh <worktree> <out-binary>  -- build test_kernels for exactly this worktree.
# Two traps make a shared CARGO_TARGET_DIR unsafe across worktrees, and both silently hand back
# another worktree's binary: the furiosa-opt kernel cache is keyed by kernel name only (so each
# worktree gets its own FURIOSA_OPT_OUT_DIR), and every worktree is the same cargo package, so its
# fingerprint says "fresh" after any other worktree built (so the local fingerprints are dropped).
set -u
. "$(dirname "$0")/env.sh"
# One cargo at a time in the shared target dir: build.sh drops fingerprints, which must not race
# another build (it once deleted a fingerprint dir mid-write).
exec 9>"$OPT_WORK/.build.lock"; flock 9
WT="$1"; OUT="$2"
export FURIOSA_OPT_OUT_DIR="$OPT_WORK/kern_$(basename "$WT")"
rm -rf "$CARGO_TARGET_DIR"/release/.fingerprint/furiosa-opt-gemma4-* "$FURIOSA_OPT_OUT_DIR"
cd "$WT" || exit 1
if ! cargo furiosa-opt build --release --bin test_kernels > "$OUT.log" 2>&1; then
  echo "BUILD FAIL $(basename "$WT")"; grep -m5 -A4 "^error" "$OUT.log"; exit 1
fi
cp -f "$CARGO_TARGET_DIR/release/test_kernels" "$OUT"
echo "built $(basename "$WT"): $(grep -c 'Compiling furiosa_opt_gemma4::ops::' "$OUT.log") kernels, md5 $(md5sum "$OUT" | cut -c1-8)"
