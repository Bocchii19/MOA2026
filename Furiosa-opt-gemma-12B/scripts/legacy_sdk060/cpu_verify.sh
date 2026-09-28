#!/bin/bash
# Runs tests/test_kernels.rs against the host CPU backend instead of RNGD, so kernel
# numerics can be checked without hardware or an Arena account.
#
# Two gaps in the host emulator are patched in a vendored copy of furiosa-opt-std that
# lives outside the repo and is injected with `--config`, so nothing here changes the
# crate or the submission:
#
#   * `FpUnaryOp::Erf` is `todo!()` on the CPU backend; GeGLU needs it. The stand-in is
#     Abramowitz & Stegun 7.1.26 (~1.5e-7 absolute error), far inside the kernel tolerances.
#
# The CPU backend is not the NPU: it does not model contraction accumulation order or
# cycle counts. A pass here is evidence, not a substitute for `./scripts/rngd_test.sh`.
set -euo pipefail

CRATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CRATE"

VENDOR="${GEMMA4_CPU_VENDOR:-${TMPDIR:-/tmp}/gemma4-cpu-verify}"
REGISTRY="$(echo "$HOME"/.cargo/registry/src/*/furiosa-opt-std-0.6.0)"

if [ ! -f "ref/fixtures.safetensors" ]; then
    echo "cpu_verify.sh: ref/fixtures.safetensors is missing -- generate it first:" >&2
    echo "    python3 scripts/generate_references.py" >&2
    exit 1
fi

if [ ! -d "$VENDOR/furiosa-opt-std" ]; then
    echo "==> vendoring furiosa-opt-std into $VENDOR"
    rm -rf "$VENDOR"
    mkdir -p "$VENDOR"
    cp -r "$REGISTRY" "$VENDOR/furiosa-opt-std"
    chmod -R u+w "$VENDOR"

    python3 - "$VENDOR/furiosa-opt-std/src/engine/vector/op/semantics.rs" <<'PY'
import sys

path = sys.argv[1]
source = open(path).read()
old = '            Self::Erf => |_x| todo!("Erf not implemented"),'
new = '''            Self::Erf => |x| {
                let sign = if x < 0.0 { -1.0f32 } else { 1.0f32 };
                let x = x.abs();
                let t = 1.0 / (1.0 + 0.327_591_1 * x);
                let y = 1.0
                    - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
                        + 0.254_829_592)
                        * t
                        * (-x * x).exp();
                sign * y
            },'''
if old not in source:
    raise SystemExit(f"{path}: Erf stub not found -- furiosa-opt-std may have changed")
open(path, "w").write(source.replace(old, new))
PY
fi

echo "==> running test_kernels on the CPU backend"
exec cargo test --release --test test_kernels \
    --config "patch.crates-io.furiosa-opt-std.path='$VENDOR/furiosa-opt-std'" "$@"
