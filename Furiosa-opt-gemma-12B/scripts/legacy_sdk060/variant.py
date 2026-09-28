#!/usr/bin/env python3
"""Rewrite a kernel source knob in place, so candidates can be built without hand-editing.

    python3 scripts/variant.py down-scale-order 1
    python3 scripts/variant.py down-split 16,12,2
    python3 scripts/variant.py show

The device compiler lowers MIR before const-folding, so a `match` on a `const` inside a
`#[device]` call graph fails with "Mapping not found" and a `const` of a custom enum fails
with "const fill does not support this scalar type". Selecting a variant therefore has to
happen in the source text, not at compile time -- which is what this script does. Each knob
marks its region with a `// VARIANT:<name>=<value>` comment so the current setting is
readable from the file and `show` can report it.
"""

import re
import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parent.parent
MLP = CRATE / "src" / "device" / "shared" / "mlp13.rs"

# --- knob: down-scale-order -------------------------------------------------------------
# Which transfer lands last in the strictly serial DMA queue.
DOWN_SCALE_ORDER_BODIES = {
    "0": """    let a = load_down16(ctx, w, 0);
    let b = load_down10(ctx, w, 16);
    let c = load_down4(ctx, w, 26);
    let s: DownScale = scale_w.to_dm(&mut ctx.tdma);
    ((a, b, c), s)""",
    "1": """    let a = load_down16(ctx, w, 0);
    let b = load_down10(ctx, w, 16);
    let s: DownScale = scale_w.to_dm(&mut ctx.tdma);
    let c = load_down4(ctx, w, 26);
    ((a, b, c), s)""",
    "2": """    let a = load_down16(ctx, w, 0);
    let s: DownScale = scale_w.to_dm(&mut ctx.tdma);
    let b = load_down10(ctx, w, 16);
    let c = load_down4(ctx, w, 26);
    ((a, b, c), s)""",
    "3": """    let s: DownScale = scale_w.to_dm(&mut ctx.tdma);
    let a = load_down16(ctx, w, 0);
    let b = load_down10(ctx, w, 16);
    let c = load_down4(ctx, w, 26);
    ((a, b, c), s)""",
}

FN_RE = re.compile(
    r"(// VARIANT:down-scale-order=)(\S+)(\n"
    r"fn load_down_chunks_and_scale\([^)]*\) -> \(DownChunks, DownScale\) \{\n)"
    r".*?"
    r"(\n\}\n)",
    re.S,
)


def set_down_scale_order(value):
    if value not in DOWN_SCALE_ORDER_BODIES:
        sys.exit(f"down-scale-order must be one of {sorted(DOWN_SCALE_ORDER_BODIES)}")
    src = MLP.read_text()
    m = FN_RE.search(src)
    if not m:
        sys.exit(
            "no down-scale-order variant region in mlp13.rs.\n"
            "This knob was measured and found to be a NO-OP (see WORK_LOG.md section 5, F001):\n"
            "all four orders compile to a byte-identical binary with identical schedule\n"
            "makespan, because the scheduler -- not `to_dm` call order -- places the DMA queue,\n"
            "and it already issues the down block-scale sixth of nine, fully hidden. The\n"
            "refactor was therefore reverted. Re-apply it only to re-check that result."
        )
    new = m.group(1) + value + m.group(3) + DOWN_SCALE_ORDER_BODIES[value] + m.group(4)
    MLP.write_text(src[:m.start()] + new + src[m.end():])
    print(f"down-scale-order = {value}")


def show():
    src = MLP.read_text()
    for name, path in (("down-scale-order", MLP),):
        m = re.search(rf"// VARIANT:{re.escape(name)}=(\S+)", src)
        print(f"{name:<20} {m.group(1) if m else '?'}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "show":
        show()
        return 0
    knob = sys.argv[1]
    if knob == "down-scale-order":
        set_down_scale_order(sys.argv[2])
        return 0
    sys.exit(f"unknown knob {knob!r}")


if __name__ == "__main__":
    sys.exit(main())
