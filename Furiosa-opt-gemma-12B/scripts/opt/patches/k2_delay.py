#!/usr/bin/env python3
"""k2_delay.py <worktree> (env NC=4) -- length of K2's identity-copy delay chain (ver16: 24)."""
import sys, os
wt = sys.argv[1]; n = int(os.environ.get("NC", "4"))
p = wt + "/src/device/sliding/projection.rs"; t = open(p).read()
i = t.index("fn delay_output_input("); j = t.index("\n}", t.index("copy_output_input(device, &x)\n}", i)) + 2
body = ("fn delay_output_input(\n    device: &mut Device,\n"
        "    x: &DmTensor<bf16, Chip, OutputCluster, OutputRowsColumns, m![Qs % 1024]>,\n"
        ") -> DmTensor<bf16, Chip, OutputCluster, OutputRowsColumns, m![Qs % 1024]> {\n"
        "    let x = copy_output_input(device, x);\n" + "    let x = copy_output_input(device, &x);\n" * (n - 2) +
        "    copy_output_input(device, &x)\n}\n")
open(p, "w").write(t[:i] + body + t[j:])
print(f"K2 delay chain {n}")
