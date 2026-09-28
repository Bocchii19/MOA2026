#!/usr/bin/env python3
"""k3_uptiles.py <worktree>  (env UP=a,b  GATE=a,b)

Up/Gate weights in several row tiles so each contraction starts when its first tile lands instead
of after the whole 14.7 MB/cluster matrix. The tile view names the row window as its own axis
(UpR), the same trick the Down loader uses with KhR: 0.8.1 rejects a window on an inner digit of L."""
import sys, os
wt = sys.argv[1]
up = [int(x) for x in os.environ.get("UP", "30").split(",")]
gate = [int(x) for x in os.environ.get("GATE", "30").split(",")]
assert sum(up) == 30 and sum(gate) == 30
p = wt + "/src/device/shared/mlp.rs"; t = open(p).read()
old = """            weight
                .view()
                .tile::<m![L % 30], $rows, m![L / 30, L % 30 = $rows # 30, H]>(r0)
                .to_dm(&mut device.tdma)"""
new = """            let v: HbmTensorView<'_, f4e2m1, Chip, m![UpG, UpR, H]> = unsafe { weight.view().reshape() };
            let d: DmTensor<f4e2m1, Chip, m![UpG / 256], m![UpG % 256], m![UpR = $rows, H]> =
                v.tile::<m![UpR], $rows, m![UpG, UpR = $rows # 30, H]>(r0).to_dm(&mut device.tdma);
            unsafe { d.reshape() }"""
assert old in t; t = t.replace(old, new, 1)
t = t.replace("macro_rules! up_gate_tile_fns {",
              "/// L as (UpG, UpR): 512 slice groups x 30 rows, so a row window is a whole named axis.\n"
              "axes![UpG = 512, UpR = 30];\n\nmacro_rules! up_gate_tile_fns {", 1)
have = {6, 12, 18, 24, 30}
for r in set(up + gate) - have:
    t = t.replace("up_gate_tile_fns!(load_up_gate_tile30, up_gate_tile30, 30);",
                  f"up_gate_tile_fns!(load_up_gate_tile30, up_gate_tile30, 30);\n"
                  f"up_gate_tile_fns!(load_up_gate_tile{r}, up_gate_tile{r}, {r});", 1)
def drv(rows):
    off = 0; loads = []; tiles = []
    for i, r in enumerate(rows):
        loads.append(f"    let w{i} = load_up_gate_tile{r}(device, weight, {off});")
        tiles.append(f"    up_gate_tile{r}(device, x, w{i}, &mut blocks0, {off});")
        off += r
    return loads, tiles
for which, rows, scale_line in (("first", up, "    let scale = load_up_gate_scale4(device, scale);"),
                                ("last", gate, "    let scale = load_up_gate_scale(device, scale);")):
    old = f"""    let w0 = load_up_gate_tile30(device, weight, 0);
{scale_line}
    let mut result: UpResult = DmTensor::new();
    let mut blocks0: UpBlocks = DmTensor::new();
    up_gate_tile30(device, x, w0, &mut blocks0, 0);"""
    loads, tiles = drv(rows)
    new = "\n".join(loads) + "\n" + scale_line + """
    let mut result: UpResult = DmTensor::new();
    let mut blocks0: UpBlocks = DmTensor::new();
""" + "\n".join(tiles)
    assert old in t, which
    t = t.replace(old, new, 1)
open(p, "w").write(t)
print(f"up tiles {up}  gate tiles {gate}")
