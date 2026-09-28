#!/usr/bin/env python3
"""k3_delay.py <worktree>  (env UPC=n GATEC=m)

ver16's K2 trick applied to K3: identity-copy chains make each contraction's activation ready late
in the STATIC schedule, so the scheduler lists the next weight DMA before that contraction. In the
baseline the Gate weight DMA (static 34.4k) is listed after the Up contraction (30.5k) and the Down
weight DMA (65.6k) after the Gate contraction (61.1k); on hardware each of those streams then waits
for the preceding contraction's dispatch (inter-stream gaps of ~7k and ~13k). Up gets UPC copies of
x before its TRF load, Gate a separate TRF load after GATEC more."""
import sys, os
wt = sys.argv[1]; upc = int(os.environ.get("UPC", "2")); gatec = int(os.environ.get("GATEC", "3"))
p = wt + "/src/device/shared/mlp.rs"; t = open(p).read()
helper = '''
/// Identity copy of the broadcast [hi, lo] activation (a scheduling delay, changes no value).
fn copy_up_x(
    device: &mut Device,
    x: &DmTensor<f8e4m3, Chip, UpCluster, UpRows, m![Gs, H]>,
) -> DmTensor<f8e4m3, Chip, UpCluster, UpRows, m![Gs, H]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit()
}

fn up_x_trf(device: &mut Device, x: &DmTensor<f8e4m3, Chip, UpCluster, UpRows, m![Gs, H]>) -> UpActivation {
    device
        .sub
        .begin(x.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf()
}
'''
anchor = "pub(crate) fn project_up_and_gate("
if anchor not in t:
    anchor = "fn project_up_and_gate("
assert anchor in t
t = t.replace(anchor, helper.strip() + "\n\n" + anchor, 1)
old = """    let x: DmTensor<f8e4m3, Chip, UpCluster, UpRows, m![Gs, H]> = unsafe { x_full.reshape() };
    let x_trf: UpActivation = device
        .sub
        .begin(x.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let up = project_up_gate_small_first(device, &x_trf, up_weight_packed, up_weight_scale);
    let gate = project_up_gate_small_last(device, &x_trf, gate_weight_packed, gate_weight_scale);"""
assert old in t
lines = ["    let x: DmTensor<f8e4m3, Chip, UpCluster, UpRows, m![Gs, H]> = unsafe { x_full.reshape() };",
         "    // Delay chains (no value changes): each contraction becomes ready after the next weight DMA",
         "    // in the static schedule, so that DMA is listed -- and on hardware issued -- first."]
cur = "x"
for i in range(upc):
    lines.append(f"    let xu{i} = copy_up_x(device, &{cur});"); cur = f"xu{i}"
lines.append(f"    let x_trf_up = up_x_trf(device, &{cur});")
for i in range(gatec):
    lines.append(f"    let xg{i} = copy_up_x(device, &{cur});"); cur = f"xg{i}"
lines.append(f"    let x_trf_gate = up_x_trf(device, &{cur});")
lines.append("    let up = project_up_gate_small_first(device, &x_trf_up, up_weight_packed, up_weight_scale);")
lines.append("    let gate = project_up_gate_small_last(device, &x_trf_gate, gate_weight_packed, gate_weight_scale);")
t = t.replace(old, "\n".join(lines), 1)
open(p, "w").write(t)
print(f"K3 delay chains: up {upc}, gate +{gatec}")
