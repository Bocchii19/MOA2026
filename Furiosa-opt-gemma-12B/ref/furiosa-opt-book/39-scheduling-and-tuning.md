# Scheduling and Tuning

Scheduling and tuning turn a correct kernel into a measured plan.
Inspect the schedule, identify the limiting dependency or resource, change one supported choice, and compare under the same conditions.

Scheduling turns a compiled kernel into a testable hypothesis about context overlap, memory dependencies, and resource conflicts.
The chapter follows one loop:

1. Schedule: understand contexts, dependencies, and operation order.
2. Diagnosis: inspect a dumped schedule and state a bottleneck hypothesis.
3. Tuning: test one change, compare the same metric, and keep or revert it.

The chapter does not re-explain mapping, tensor movement, or compute-engine contracts.
It links to those chapters when a scheduling decision depends on them.

For the four validation meanings and their caveats, see Quick Start: Kernel Validation.
Use Diagnosis and Tuning to interpret fixed schedule candidates and make decisions.
