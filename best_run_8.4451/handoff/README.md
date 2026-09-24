# Official best 8.4451x — 9d65fb99

This crate uses the exact accepted official device source and ops wrapper.
Official server builds its own binary and runs its own seeded tests.
The recorded custom-Arena binary hash is provenance, not the server binary hash.
No target cache, .git directory, credentials, or executable binary is packaged.

Build with the included rust-toolchain.toml and Cargo.lock, cargo-furiosa-opt 0.8.1.
Submit this crate directory using moa-submitter submit --source <absolute-path>.
For custom Arena, use the campaign single-run harness: one seed, QKV x1, SAO x1, FFN x1.
Do not assume a rebuilt binary reproduces timing; validate correctness and remeasure.
This score is an observed official result, not a statistically established median.
Verify every file using SOURCE_MANIFEST.json after extraction.
