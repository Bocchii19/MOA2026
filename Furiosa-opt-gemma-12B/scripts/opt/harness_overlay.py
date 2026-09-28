#!/usr/bin/env python3
"""harness_overlay.py <worktree> [--runs N] [--spans]

Measurement-only changes to a worktree's harness; never commit them to main.
  --runs N  seeded runs per kernel (default 9) in both test_kernels.rs and generate_references.py,
            regenerates the fixture, and prints min / p25 beside the median.
  --spans   captures each span's name and prints every span when GEMMA4_DUMP_SPANS is set."""
import sys, re, subprocess
wt = sys.argv[1]; args = sys.argv[2:]
runs = int(args[args.index("--runs") + 1]) if "--runs" in args else 9
tk = wt + "/src/bin/test_kernels.rs"; gen = wt + "/scripts/generate_references.py"
t = open(tk).read()
t = re.sub(r"^const RUNS: usize = \d+;", f"const RUNS: usize = {runs};", t, flags=re.M)
old = """                println!(
                    "    median cycles={} (of {} runs: {:?})",
                    median(&cycles_by_run),
                    cycles_by_run.len(),
                    cycles_by_run
                );"""
new = """                let mut sorted = cycles_by_run.clone();
                sorted.sort_unstable();
                println!(
                    "    median cycles={} min={} p25={} (of {} runs: {:?})",
                    median(&cycles_by_run),
                    sorted[0],
                    sorted[sorted.len() / 4],
                    cycles_by_run.len(),
                    cycles_by_run
                );"""
if old in t: t = t.replace(old, new, 1)
if "--spans" in args and "fn dump(&self" not in t:
    t = t.replace("#[derive(Clone, Copy)]\nstruct Span {\n    cluster: u64,\n    begin: u64,\n    end: u64,\n}",
                  "#[derive(Clone)]\nstruct Span {\n    cluster: u64,\n    begin: u64,\n    end: u64,\n    name: String,\n}", 1)
    t = t.replace("struct FieldExtractor {\n    begin: Option<u64>,\n    end: Option<u64>,\n    cluster: Option<u64>,\n}",
                  "struct FieldExtractor {\n    begin: Option<u64>,\n    end: Option<u64>,\n    cluster: Option<u64>,\n    name: Option<String>,\n}", 1)
    t = t.replace("    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}\n}",
                  "    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {\n"
                  "        if field.name() == \"name\" { self.name = Some(format!(\"{value:?}\").trim_matches('\"').to_string()); }\n    }\n"
                  "    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {\n"
                  "        if field.name() == \"name\" { self.name = Some(value.to_string()); }\n    }\n}", 1)
    t = t.replace("                self.spans.lock().unwrap().push(Span { cluster, begin, end });",
                  "                let name = extractor.name.clone().unwrap_or_default();\n"
                  "                self.spans.lock().unwrap().push(Span { cluster, begin, end, name });", 1)
    t = t.replace("    fn window_cycles(&self) -> Option<u64> {",
                  "    fn dump(&self, label: &str) {\n"
                  "        let spans = self.spans.lock().unwrap();\n"
                  "        let mut base: HashMap<u64, u64> = HashMap::new();\n"
                  "        for s in spans.iter() { let e = base.entry(s.cluster).or_insert(s.begin); *e = (*e).min(s.begin); }\n"
                  "        let mut rows: Vec<&Span> = spans.iter().collect();\n"
                  "        rows.sort_by_key(|s| (s.cluster, s.begin));\n"
                  "        for s in rows { let b = base[&s.cluster];\n"
                  "            println!(\"SPAN {label} cl={} beg={} end={} dur={} name={}\", s.cluster, s.begin - b, s.end - b, s.end - s.begin, s.name); }\n"
                  "    }\n\n    fn window_cycles(&self) -> Option<u64> {", 1)
    t = t.replace("                tokio::time::sleep(settle).await;\n                collector.window_cycles()",
                  "                tokio::time::sleep(settle).await;\n"
                  "                if std::env::var(\"GEMMA4_DUMP_SPANS\").is_ok() { collector.dump(&format!(\"{}/run{run}\", test.name)); }\n"
                  "                collector.window_cycles()", 1)
# GEMMA4_ONLY=<test name> runs one kernel (the host CPU backend cannot emulate ver16's K1).
if "GEMMA4_ONLY" not in t:
    t = t.replace("    for test in TESTS {\n", "    for test in TESTS {\n        if std::env::var(\"GEMMA4_ONLY\").map_or(false, |o| !o.is_empty() && o != test.name) { continue; }\n", 1)
open(tk, "w").write(t)
g = open(gen).read()
g = re.sub(r"^RUNS = \d+\r?$", f"RUNS = {runs}", g, flags=re.M)
open(gen, "w").write(g)
subprocess.run(["python3", "scripts/generate_references.py"], cwd=wt, check=True, capture_output=True)
print(f"harness overlay: {runs} runs{' + spans' if '--spans' in args else ''}")
