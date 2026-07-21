//! CLI runner for the dependent-memory workload.
//!
//! Milestone A deliverable: deterministic output, repeatable CLI, JSON export
//! using the schema that the Milestone B browser harness must also emit.

use fair_compute_bench::workload::{self, Params};
use fair_compute_bench::implementation_hash;

use std::time::Instant;

const SCHEMA_VERSION: &str = "fair-compute-bench/1";
const DEFAULT_RUNS: u64 = 7;
const DEFAULT_WARMUP: u64 = 2;

struct Config {
    params: Params,
    runs: u64,
    warmup: u64,
    json_path: Option<String>,
    quiet: bool,
    label: String,
}

fn usage() -> String {
    format!(
        "fair-compute-bench {}

Deterministic dependent-memory-latency benchmark (Milestone A: native).

USAGE:
    fair-compute-bench [OPTIONS]

OPTIONS:
    --seed <u64>              Seed, decimal or 0x-prefixed hex   [default: 0]
    --scratchpad-mib <n>      Scratchpad size in MiB, power of two [default: 32]
    --scratchpad-words <n>    Exact word count; overrides --scratchpad-mib
    --steps <n>               Dependent accesses per run       [default: 20000000]
    --runs <n>                Timed runs                       [default: {}]
    --warmup <n>              Untimed warmup runs              [default: {}]
    --label <s>               Free-text label recorded in JSON
    --json <path>             Write results as JSON
    --quiet                   Suppress human-readable output
    --print-kat               Print known-answer vectors and exit
    --help                    This text

NOTES:
    Scratchpad initialisation is excluded from the timed region: it is a
    streaming write and would contaminate a latency measurement.

    The scratchpad is re-initialised before every run. Runs are therefore
    independent and must all produce an identical digest; if they do not,
    determinism_ok is false and the result is not usable.

    Build with --release. A debug build measures the optimiser's absence.
",
        env!("CARGO_PKG_VERSION"),
        DEFAULT_RUNS,
        DEFAULT_WARMUP
    )
}

fn parse_u64(s: &str, flag: &str) -> u64 {
    let r = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(&hex.replace('_', ""), 16)
    } else {
        s.replace('_', "").parse::<u64>()
    };
    match r {
        Ok(v) => v,
        Err(_) => {
            eprintln!("error: {} expects a u64, got '{}'", flag, s);
            std::process::exit(2);
        }
    }
}

fn parse_args() -> Config {
    let argv: Vec<String> = std::env::args().skip(1).collect();

    let mut seed: u64 = 0;
    let mut mib: u64 = 32;
    let mut words: Option<u64> = None;
    let mut steps: u64 = 20_000_000;
    let mut runs = DEFAULT_RUNS;
    let mut warmup = DEFAULT_WARMUP;
    let mut json_path = None;
    let mut quiet = false;
    let mut label = String::new();

    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].as_str();
        let next = |i: &mut usize| -> String {
            *i += 1;
            match argv.get(*i) {
                Some(v) => v.clone(),
                None => {
                    eprintln!("error: {} requires a value", arg);
                    std::process::exit(2);
                }
            }
        };
        match arg {
            "--help" | "-h" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "--print-kat" => {
                print_kat();
                std::process::exit(0);
            }
            "--seed" => { let v = next(&mut i); seed = parse_u64(&v, "--seed"); }
            "--scratchpad-mib" => { let v = next(&mut i); mib = parse_u64(&v, "--scratchpad-mib"); }
            "--scratchpad-words" => { let v = next(&mut i); words = Some(parse_u64(&v, "--scratchpad-words")); }
            "--steps" => { let v = next(&mut i); steps = parse_u64(&v, "--steps"); }
            "--runs" => { let v = next(&mut i); runs = parse_u64(&v, "--runs"); }
            "--warmup" => { let v = next(&mut i); warmup = parse_u64(&v, "--warmup"); }
            "--label" => { label = next(&mut i); }
            "--json" => { json_path = Some(next(&mut i)); }
            "--quiet" => { quiet = true; }
            other => {
                eprintln!("error: unknown argument '{}'\n\n{}", other, usage());
                std::process::exit(2);
            }
        }
        i += 1;
    }

    if runs == 0 {
        eprintln!("error: --runs must be > 0");
        std::process::exit(2);
    }

    let scratchpad_words = words.unwrap_or_else(|| {
        mib.saturating_mul(1024 * 1024) / 8
    });

    let params = Params { seed, scratchpad_words, steps };
    if let Err(e) = params.validate() {
        eprintln!("error: {}", e);
        std::process::exit(2);
    }

    Config { params, runs, warmup, json_path, quiet, label }
}

fn print_kat() {
    // These vectors are cross-checked against reference/reference_workload.py,
    // an independent transcription. See tests/determinism.rs.
    let vectors: [(u64, u64, u64); 5] = [
        (0, 1024, 10_000),
        (0, 1024, 10_001),
        (1, 1024, 10_000),
        (0xDEAD_BEEF_CAFE_F00D, 4096, 100_000),
        (0x0123_4567_89AB_CDEF, 65536, 250_000),
    ];
    for (seed, w, s) in vectors.iter() {
        let d = workload::execute(Params { seed: *seed, scratchpad_words: *w, steps: *s })
            .expect("KAT parameters must be valid");
        println!("KAT seed=0x{:016x} words={} steps={} -> {}", seed, w, s, d.to_hex());
    }
}

fn cpu_model() -> String {
    // Best effort. Absence is recorded honestly rather than guessed at.
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(txt) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in txt.lines() {
                if let Some((k, v)) = line.split_once(':') {
                    if k.trim() == "model name" {
                        return v.trim().to_string();
                    }
                }
            }
        }
    }
    "unknown".to_string()
}

fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

struct RunResult {
    elapsed_ns: u128,
    ns_per_step: f64,
    digest: String,
}

fn main() {
    let cfg = parse_args();
    let p = cfg.params;

    let debug_build = cfg!(debug_assertions);
    if debug_build && !cfg.quiet {
        eprintln!(
            "WARNING: this is a debug build. Timings are meaningless.\n\
             Rebuild with: cargo run --release --"
        );
    }

    let mut pad = vec![0u64; p.scratchpad_words as usize];

    // Warmup: pages the scratchpad in, settles frequency scaling, and lets any
    // first-touch allocation cost land outside the measured runs.
    for _ in 0..cfg.warmup {
        workload::init_scratchpad(&mut pad, p.seed);
        let d = workload::run_chain(&mut pad, p);
        std::hint::black_box(d);
    }

    let mut results: Vec<RunResult> = Vec::with_capacity(cfg.runs as usize);

    for run_i in 0..cfg.runs {
        // Re-initialise OUTSIDE the timed region. Required for run
        // independence: run_chain mutates the pad, so without this each run
        // would start from a different state and digests would diverge.
        workload::init_scratchpad(&mut pad, p.seed);

        let t0 = Instant::now();
        let digest = workload::run_chain(&mut pad, p);
        let elapsed = t0.elapsed();

        let digest = std::hint::black_box(digest);
        let elapsed_ns = elapsed.as_nanos();
        let ns_per_step = elapsed_ns as f64 / p.steps as f64;

        if !cfg.quiet {
            println!(
                "run {:>2}/{:<2}  {:>10.3} ms   {:>7.3} ns/step   {:>10.2} Mstep/s   {}",
                run_i + 1,
                cfg.runs,
                elapsed_ns as f64 / 1e6,
                ns_per_step,
                (p.steps as f64 / (elapsed_ns as f64 / 1e9)) / 1e6,
                &digest.to_hex()[..16]
            );
        }

        results.push(RunResult {
            elapsed_ns,
            ns_per_step,
            digest: digest.to_hex(),
        });
    }

    let first = results[0].digest.clone();
    let determinism_ok = results.iter().all(|r| r.digest == first);

    let mut sorted: Vec<f64> = results.iter().map(|r| r.ns_per_step).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    let min = sorted[0];
    let max = sorted[n - 1];
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let variance = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let stddev = variance.sqrt();
    // Relative spread is the honest noise indicator. If this is large, the
    // machine was not quiet and the numbers should not be quoted.
    let rel_spread = if median > 0.0 { (max - min) / median } else { 0.0 };

    let impl_hash = implementation_hash();

    if !cfg.quiet {
        println!();
        println!("workload            dependent-memory-latency chain");
        println!("implementation      {}", impl_hash);
        println!("scratchpad          {} words ({:.2} MiB)", p.scratchpad_words,
                 p.scratchpad_bytes() as f64 / (1024.0 * 1024.0));
        println!("steps/run           {}", p.steps);
        println!("digest              {}", first);
        println!("determinism         {}", if determinism_ok { "OK (all runs identical)" } else { "FAILED" });
        println!();
        println!("ns/step  min {:.3}   median {:.3}   max {:.3}   stddev {:.3}", min, median, max, stddev);
        println!("relative spread     {:.2}%{}", rel_spread * 100.0,
                 if rel_spread > 0.05 { "   <- noisy; quiet the machine before quoting this" } else { "" });
        if debug_build {
            println!();
            println!("BUILD               debug -- timings above are not valid");
        }
        if !determinism_ok {
            println!();
            println!("DIGEST MISMATCH ACROSS RUNS. Do not use these numbers.");
            for (i, r) in results.iter().enumerate() {
                println!("  run {}: {}", i + 1, r.digest);
            }
        }
    }

    if let Some(path) = &cfg.json_path {
        let mut runs_json = String::new();
        for (i, r) in results.iter().enumerate() {
            if i > 0 {
                runs_json.push_str(",\n");
            }
            runs_json.push_str(&format!(
                "      {{ \"index\": {}, \"elapsed_ns\": {}, \"ns_per_step\": {:.6}, \"digest\": \"{}\" }}",
                i, r.elapsed_ns, r.ns_per_step, r.digest
            ));
        }

        let json = format!(
"{{
  \"schema\": \"{schema}\",
  \"harness\": {{
    \"name\": \"fair-compute-bench\",
    \"version\": \"{ver}\",
    \"runtime\": \"native\",
    \"build_profile\": \"{profile}\",
    \"label\": \"{label}\"
  }},
  \"workload\": {{
    \"name\": \"dependent-memory-latency-chain\",
    \"implementation_hash\": \"{impl_hash}\",
    \"seed\": {seed},
    \"scratchpad_words\": {words},
    \"scratchpad_bytes\": {bytes},
    \"steps\": {steps}
  }},
  \"host\": {{
    \"os\": \"{os}\",
    \"arch\": \"{arch}\",
    \"pointer_width_bits\": {ptr},
    \"cpu_model\": \"{cpu}\",
    \"logical_cpus\": {ncpu},
    \"toolchain\": \"{toolchain}\",
    \"target\": \"{target}\"
  }},
  \"protocol\": {{
    \"warmup_runs\": {warmup},
    \"timed_runs\": {nruns},
    \"init_excluded_from_timing\": true,
    \"scratchpad_reinitialised_between_runs\": true,
    \"threads\": 1
  }},
  \"runs\": [
{runs}
  ],
  \"summary\": {{
    \"digest\": \"{digest}\",
    \"determinism_ok\": {det},
    \"ns_per_step_min\": {min:.6},
    \"ns_per_step_median\": {median:.6},
    \"ns_per_step_max\": {max:.6},
    \"ns_per_step_mean\": {mean:.6},
    \"ns_per_step_stddev\": {stddev:.6},
    \"relative_spread\": {spread:.6},
    \"steps_per_second_median\": {sps:.3}
  }}
}}
",
            schema = SCHEMA_VERSION,
            ver = env!("CARGO_PKG_VERSION"),
            profile = if debug_build { "debug" } else { "release" },
            label = json_escape(&cfg.label),
            impl_hash = impl_hash,
            seed = p.seed,
            words = p.scratchpad_words,
            bytes = p.scratchpad_bytes(),
            steps = p.steps,
            os = std::env::consts::OS,
            arch = std::env::consts::ARCH,
            ptr = usize::BITS,
            cpu = json_escape(&cpu_model()),
            ncpu = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(0),
            toolchain = json_escape(env!("FCB_RUSTC_VERSION")),
            target = json_escape(env!("FCB_TARGET")),
            warmup = cfg.warmup,
            nruns = cfg.runs,
            runs = runs_json,
            digest = first,
            det = determinism_ok,
            min = min,
            median = median,
            max = max,
            mean = mean,
            stddev = stddev,
            spread = rel_spread,
            sps = 1e9 / median,
        );

        if let Err(e) = std::fs::write(path, json) {
            eprintln!("error: could not write {}: {}", path, e);
            std::process::exit(1);
        }
        if !cfg.quiet {
            println!("\nwrote {}", path);
        }
    }

    if !determinism_ok {
        std::process::exit(1);
    }
}
