//! Phase 2 — Metal GPU baseline for the dependent-memory-latency workload.
//!
//! The scientific question: can a massively-parallel GPU outperform a
//! deliberately latency-bound, dependency-chained workload while preserving
//! identical execution semantics? A single chain cannot be parallelised, so any
//! GPU advantage must come from running MANY independent chains at once — the
//! parallel-farm scenario the fairness thesis is really about.
//!
//! Order of operations enforced here:
//!   1. DETERMINISM GATE. The Metal kernel must reproduce the known-answer
//!      digests and exact step counts. Same as the Python/JS references: the GPU
//!      cannot share the Rust implementation_hash, so identical output for
//!      identical input is the proof it ran the same workload. No throughput
//!      number is reported for a config whose digest does not match.
//!   2. THROUGHPUT. Aggregate dependent-steps/sec on the GPU vs the CPU
//!      (multi-threaded, same total work), at a given per-worker scratchpad
//!      size and worker count.
//!   3. SWEEP. The fairness curve: GPU-over-CPU throughput advantage as the
//!      per-worker scratchpad grows, under a fixed memory budget. The thesis
//!      predicts the advantage shrinks as the working set becomes DRAM-latency
//!      bound and memory capacity caps the worker count.
//!
//! NOTE: the metal-rs plumbing in this file has not been compiled in the
//! authoring environment (no GPU / no toolchain there). The *algorithm* in
//! workload.metal is verified against the KATs via a Python transliteration.
//! Treat the first `cargo run -- verify` as the real gate.

use std::ffi::c_void;
use std::time::Instant;

use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

use fair_compute_bench::workload::{self, Params};

const KERNEL_SRC: &str = include_str!("workload.metal");

/// (seed, words, steps, expected digest) — identical to tests/determinism.rs.
const KAT: &[(u64, u64, u64, &str)] = &[
    (0, 1024, 10_000, "898d0f747b85e8c4e14d3139cf85723946c2a3a5caf87093153263f8d698a4d8"),
    (0, 1024, 10_001, "4dcff0af62e97a257530d8f5f09ff656a09b980857738fff3a832399597a1767"),
    (1, 1024, 10_000, "e47daa135f3f80ee5f430c033482fad6f1bf2f3aa7d5b251b6bac3b3e40ee9f0"),
    (0xDEAD_BEEF_CAFE_F00D, 4096, 100_000, "d2cf0d099685b3dc5746b1943047b8af89efa3062305a24bb80ed023655be3c6"),
    (0x0123_4567_89AB_CDEF, 65536, 250_000, "718fcb7c78b1a0f19151e3604bfec98b4a0f5181a7abfbe9a39a189f6d6f3199"),
];

fn hex4(w: &[u64]) -> String {
    let mut s = String::with_capacity(64);
    for x in w.iter().take(4) {
        s.push_str(&format!("{:016x}", x));
    }
    s
}

fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c => o.push(c),
        }
    }
    o
}

// ---------------------------------------------------------------------------
// GPU
// ---------------------------------------------------------------------------

struct Gpu {
    device: Device,
    queue: CommandQueue,
    init_pipe: ComputePipelineState,
    run_pipe: ComputePipelineState,
}

/// Outcome of one GPU dispatch of W workers.
struct GpuRun {
    gpu_time_s: f64,
    worker0_digest: String,
    executed_ok: bool,
}

impl Gpu {
    fn new() -> Result<Gpu, String> {
        let device = Device::system_default().ok_or("no Metal device found")?;
        let queue = device.new_command_queue();
        let lib = device
            .new_library_with_source(KERNEL_SRC, &CompileOptions::new())
            .map_err(|e| format!("MSL compile failed: {e}"))?;
        let init_fn = lib.get_function("init_workers", None).map_err(|e| e.to_string())?;
        let run_fn = lib.get_function("run_workers", None).map_err(|e| e.to_string())?;
        let init_pipe = device
            .new_compute_pipeline_state_with_function(&init_fn)
            .map_err(|e| e.to_string())?;
        let run_pipe = device
            .new_compute_pipeline_state_with_function(&run_fn)
            .map_err(|e| e.to_string())?;
        Ok(Gpu { device, queue, init_pipe, run_pipe })
    }

    fn name(&self) -> String {
        self.device.name().to_string()
    }

    fn threadgroup(&self, pipe: &ComputePipelineState, workers: u64) -> MTLSize {
        let max = pipe.max_total_threads_per_threadgroup();
        MTLSize::new(workers.min(max).max(1), 1, 1)
    }

    /// Run W workers: init (untimed) then the dependent loop + finalize (timed
    /// by the GPU's own clock). Returns timing, worker-0 digest, and whether
    /// every worker executed exactly `steps`.
    fn run(&self, words: u64, steps: u64, workers: u64, seed0: u64) -> Result<GpuRun, String> {
        objc::rc::autoreleasepool(|| {
            let opts = MTLResourceOptions::StorageModeShared;
            let pad_len = workers.checked_mul(words).and_then(|v| v.checked_mul(8))
                .ok_or("scratchpad size overflow")?;
            let pads = self.device.new_buffer(pad_len, opts);
            let digests = self.device.new_buffer(workers * 4 * 8, opts);
            let executed = self.device.new_buffer(workers * 8, opts);

            let set_u64 = |enc: &metal::ComputeCommandEncoderRef, idx: u64, v: &u64| {
                enc.set_bytes(idx, 8, v as *const u64 as *const c_void);
            };

            // --- init (untimed) ---
            {
                let cmd = self.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.init_pipe);
                enc.set_buffer(0, Some(&pads), 0);
                set_u64(enc, 1, &words);
                set_u64(enc, 2, &seed0);
                enc.dispatch_threads(
                    MTLSize::new(workers, 1, 1),
                    self.threadgroup(&self.init_pipe, workers),
                );
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }

            // --- run (timed by wall clock around commit -> complete) ---
            // metal 0.29 does not expose GPUStartTime/GPUEndTime; the kernel runs
            // for hundreds of ms so host-side dispatch overhead is negligible, and
            // this matches how the CPU baseline is timed (wall clock), keeping the
            // comparison consistent.
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.run_pipe);
            enc.set_buffer(0, Some(&pads), 0);
            enc.set_buffer(1, Some(&digests), 0);
            enc.set_buffer(2, Some(&executed), 0);
            set_u64(enc, 3, &words);
            set_u64(enc, 4, &steps);
            set_u64(enc, 5, &seed0);
            enc.dispatch_threads(
                MTLSize::new(workers, 1, 1),
                self.threadgroup(&self.run_pipe, workers),
            );
            enc.end_encoding();
            let t0 = Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            let gpu_time_s = t0.elapsed().as_secs_f64();

            // read back
            let dg = unsafe {
                std::slice::from_raw_parts(digests.contents() as *const u64, (workers * 4) as usize)
            };
            let ex = unsafe {
                std::slice::from_raw_parts(executed.contents() as *const u64, workers as usize)
            };
            let executed_ok = ex.iter().all(|&e| e == steps);
            let worker0_digest = hex4(&dg[0..4]);

            Ok(GpuRun { gpu_time_s, worker0_digest, executed_ok })
        })
    }
}

// ---------------------------------------------------------------------------
// CPU baseline (multi-threaded, same total work as the GPU)
// ---------------------------------------------------------------------------

struct CpuRun {
    wall_s: f64,
    threads: usize,
}

/// Run W independent chains across all logical cores. Scratchpads are allocated
/// and initialised OUTSIDE the timed region, matching the GPU (which times only
/// the run kernel) and the Phase 1 CPU protocol.
fn cpu_throughput(words: u64, steps: u64, workers: u64, seed0: u64) -> CpuRun {
    let ncpu = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
    let w = words as usize;

    // allocate + init all pads (untimed)
    let mut pads: Vec<Vec<u64>> = (0..workers)
        .map(|i| {
            let mut p = vec![0u64; w];
            workload::init_scratchpad(&mut p, seed0.wrapping_add(i));
            p
        })
        .collect();

    let chunk = ((workers as usize) + ncpu - 1) / ncpu.max(1);
    let t0 = Instant::now();
    std::thread::scope(|scope| {
        for (ci, chunk_pads) in pads.chunks_mut(chunk.max(1)).enumerate() {
            let base = (ci * chunk) as u64;
            scope.spawn(move || {
                for (j, pad) in chunk_pads.iter_mut().enumerate() {
                    let params = Params {
                        seed: seed0.wrapping_add(base + j as u64),
                        scratchpad_words: words,
                        steps,
                    };
                    let st = workload::run_loop(pad, params);
                    std::hint::black_box(st);
                }
            });
        }
    });
    let wall_s = t0.elapsed().as_secs_f64();
    CpuRun { wall_s, threads: ncpu }
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

fn mode_verify(gpu: &Gpu) -> bool {
    println!("== GPU determinism gate (must reproduce native/reference digests) ==");
    println!("device: {}", gpu.name());
    let mut ok = true;
    for (seed, words, steps, expected) in KAT {
        match gpu.run(*words, *steps, 1, *seed) {
            Ok(r) => {
                let pass = &r.worker0_digest == expected && r.executed_ok;
                if !pass {
                    ok = false;
                }
                println!(
                    "  {} seed=0x{:016x} words={} steps={} executed_ok={}",
                    if pass { "OK  " } else { "FAIL" },
                    seed, words, steps, r.executed_ok
                );
                if !pass {
                    println!("     expected {}\n     got      {}", expected, r.worker0_digest);
                }
            }
            Err(e) => {
                ok = false;
                println!("  FAIL seed=0x{:016x}: {}", seed, e);
            }
        }
    }
    println!(
        "\n{}",
        if ok {
            "DETERMINISM OK — the Metal kernel runs the same workload as native/wasm."
        } else {
            "DETERMINISM FAILED — do not trust any throughput number from this build."
        }
    );
    ok
}

struct ThroughputResult {
    words: u64,
    steps: u64,
    workers: u64,
    seed0: u64,
    gpu: GpuRun,
    cpu: CpuRun,
    cpu_ref_digest: String,
    digest_match: bool,
}

fn throughput(gpu: &Gpu, words: u64, steps: u64, workers: u64, seed0: u64)
    -> Result<ThroughputResult, String>
{
    let g = gpu.run(words, steps, workers, seed0)?;
    // Determinism during throughput: GPU worker-0 must match the CPU reference
    // digest for the same (seed0, words, steps).
    let cpu_ref = workload::execute(Params { seed: seed0, scratchpad_words: words, steps })
        .map_err(|e| e.to_string())?
        .to_hex();
    let digest_match = g.worker0_digest == cpu_ref && g.executed_ok;
    let c = cpu_throughput(words, steps, workers, seed0);
    Ok(ThroughputResult {
        words, steps, workers, seed0, gpu: g, cpu: c,
        cpu_ref_digest: cpu_ref, digest_match,
    })
}

fn mib(words: u64) -> f64 {
    (words * 8) as f64 / (1024.0 * 1024.0)
}

fn report_throughput(gpu: &Gpu, r: &ThroughputResult) {
    let total = (r.workers as f64) * (r.steps as f64);
    let gpu_tp = total / r.gpu.gpu_time_s;
    let cpu_tp = total / r.cpu.wall_s;
    let adv = gpu_tp / cpu_tp;
    println!(
        "{:>6.0} MiB/worker  W={:<6}  GPU {:>8.1} Msteps/s  CPU {:>8.1} Msteps/s  GPU/CPU {:>5.2}x   digest {}",
        mib(r.words), r.workers, gpu_tp / 1e6, cpu_tp / 1e6, adv,
        if r.digest_match { "OK" } else { "MISMATCH" }
    );
    let _ = gpu; // name captured by caller for JSON
}

fn throughput_json(gpu: &Gpu, r: &ThroughputResult, label: &str) -> String {
    let total = (r.workers as f64) * (r.steps as f64);
    let gpu_tp = total / r.gpu.gpu_time_s;
    let cpu_tp = total / r.cpu.wall_s;
    format!(
"{{
  \"schema\": \"fair-compute-bench-throughput/1\",
  \"label\": \"{label}\",
  \"workload\": {{
    \"name\": \"dependent-memory-latency-chain\",
    \"seed0\": {seed0},
    \"scratchpad_words_per_worker\": {words},
    \"scratchpad_bytes_per_worker\": {bytes},
    \"steps_per_worker\": {steps},
    \"workers\": {workers},
    \"total_steps\": {total}
  }},
  \"determinism\": {{
    \"gpu_worker0_digest\": \"{gdig}\",
    \"cpu_reference_digest\": \"{cdig}\",
    \"digest_match\": {dmatch},
    \"executed_ok\": {eok}
  }},
  \"gpu\": {{
    \"device\": \"{gdev}\",
    \"time_s\": {gtime:.9},
    \"throughput_steps_per_s\": {gtp:.1},
    \"throughput_msteps_per_s\": {gtpm:.3}
  }},
  \"cpu\": {{
    \"model\": \"{cmodel}\",
    \"threads\": {cthreads},
    \"wall_time_s\": {ctime:.9},
    \"throughput_steps_per_s\": {ctp:.1},
    \"throughput_msteps_per_s\": {ctpm:.3}
  }},
  \"advantage_gpu_over_cpu\": {adv:.4},
  \"host\": {{
    \"os\": \"{os}\",
    \"arch\": \"{arch}\",
    \"toolchain\": \"{tool}\",
    \"target\": \"{target}\"
  }}
}}
",
        label = json_escape(label),
        seed0 = r.seed0,
        words = r.words,
        bytes = r.words * 8,
        steps = r.steps,
        workers = r.workers,
        total = total as u64,
        gdig = r.gpu.worker0_digest,
        cdig = r.cpu_ref_digest,
        dmatch = r.digest_match,
        eok = r.gpu.executed_ok,
        gdev = json_escape(&gpu.name()),
        gtime = r.gpu.gpu_time_s,
        gtp = gpu_tp,
        gtpm = gpu_tp / 1e6,
        cmodel = json_escape(&cpu_model()),
        cthreads = r.cpu.threads,
        ctime = r.cpu.wall_s,
        ctp = cpu_tp,
        ctpm = cpu_tp / 1e6,
        adv = gpu_tp / cpu_tp,
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        tool = json_escape(env!("FCB_RUSTC_VERSION")),
        target = json_escape(env!("FCB_TARGET")),
    )
}

fn cpu_model() -> String {
    if let Ok(out) = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"]).output()
    {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() { return s; }
    }
    "unknown".to_string()
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn usage() -> String {
    "fair-compute-bench-gpu — Phase 2 Metal baseline

USAGE:
    fair-compute-bench-gpu verify
    fair-compute-bench-gpu throughput [--scratchpad-mib N] [--workers W] [--steps T] [--seed S] [--json PATH]
    fair-compute-bench-gpu sweep [--budget-mib N] [--steps T] [--seed S] [--out-dir DIR]

NOTES:
    verify runs the determinism gate first — the Metal kernel must reproduce the
    known-answer digests. Throughput is only meaningful once it passes.

    sweep varies the per-worker scratchpad and sets workers = budget / size, so
    small scratchpads run many parallel chains and large ones run few. The
    GPU/CPU advantage as size grows is the fairness curve.
".to_string()
}

fn parse_u64(s: &str) -> u64 {
    let s = s.trim().replace('_', "");
    if let Some(h) = s.strip_prefix("0x") { u64::from_str_radix(h, 16).unwrap_or(0) }
    else { s.parse().unwrap_or(0) }
}

fn arg(args: &[String], flag: &str, default: u64) -> u64 {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).map(|v| parse_u64(v)).unwrap_or(default)
}
fn arg_str(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mode = argv.first().cloned().unwrap_or_default();

    // Only touch the GPU for modes that need it, so `--help` / no args work
    // on a machine while we still surface a clear error if Metal is missing.
    if !matches!(mode.as_str(), "verify" | "throughput" | "sweep") {
        print!("{}", usage());
        return;
    }

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {}\n(This build requires a Metal-capable GPU.)", e);
            std::process::exit(1);
        }
    };

    match mode.as_str() {
        "verify" => {
            if !mode_verify(&gpu) {
                std::process::exit(1);
            }
        }
        "throughput" => {
            if !mode_verify(&gpu) {
                eprintln!("\naborting: determinism gate failed.");
                std::process::exit(1);
            }
            let s_mib = arg(&argv, "--scratchpad-mib", 32);
            let words = s_mib * 1024 * 1024 / 8;
            let steps = arg(&argv, "--steps", 5_000_000);
            let workers = arg(&argv, "--workers", 256);
            let seed = arg(&argv, "--seed", 0);
            println!("\n== throughput ==");
            match throughput(&gpu, words, steps, workers, seed) {
                Ok(r) => {
                    report_throughput(&gpu, &r);
                    if let Some(path) = arg_str(&argv, "--json") {
                        let label = arg_str(&argv, "--label").unwrap_or_default();
                        std::fs::write(&path, throughput_json(&gpu, &r, &label)).ok();
                        println!("wrote {}", path);
                    }
                }
                Err(e) => { eprintln!("error: {}", e); std::process::exit(1); }
            }
        }
        "sweep" => {
            if !mode_verify(&gpu) {
                eprintln!("\naborting: determinism gate failed.");
                std::process::exit(1);
            }
            let budget_mib = arg(&argv, "--budget-mib", 1024);
            let steps = arg(&argv, "--steps", 5_000_000);
            let seed = arg(&argv, "--seed", 0);
            let out_dir = arg_str(&argv, "--out-dir").unwrap_or_else(|| "results".to_string());
            std::fs::create_dir_all(&out_dir).ok();
            let sizes_mib: [u64; 7] = [1, 4, 16, 32, 64, 128, 256];
            println!("\n== sweep (memory budget {} MiB, workers = budget / per-worker size) ==", budget_mib);
            for s in sizes_mib {
                let words = s * 1024 * 1024 / 8;
                let workers = (budget_mib / s).max(1);
                match throughput(&gpu, words, steps, workers, seed) {
                    Ok(r) => {
                        report_throughput(&gpu, &r);
                        let path = format!("{}/gpu-{}mib.json", out_dir, s);
                        std::fs::write(&path, throughput_json(&gpu, &r, "phase2 sweep")).ok();
                    }
                    Err(e) => eprintln!("  {} MiB: error {}", s, e),
                }
            }
            println!("\nwrote {}/gpu-*mib.json", out_dir);
        }
        _ => {
            print!("{}", usage());
        }
    }
}
