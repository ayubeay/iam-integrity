# Browser-vs-native comparison — fair-compute-bench

_Generated 2026-07-21 23:17 UTC by `compare.py`. First research artifact of the browser-fair-compute reserve._

## What this does and does not claim

This report compares **performance** between two runtimes already shown to produce **byte-identical output** for identical input. It is evidence, not proof, and its scope is deliberately narrow: one deterministic dependent-memory workload, on one machine. The headline claim is drawn only from the **stable** measurements (relative spread ≤ 15% in both runtimes) — in practice the larger, DRAM-latency-bound sizes, where the timing is not dominated by cache residency or scheduler noise. Smaller sizes are reported in the table but excluded from the headline when noisy.

A single machine cannot establish hardware fairness. Phases 2 and 3 of the reserve — diverse devices, a massively-parallel GPU baseline, and adversarial replication — remain open, and are where the fairness thesis is actually tested rather than mere runtime parity.

## Implementation identity

All compared runs share one `implementation_hash`:

    50f713afa874d4456c399eb592d14f4260fa544c37c30f116a811b293c1110c6

The generator refuses to emit a comparison when this differs across runtimes, so the ratios below are between the same source compiled two ways — not two different programs.

## Efficiency ratio (browser ÷ native)

Lower `ns/step` is faster. A ratio near **1.0** means browser execution is competitive; a large ratio means native wins. `min` is the least-contended run and is the more defensible estimator; `median` is shown alongside because browser timers are coarsened and median absorbs that noise.

| Scratchpad | Native min | Browser min | **Ratio (min)** | Native median | Browser median | Ratio (median) |
|---|---|---|---|---|---|---|
| 1 MiB | 11.8 | 12.0 | **1.01×** | 12.0 | 12.4 | 1.03× |
| 2 MiB | — | 12.4 | **—** | — | 12.6 | — |
| 8 MiB | 14.3 | 13.8 | **0.96×** | 23.3 | 15.2 | 0.65× |
| 16 MiB | — | 36.8 | **—** | — | 37.6 | — |
| 32 MiB | 75.6 | 71.0 | **0.94×** | 79.5 | 76.8 | 0.97× |
| 128 MiB | 107.3 | 105.2 | **0.98×** | 113.2 | 106.0 | 0.94× |
| 256 MiB | 112.8 | 110.8 | **0.98×** | 113.8 | 112.4 | 0.99× |

**Headline (stable measurements only — 128 MiB, 256 MiB):** across 2 size(s) where both runtimes measured with ≤ 15% spread, the browser/native ratio on `min` is **0.98×–0.98×** (mean 0.98×). These are the DRAM-latency-bound sizes; the correct reading is that browser WebAssembly and native Rust are statistically similar once memory latency dominates execution, differing by only a few percent.

_Excluded from the headline as too noisy to quote: 1 MiB, 8 MiB, 32 MiB (see noise flags below). They remain in the table for completeness._

### Noise flags

These sizes had relative spread above 5% in at least one runtime and should not be quoted as citable figures without a quieter machine:

- 1 MiB — native spread 35%, browser spread 5%
- 8 MiB — native spread 192%, browser spread 55%
- 16 MiB — native spread —, browser spread 25%
- 32 MiB — native spread 6%, browser spread 37%
- 128 MiB — native spread 12%, browser spread 2%
- 256 MiB — native spread 7%, browser spread 4%

## Reproducibility

### native

- **os**: macos
- **arch**: aarch64
- **cpu_model**: Apple M1
- **logical_cpus**: 8
- **toolchain**: rustc 1.93.0 (254b59607 2026-01-19)
- **target**: aarch64-apple-darwin
- **build_profile**: release
- **timed_region**: dependent-access loop only
- **scheduling_hint**: user_interactive (macos qos hint)
- **warmup_runs**: 2
- **timed_runs**: 7

### browser

- **os**: MacIntel
- **arch**: wasm32
- **cpu_model**: browser (not exposed)
- **logical_cpus**: 8
- **user_agent**: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/27.0 Safari/605.1.15
- **build_profile**: wasm32-release
- **timed_region**: dependent-access loop only
- **scheduling_hint**: browser tab (performance cores at foreground QoS)
- **timer**: performance.now (browser-coarsened)
- **warmup_runs**: 2
- **timed_runs**: 7

### Scheduling caveat (read before trusting any ratio)

On heterogeneous CPUs (Apple Silicon P/E cores, Intel P/E cores) the core a process lands on dominates a memory-latency result: a foreground browser tab runs on performance cores, while a CLI process may run on efficiency cores with a much smaller L2. At a working-set size that fits one core's L2 but not the other's, identical code can differ ~10x. The native harness now requests performance-core scheduling (`scheduling_hint`), but this is a bias, not a guarantee — confirm core residency out of band before quoting a ratio, and prefer a size that is DRAM-bound on both runtimes for the cleanest comparison.

_Not captured automatically — record by hand for a citable run:_ power mode (plugged in vs battery), background load, thermal state, actual core residency, and for the browser, the exact version and whether the tab was focused.
