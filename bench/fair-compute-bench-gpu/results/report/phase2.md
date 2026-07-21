# Phase 2 — GPU vs CPU parallel throughput

_Generated 2026-07-21 23:51 UTC by `gpu_report.py`._

**Question.** Can a massively-parallel GPU outperform a deliberately latency-bound, dependency-chained workload while preserving identical execution semantics? Every row below passed the determinism gate — the Metal kernel reproduced the CPU/native digest, so both devices ran the same workload.

Each worker runs one independent dependent chain over its own scratchpad. Workers scale as `budget / per-worker size`, so small scratchpads mean many parallel chains and large ones mean few — the regime where memory capacity and latency should cap a parallel device.

| Scratchpad/worker | Workers | GPU (Msteps/s) | CPU (Msteps/s) | GPU / CPU |
|---|---|---|---|---|
| 1 MiB | 1024 | 118.9 | 202.5 | **0.59×** |
| 4 MiB | 256 | 122.1 | 45.9 | **2.66×** |
| 16 MiB | 64 | 68.3 | 40.1 | **1.70×** |
| 32 MiB | 32 | 36.1 | 38.2 | **0.94×** |
| 64 MiB | 16 | 23.2 | 35.5 | **0.65×** |
| 128 MiB | 8 | 14.0 | 35.9 | **0.39×** |
| 256 MiB | 4 | 8.0 | 21.2 | **0.38×** |

## Reading

GPU/CPU > 1 means the GPU wins; < 1 means the CPU wins. The curve is **not monotonic** — read the peak and the large-size behaviour, not just the endpoints.

- Peak GPU advantage: **2.66×** at 4 MiB/worker (256 workers)
- Largest scratchpad (256 MiB/worker, 4 workers): **0.38×**
- Most CPU-favourable: **0.38×** at 256 MiB/worker

**On this Apple Silicon configuration, the experiment provides evidence consistent with the fairness hypothesis — but the result is parameter-dependent and not yet generalizable.** At large, DRAM-latency-bound scratchpads the GPU advantage falls **below 1×** (0.38× at 256 MiB — the CPU is ~2.6× faster): with only a few workers, the GPU cannot fill its ALUs and each thread pays higher memory latency than a CPU core, so the parallel device does **not** win. But there is a **vulnerability window** at intermediate sizes, peaking at 2.66× at 4 MiB/worker — where the CPU has fallen off its cache cliff yet enough parallel chains still fit to keep the GPU busy. The design implication is concrete: a fair-compute workload must size the per-worker scratchpad past that window (above ~32 MiB/worker here), or a GPU farm reclaims a real edge.

This is an engineering result on **one** GPU architecture, **one** memory budget (see workers column), **one** implementation, and **one** machine. It is enough to turn scratchpad size from an arbitrary knob into a studied variable, and to motivate the right next question — *where does this transition sit on other hardware?* — rather than the overbroad *do GPUs always lose?*. Generalizing needs Phase 2B (a data-center GPU) and a budget-varying sweep.

## Scope and caveats

- One machine, one GPU (Apple M1) vs one CPU (Apple M1, 8 threads). An integrated GPU is not a data-center GPU; a discrete A100/H100 has far more memory and bandwidth and may behave differently. That is Phase 2B (CUDA).
- Throughput here is aggregate dependent-steps/sec across all workers, timed by wall clock on both devices (GPU: around kernel commit→complete; CPU: around the parallel run over all cores). Init and allocation are excluded on both.
- This tests parallel throughput, not single-chain latency (Phase 1) and not adversarial Sybil replication (Phase 3).

## Reproducibility

- GPU: Apple M1
- CPU: Apple M1 (8 threads)
- toolchain: rustc 1.93.0 (254b59607 2026-01-19)
- target: aarch64-apple-darwin
