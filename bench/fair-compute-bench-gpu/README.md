# fair-compute-bench-gpu — Phase 2 (Metal GPU baseline)

Scope and doctrine: [`docs/browser-fair-compute-reserve.md`](../../docs/browser-fair-compute-reserve.md).
Depends on the Phase 1 crate [`../fair-compute-bench`](../fair-compute-bench) for
the CPU baseline and the reference digests.

## The question

Phase 1 established runtime parity: browser WebAssembly ≈ native Rust on the
dependent-memory workload. Phase 2 tests the actual thesis:

> Can a massively-parallel GPU outperform a deliberately latency-bound,
> dependency-chained workload while preserving identical execution semantics?

A single dependent chain cannot be parallelised — each step needs the previous
load before it can compute the next address. So any GPU advantage must come from
running **many independent chains at once**. That is the parallel-farm scenario
the fairness thesis is really about, and it is bounded by two things: each chain
is latency-bound, and a fixed memory budget caps how many large scratchpads fit.

## Why Metal + Rust

- **Metal**, because MSL has native 64-bit `ulong`. The workload is u64
  splitmix over a u64 accumulator; WGSL would force 64-bit emulation, which is
  where a determinism-breaking bug would hide. `workload.metal` uses the same
  constants as `src/workload.rs`, and `ulong` overflow wraps mod 2⁶⁴ exactly
  like Rust's `wrapping_*`.
- **Rust harness**, because it reuses the Phase 1 workload for the CPU baseline
  and the reference digests — keeping the two implementations semantically close
  and the comparison defensible.

This crate is **not** zero-dependency (`metal`, `objc`) — GPU work needs
bindings. That is why it is a separate crate; the core stays clean.

## Determinism first, always

The Metal kernel is an independent implementation, like
`reference_workload.py` and the JS reference. It cannot share the Rust
`implementation_hash`, so equivalence is proven the same way: **it must
reproduce the known-answer digests bit for bit.** `verify` runs that gate, and
`throughput`/`sweep` refuse to report a number for any config whose GPU digest
does not match the CPU reference. The kernel's algorithm has been checked
against all five KATs via a Python transliteration; the wasm-style digest match
on real hardware is the final gate.

## Build and run

Requires a Metal-capable GPU (any Apple Silicon or recent Intel Mac).

```
cargo run --release -- verify

# one point: 32 MiB per worker, 256 parallel chains
cargo run --release -- throughput --scratchpad-mib 32 --workers 256 --steps 5000000 \
    --json results/gpu-32mib.json

# the fairness curve: fixed memory budget, workers = budget / per-worker size
cargo run --release -- sweep --budget-mib 1024 --steps 5000000 --out-dir results

python3 gpu_report.py            # results/gpu-*.json -> results/report/phase2.{md,csv,json}
```

`verify` runs automatically before `throughput` and `sweep`; a failed gate
aborts with a non-zero exit.

## What the sweep measures

For each per-worker scratchpad size, `workers = budget / size`, so small
scratchpads run many parallel chains and large ones run few. At each size it
reports aggregate dependent-steps/sec on the GPU (timed by the GPU's own clock)
and on the CPU (multi-threaded, same total work, wall time), and their ratio.

- Init and allocation are excluded from timing on both, matching Phase 1.
- The **GPU/CPU advantage vs scratchpad size** is the result. The thesis
  predicts it is large when the working set is small (GPU parallelism wins) and
  collapses toward ~1× (or below) as the working set becomes DRAM-latency bound
  and memory capacity caps the worker count.

## Scope

This is Phase 2A: an **integrated** GPU (e.g. Apple M1) vs its own CPU, on one
machine. An integrated GPU is not a data-center GPU — a discrete A100/H100 has
far more memory and bandwidth and could behave differently. That is Phase 2B
(CUDA). This crate deliberately does not attempt portability (Phase 2C, `wgpu`)
or adversarial Sybil replication (Phase 3).

## Status

- [x] MSL kernel algorithm verified against the KATs (Python transliteration)
- [x] Harness written: `verify`, `throughput`, `sweep`; throughput JSON schema
- [x] Phase 2 report generator with a hard determinism gate
- [ ] Compiled and run on hardware — the metal-rs plumbing is unverified in the
      authoring environment; first `cargo run -- verify` on a Mac is the gate

## Not in scope

No token. No network. No use of anyone's GPU without them running this
themselves. The product doctrine in the reserve applies here too.
