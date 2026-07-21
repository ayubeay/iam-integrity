// fair-compute-bench Phase 2 — Metal (MSL) implementation of the same
// dependent-memory-latency workload the native and wasm builds run.
//
// This is an INDEPENDENT implementation, like reference_workload.py and the JS
// reference. It cannot share the Rust `implementation_hash` (different source),
// so equivalence is proven the same way the other references prove it: it must
// reproduce the known-answer digests bit for bit. The Rust harness verifies
// that before any throughput number is trusted.
//
// Why Metal and not WGSL: MSL has native 64-bit `ulong`, so the u64 splitmix
// arithmetic and the u64 accumulator are exact. WGSL lacks native 64-bit ints
// and would require emulation, which is where a determinism-breaking bug would
// hide. `ulong` overflow wraps mod 2^64 (C semantics), matching Rust's
// wrapping_* ops exactly.
//
// Parallelism model: one GPU thread = one INDEPENDENT chain. Thread `gid` owns
// scratchpad region [gid*words, (gid+1)*words) and uses seed = seed0 + gid.
// A single dependent chain cannot be parallelised (each step needs the previous
// load), so throughput comes only from running many independent chains at once
// — which is exactly the parallel-farm scenario the fairness thesis is about.

#include <metal_stdlib>
using namespace metal;

// splitmix64 — identical constants to src/workload.rs `mix`.
static inline ulong mix(ulong x) {
    ulong z = x + 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}

// Fill one worker's scratchpad. Streaming write — kept in its own kernel so it
// can be excluded from the timed region, exactly as init is on the CPU.
kernel void init_workers(
    device ulong*   pads   [[buffer(0)]],   // W * words
    constant ulong& words  [[buffer(1)]],
    constant ulong& seed0  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    device ulong* pad = pads + (ulong)gid * words;
    ulong seed = seed0 + (ulong)gid;
    for (ulong i = 0; i < words; i++) {
        pad[i] = mix(seed ^ (i * 0x10000000000001B3UL));
    }
}

// The timed kernel: the dependent-access loop, then finalize. Mirrors
// workload.rs `run_loop` + `finalize`. Writes each worker's 4-word digest and
// its executed-step count so the host can assert both.
kernel void run_workers(
    device ulong*   pads      [[buffer(0)]],   // W * words (mutated in place)
    device ulong*   digests   [[buffer(1)]],   // W * 4
    device ulong*   executed  [[buffer(2)]],   // W
    constant ulong& words     [[buffer(3)]],
    constant ulong& steps     [[buffer(4)]],
    constant ulong& seed0     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    device ulong* pad = pads + (ulong)gid * words;
    ulong seed = seed0 + (ulong)gid;

    // run_loop
    ulong mask = words - 1;                       // words is a power of two
    ulong acc = mix(seed ^ 0xA5A55A5AC3C33C3CUL);
    ulong s = 0;
    for (; s < steps; s++) {
        ulong idx = acc & mask;
        ulong v = pad[idx];
        acc = mix(acc ^ v) + s;
        pad[idx] = v ^ acc;
    }
    executed[gid] = s;

    // finalize — strided sample, identical to the CPU finalize.
    ulong h0 = mix(acc);
    ulong h1 = mix(acc ^ steps);
    ulong h2 = mix(acc ^ words);
    ulong h3 = mix(acc ^ seed);
    ulong SAMP = 1024;
    ulong stride = words > SAMP ? words / SAMP : 1;
    for (ulong i = 0; i < words; i += stride) {
        ulong v = pad[i];
        h0 = mix(h0 ^ v);
        h1 = h1 + mix(v ^ i);
        h2 = h2 ^ mix(h2 ^ v);
        h3 = mix(h3 + v);
    }
    device ulong* out = digests + (ulong)gid * 4;
    out[0] = h0; out[1] = h1; out[2] = h2; out[3] = h3;
}
