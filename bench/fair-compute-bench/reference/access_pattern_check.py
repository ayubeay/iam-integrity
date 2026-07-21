#!/usr/bin/env python3
"""Structural check on the access pattern, independent of timing.

A dependent-access chain can silently degenerate in two ways that timing alone
will not reveal:

  1. It falls into a short cycle and revisits a handful of slots, so everything
     stays resident in L1 and the benchmark measures cache hit latency forever.
  2. Consecutive addresses turn out to be spatially close, so the hardware
     prefetcher hides the latency the workload is supposed to expose.

Either failure would make the workload useless while still producing stable,
deterministic, plausible-looking numbers. This script checks for both before any
timing work is trusted.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_workload import mix, init_scratchpad, M64


def trace(seed, words, steps):
    pad = init_scratchpad(words, seed)
    mask = words - 1
    acc = mix(seed ^ 0xA5A55A5AC3C33C3C)
    touched, order = {}, []
    for step in range(steps):
        idx = acc & mask
        touched[idx] = touched.get(idx, 0) + 1
        order.append(idx)
        v = pad[idx]
        acc = (mix(acc ^ v) + step) & M64
        pad[idx] = v ^ acc
    return touched, order


def shortest_cycle(order, max_period=1000, window=4000):
    tail = order[-window:]
    for period in range(1, max_period):
        if len(tail) >= 2 * period and tail[-period:] == tail[-2 * period:-period]:
            return period
    return None


def main():
    ok = True
    print(f"{'words':>8} {'steps':>9} {'coverage':>10} {'max hits':>9} {'expected':>9} {'cycle':>7}")
    for words, steps in [(1024, 20_000), (4096, 40_000), (65536, 200_000)]:
        touched, order = trace(0, words, steps)
        cov = len(touched) / words
        mx = max(touched.values())
        exp = steps / words
        cyc = shortest_cycle(order)
        print(f"{words:>8} {steps:>9} {cov:>9.2%} {mx:>9} {exp:>9.1f} {str(cyc):>7}")
        if cov < 0.90:
            print("  FAIL: chain does not reach most of the scratchpad")
            ok = False
        if cyc is not None:
            print(f"  FAIL: address stream repeats with period {cyc}")
            ok = False

    _, order = trace(0, 65536, 50_000)
    jumps = [abs(order[i + 1] - order[i]) for i in range(len(order) - 1)]
    near = sum(1 for j in jumps if j < 8) / len(jumps)
    print(f"\nconsecutive accesses within 8 words: {near:.4%}")
    if near > 0.01:
        print("  FAIL: access pattern has spatial locality; prefetcher will hide latency")
        ok = False

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
