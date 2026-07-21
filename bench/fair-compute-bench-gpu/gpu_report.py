#!/usr/bin/env python3
"""Phase 2 report: GPU-vs-CPU parallel throughput as a function of per-worker
scratchpad size — the fairness curve.

The thesis it tests: a dependent-memory-latency workload should blunt a
massively-parallel device's advantage as the per-worker working set grows,
because (a) each chain is latency-bound and cannot be parallelised, and (b) a
fixed memory budget caps how many parallel chains fit. If the GPU/CPU advantage
stays large at big scratchpad sizes, the thesis is weakened; if it collapses
toward ~1x, it is supported.

Hard gate: every input must have digest_match == true. If the GPU did not
reproduce the CPU/native digest, it did not run the same workload and no
throughput comparison is valid. Zero dependencies; stdlib only.

    python3 gpu_report.py                 # scans results/, writes results/report/phase2.md
    python3 gpu_report.py --results-dir DIR --out-dir DIR
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

SCHEMA = "fair-compute-bench-throughput/1"


def die(msg):
    print("error: " + msg, file=sys.stderr)
    sys.exit(1)


def load(results_dir):
    rows = []
    if not os.path.isdir(results_dir):
        die("results dir not found: " + results_dir)
    for name in sorted(os.listdir(results_dir)):
        if not (name.startswith("gpu-") and name.endswith(".json")):
            continue
        try:
            with open(os.path.join(results_dir, name)) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print("warning: skip " + name + ": " + str(e), file=sys.stderr)
            continue
        if d.get("schema") != SCHEMA:
            print("warning: skip " + name + ": wrong schema", file=sys.stderr)
            continue
        rows.append((name, d))
    return rows


def mib(d):
    return d["workload"]["scratchpad_bytes_per_worker"] / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "report")

    files = load(args.results_dir)
    if not files:
        die("no gpu-*.json throughput files in " + args.results_dir)

    # Determinism gate — hard refusal.
    bad = [n for n, d in files if not d["determinism"].get("digest_match")]
    if bad:
        die("digest_match is false for: " + ", ".join(bad)
            + "\nThe GPU did not reproduce the workload digest; refusing to report throughput.")

    rows = sorted((d for _, d in files), key=mib)

    sample = rows[0]
    gpu_dev = sample["gpu"]["device"]
    cpu_model = sample["cpu"]["model"]
    cpu_threads = sample["cpu"]["threads"]

    os.makedirs(out_dir, exist_ok=True)

    # CSV
    with open(os.path.join(out_dir, "phase2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scratchpad_mib_per_worker", "workers", "gpu_msteps_per_s",
                    "cpu_msteps_per_s", "advantage_gpu_over_cpu"])
        for d in rows:
            w.writerow([round(mib(d), 3), d["workload"]["workers"],
                        round(d["gpu"]["throughput_msteps_per_s"], 3),
                        round(d["cpu"]["throughput_msteps_per_s"], 3),
                        round(d["advantage_gpu_over_cpu"], 4)])

    # JSON
    with open(os.path.join(out_dir, "phase2.json"), "w") as f:
        json.dump({"schema": "fair-compute-bench-phase2-report/1",
                   "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "gpu_device": gpu_dev, "cpu_model": cpu_model,
                   "rows": rows}, f, indent=2)

    # Markdown
    L = []
    L.append("# Phase 2 — GPU vs CPU parallel throughput")
    L.append("")
    L.append("_Generated " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
             + " by `gpu_report.py`._")
    L.append("")
    L.append("**Question.** Can a massively-parallel GPU outperform a deliberately "
             "latency-bound, dependency-chained workload while preserving identical "
             "execution semantics? Every row below passed the determinism gate — the "
             "Metal kernel reproduced the CPU/native digest, so both devices ran the "
             "same workload.")
    L.append("")
    L.append("Each worker runs one independent dependent chain over its own "
             "scratchpad. Workers scale as `budget / per-worker size`, so small "
             "scratchpads mean many parallel chains and large ones mean few — the "
             "regime where memory capacity and latency should cap a parallel device.")
    L.append("")
    L.append("| Scratchpad/worker | Workers | GPU (Msteps/s) | CPU (Msteps/s) | GPU / CPU |")
    L.append("|---|---|---|---|---|")
    for d in rows:
        L.append("| %.0f MiB | %d | %.1f | %.1f | **%.2f×** |" % (
            mib(d), d["workload"]["workers"],
            d["gpu"]["throughput_msteps_per_s"],
            d["cpu"]["throughput_msteps_per_s"],
            d["advantage_gpu_over_cpu"]))
    L.append("")

    # Data-driven reading — the curve is not assumed monotonic.
    adv = [(mib(d), d["workload"]["workers"], d["advantage_gpu_over_cpu"]) for d in rows]
    peak = max(adv, key=lambda t: t[2])
    trough = min(adv, key=lambda t: t[2])
    large = adv[-1]
    wins = [a for a in adv if a[2] >= 1.0]          # GPU beats CPU
    # smallest per-worker size at/above which the GPU never wins again
    fairness_floor = None
    for i, a in enumerate(adv):
        if all(x[2] < 1.0 for x in adv[i:]):
            fairness_floor = a[0]
            break

    L.append("## Reading")
    L.append("")
    L.append("GPU/CPU > 1 means the GPU wins; < 1 means the CPU wins. The curve is "
             "**not monotonic** — read the peak and the large-size behaviour, not just "
             "the endpoints.")
    L.append("")
    L.append("- Peak GPU advantage: **%.2f×** at %.0f MiB/worker (%d workers)"
             % (peak[2], peak[0], peak[1]))
    L.append("- Largest scratchpad (%.0f MiB/worker, %d workers): **%.2f×**"
             % (large[0], large[1], large[2]))
    L.append("- Most CPU-favourable: **%.2f×** at %.0f MiB/worker" % (trough[2], trough[0]))
    L.append("")

    if large[2] < 1.0 and peak[2] >= 1.2:
        floor_txt = ("above ~%.0f MiB/worker" % fairness_floor) if fairness_floor else "at large sizes"
        L.append("**On this Apple Silicon configuration, the experiment provides evidence "
                 "consistent with the fairness hypothesis — but the result is "
                 "parameter-dependent and not yet generalizable.** At large, "
                 "DRAM-latency-bound scratchpads the GPU advantage falls **below 1×** "
                 "(%.2f× at %.0f MiB — the CPU is ~%.1f× faster): with only a few workers, the "
                 "GPU cannot fill its ALUs and each thread pays higher memory latency than a "
                 "CPU core, so the parallel device does **not** win. But there is a "
                 "**vulnerability window** at intermediate sizes, peaking at %.2f× at %.0f "
                 "MiB/worker — where the CPU has fallen off its cache cliff yet enough parallel "
                 "chains still fit to keep the GPU busy. The design implication is concrete: a "
                 "fair-compute workload must size the per-worker scratchpad past that window "
                 "(%s here), or a GPU farm reclaims a real edge."
                 % (large[2], large[0], 1.0 / large[2] if large[2] else 0,
                    peak[2], peak[0], floor_txt))
        L.append("")
        L.append("This is an engineering result on **one** GPU architecture, **one** memory "
                 "budget (see workers column), **one** implementation, and **one** machine. It "
                 "is enough to turn scratchpad size from an arbitrary knob into a studied "
                 "variable, and to motivate the right next question — *where does this "
                 "transition sit on other hardware?* — rather than the overbroad *do GPUs "
                 "always lose?*. Generalizing needs Phase 2B (a data-center GPU) and a "
                 "budget-varying sweep.")
    elif not wins:
        L.append("**Strongest support: the GPU never beats the CPU at any tested size.** "
                 "The latency-bound chained workload denies the parallel device an advantage "
                 "across the whole range on this hardware.")
    elif large[2] >= 1.0:
        L.append("**Partial/negative result:** the GPU retains an advantage (%.2f×) even at "
                 "the largest scratchpad size. On this hardware, a large working set does not "
                 "by itself neutralise the parallel device; the fairness thesis is not "
                 "supported for this workload/parameter range." % large[2])
    else:
        L.append("Mixed result — see the table; the GPU wins in some regimes and loses in "
                 "others. No clean single-sentence conclusion.")
    L.append("")
    L.append("## Scope and caveats")
    L.append("")
    L.append("- One machine, one GPU (%s) vs one CPU (%s, %d threads). An integrated "
             "GPU is not a data-center GPU; a discrete A100/H100 has far more memory and "
             "bandwidth and may behave differently. That is Phase 2B (CUDA)." % (
                 gpu_dev, cpu_model, cpu_threads))
    L.append("- Throughput here is aggregate dependent-steps/sec across all workers, timed "
             "by wall clock on both devices (GPU: around kernel commit→complete; CPU: around "
             "the parallel run over all cores). Init and allocation are excluded on both.")
    L.append("- This tests parallel throughput, not single-chain latency (Phase 1) and not "
             "adversarial Sybil replication (Phase 3).")
    L.append("")
    L.append("## Reproducibility")
    L.append("")
    L.append("- GPU: %s" % gpu_dev)
    L.append("- CPU: %s (%d threads)" % (cpu_model, cpu_threads))
    L.append("- toolchain: %s" % sample["host"].get("toolchain", "?"))
    L.append("- target: %s" % sample["host"].get("target", "?"))
    L.append("")

    with open(os.path.join(out_dir, "phase2.md"), "w") as f:
        f.write("\n".join(L))

    advs = [d["advantage_gpu_over_cpu"] for d in rows]
    print("gpu:", gpu_dev)
    print("cpu:", cpu_model, "(%d threads)" % cpu_threads)
    print("sizes: %d | peak %.2fx | largest-size %.2fx"
          % (len(rows), max(advs), rows[-1]["advantage_gpu_over_cpu"]))
    print("wrote:")
    for ext in ("md", "csv", "json"):
        print("  " + os.path.join(out_dir, "phase2." + ext))


if __name__ == "__main__":
    main()
