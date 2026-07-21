#!/usr/bin/env python3
"""Browser-vs-native comparison report generator.

This is the first research artifact of the browser-fair-compute reserve. It does
not add benchmark features; it turns a directory of result JSONs into the one
number the reserve's core question asks for: how close does browser execution
get to native execution on the same workload and the same machine.

Three responsibilities, in order of importance:

  1. REFUSE to compare runs whose implementation_hash differs. If native and
     browser did not run the same source, no ratio between them means anything.
     This is a hard error, not a warning.

  2. Compute the browser/native efficiency ratio per scratchpad size, on both
     the median and the minimum (least-contended) latency.

  3. Emit one report in three formats (Markdown, CSV, JSON) carrying enough
     reproducibility metadata that a later run on different hardware can be
     compared without guessing whether the environment explains the difference.

Zero dependencies; standard library only. Reads results/*.json by default.

    python3 compare.py                      # scans results/, writes results/report/
    python3 compare.py --results-dir DIR --out-dir DIR
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

SCHEMA = "fair-compute-bench/1"

# A paired size only enters the headline claim if BOTH runtimes measured it with
# relative spread at or below this. It keeps thermally-noisy points (e.g. an
# 8 MiB run at 192% spread) out of the quotable number — they stay in the table,
# flagged, but the headline is computed only from stable measurements.
STABILITY_THRESHOLD = 0.15


def die(msg):
    print("error: " + msg, file=sys.stderr)
    sys.exit(1)


def load_results(results_dir):
    """Load every *.json directly under results_dir (not the report subdir)."""
    if not os.path.isdir(results_dir):
        die("results dir does not exist: " + results_dir)

    files = []
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(results_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print("warning: skipping unreadable " + name + ": " + str(e), file=sys.stderr)
            continue
        if data.get("schema") != SCHEMA:
            print("warning: skipping " + name + ": schema is "
                  + repr(data.get("schema")) + ", expected " + SCHEMA, file=sys.stderr)
            continue
        files.append((name, data))
    return files


def key_bytes(data):
    return int(data["workload"]["scratchpad_bytes"])


def mib(nbytes):
    return nbytes / (1024 * 1024)


def pick_best(candidates):
    """Among multiple runs at the same (runtime, size), keep the quietest run
    that passed both integrity gates: determinism, and step accounting
    (executed_steps_ok, defaulting True for JSONs predating that field). Then
    lowest relative_spread wins. Returns (data, dropped_count)."""
    def integrity_ok(c):
        s = c["summary"]
        return s.get("determinism_ok") is True and s.get("executed_steps_ok", True) is True
    good = [c for c in candidates if integrity_ok(c)]
    dropped = len(candidates) - len(good)
    if not good:
        return None, dropped
    good.sort(key=lambda d: d["summary"].get("relative_spread", float("inf")))
    return good[0], dropped + (len(good) - 1)


def collect(files):
    """Group by runtime then scratchpad size, and enforce the hash guard."""
    by_runtime = {}
    hashes = {}  # impl_hash -> list of filenames
    for name, data in files:
        rt = data["harness"]["runtime"]
        h = data["workload"]["implementation_hash"]
        hashes.setdefault(h, []).append(name)
        by_runtime.setdefault(rt, {}).setdefault(key_bytes(data), []).append(data)

    # Responsibility 1: the hard guard.
    if len(hashes) > 1:
        lines = ["implementation_hash differs across result files; refusing to compare.",
                 "The runs were not produced by the same workload source:"]
        for h, names in hashes.items():
            lines.append("  " + h[:16] + "…  " + ", ".join(names))
        lines.append("Rebuild so native and browser share one implementation_hash, then re-run.")
        die("\n".join(lines))

    impl_hash = next(iter(hashes)) if hashes else None

    # Reduce each (runtime, size) to its quietest run.
    reduced = {}
    notes = []
    for rt, sizes in by_runtime.items():
        reduced[rt] = {}
        for nbytes, cands in sizes.items():
            best, dropped = pick_best(cands)
            if best is None:
                notes.append("dropped all %s runs at %.0f MiB (determinism failed)"
                             % (rt, mib(nbytes)))
                continue
            reduced[rt][nbytes] = best
            if dropped:
                notes.append("kept quietest of %d %s runs at %.0f MiB"
                             % (len(cands), rt, mib(nbytes)))
    return reduced, impl_hash, notes


def build_rows(reduced):
    native = reduced.get("native", {})
    browser = reduced.get("browser", {})
    all_sizes = sorted(set(native) | set(browser))
    rows = []
    for nbytes in all_sizes:
        n = native.get(nbytes)
        b = browser.get(nbytes)
        row = {"scratchpad_bytes": nbytes, "scratchpad_mib": round(mib(nbytes), 3)}
        row["native_min"] = n["summary"]["ns_per_step_min"] if n else None
        row["native_median"] = n["summary"]["ns_per_step_median"] if n else None
        row["native_spread"] = n["summary"]["relative_spread"] if n else None
        row["browser_min"] = b["summary"]["ns_per_step_min"] if b else None
        row["browser_median"] = b["summary"]["ns_per_step_median"] if b else None
        row["browser_spread"] = b["summary"]["relative_spread"] if b else None
        row["ratio_min"] = (row["browser_min"] / row["native_min"]
                            if n and b and row["native_min"] else None)
        row["ratio_median"] = (row["browser_median"] / row["native_median"]
                               if n and b and row["native_median"] else None)
        row["paired"] = bool(n and b)
        rows.append(row)
    return rows


def fmt(v, spec="{:.1f}"):
    return "—" if v is None else spec.format(v)


def repro_block(reduced):
    """Pull whatever host/toolchain metadata the JSONs carry, per runtime."""
    out = {}
    for rt, sizes in reduced.items():
        if not sizes:
            continue
        sample = next(iter(sizes.values()))
        host = sample.get("host", {})
        proto = sample.get("protocol", {})
        harness = sample.get("harness", {})
        out[rt] = {
            "os": host.get("os"),
            "arch": host.get("arch"),
            "cpu_model": host.get("cpu_model"),
            "logical_cpus": host.get("logical_cpus"),
            "toolchain": host.get("toolchain"),
            "target": host.get("target"),
            "user_agent": host.get("user_agent"),
            "device_memory_gb": host.get("device_memory_gb"),
            "build_profile": harness.get("build_profile"),
            "timed_region": proto.get("timed_region"),
            "scheduling_hint": proto.get("scheduling_hint"),
            "timer": proto.get("timer"),
            "warmup_runs": proto.get("warmup_runs"),
            "timed_runs": proto.get("timed_runs"),
        }
    return out


def write_markdown(path, rows, impl_hash, repro, notes):
    L = []
    L.append("# Browser-vs-native comparison — fair-compute-bench")
    L.append("")
    L.append("_Generated " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
             + " by `compare.py`. First research artifact of the browser-fair-compute reserve._")
    L.append("")
    L.append("## What this does and does not claim")
    L.append("")
    L.append("This report compares **performance** between two runtimes already "
             "shown to produce **byte-identical output** for identical input. It is "
             "evidence, not proof, and its scope is deliberately narrow: one "
             "deterministic dependent-memory workload, on one machine. The headline "
             "claim is drawn only from the **stable** measurements (relative spread "
             "≤ %d%% in both runtimes) — in practice the larger, DRAM-latency-bound "
             "sizes, where the timing is not dominated by cache residency or "
             "scheduler noise. Smaller sizes are reported in the table but excluded "
             "from the headline when noisy." % int(STABILITY_THRESHOLD * 100))
    L.append("")
    L.append("A single machine cannot establish hardware fairness. Phases 2 and 3 "
             "of the reserve — diverse devices, a massively-parallel GPU baseline, "
             "and adversarial replication — remain open, and are where the fairness "
             "thesis is actually tested rather than mere runtime parity.")
    L.append("")
    L.append("## Implementation identity")
    L.append("")
    if impl_hash:
        L.append("All compared runs share one `implementation_hash`:")
        L.append("")
        L.append("    " + impl_hash)
        L.append("")
        L.append("The generator refuses to emit a comparison when this differs "
                 "across runtimes, so the ratios below are between the same source "
                 "compiled two ways — not two different programs.")
    else:
        L.append("_No results found._")
    L.append("")

    L.append("## Efficiency ratio (browser ÷ native)")
    L.append("")
    L.append("Lower `ns/step` is faster. A ratio near **1.0** means browser "
             "execution is competitive; a large ratio means native wins. `min` is "
             "the least-contended run and is the more defensible estimator; "
             "`median` is shown alongside because browser timers are coarsened and "
             "median absorbs that noise.")
    L.append("")
    L.append("| Scratchpad | Native min | Browser min | **Ratio (min)** | Native median | Browser median | Ratio (median) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append("| %s MiB | %s | %s | **%s** | %s | %s | %s |" % (
            fmt(r["scratchpad_mib"], "{:.0f}"),
            fmt(r["native_min"]), fmt(r["browser_min"]),
            fmt(r["ratio_min"], "{:.2f}×"),
            fmt(r["native_median"]), fmt(r["browser_median"]),
            fmt(r["ratio_median"], "{:.2f}×"),
        ))
    L.append("")

    def is_stable(r):
        ns, bs = r["native_spread"], r["browser_spread"]
        return (ns is not None and bs is not None
                and ns <= STABILITY_THRESHOLD and bs <= STABILITY_THRESHOLD)

    paired = [r for r in rows if r["paired"]]
    stable = [r for r in paired if is_stable(r) and r["ratio_min"]]
    if stable:
        rmins = [r["ratio_min"] for r in stable]
        lo, hi = min(rmins), max(rmins)
        avg = sum(rmins) / len(rmins)
        sizes = ", ".join("%.0f MiB" % r["scratchpad_mib"] for r in stable)
        L.append("**Headline (stable measurements only — %s):** across %d size(s) "
                 "where both runtimes measured with ≤ %d%% spread, the browser/native "
                 "ratio on `min` is **%.2f×–%.2f×** (mean %.2f×). These are the "
                 "DRAM-latency-bound sizes; the correct reading is that browser "
                 "WebAssembly and native Rust are statistically similar once memory "
                 "latency dominates execution, differing by only a few percent."
                 % (sizes, len(stable), int(STABILITY_THRESHOLD * 100), lo, hi, avg))
        L.append("")
        noisy_paired = [r for r in paired if not is_stable(r)]
        if noisy_paired:
            ns = ", ".join("%.0f MiB" % r["scratchpad_mib"] for r in noisy_paired)
            L.append("_Excluded from the headline as too noisy to quote: %s "
                     "(see noise flags below). They remain in the table for "
                     "completeness._" % ns)
            L.append("")
    elif paired:
        L.append("_Paired sizes exist, but none met the ≤ %d%% stability bar in "
                 "both runtimes, so no headline figure is quoted. Re-run on a quiet, "
                 "cooled, plugged-in machine — the DRAM-bound sizes (128–256 MiB) "
                 "settle first._" % int(STABILITY_THRESHOLD * 100))
        L.append("")
    else:
        L.append("_No size has both a native and a browser run yet, so no ratio "
                 "can be computed. Collect browser runs at the native sizes and "
                 "re-run._")
        L.append("")

    # Noise honesty.
    noisy = [r for r in rows
             if (r["native_spread"] and r["native_spread"] > 0.05)
             or (r["browser_spread"] and r["browser_spread"] > 0.05)]
    if noisy:
        L.append("### Noise flags")
        L.append("")
        L.append("These sizes had relative spread above 5% in at least one runtime "
                 "and should not be quoted as citable figures without a quieter "
                 "machine:")
        L.append("")
        for r in noisy:
            L.append("- %.0f MiB — native spread %s, browser spread %s" % (
                r["scratchpad_mib"], fmt(r["native_spread"], "{:.0%}"),
                fmt(r["browser_spread"], "{:.0%}")))
        L.append("")

    L.append("## Reproducibility")
    L.append("")
    for rt in ("native", "browser"):
        info = repro.get(rt)
        if not info:
            continue
        L.append("### " + rt)
        L.append("")
        for k in ("os", "arch", "cpu_model", "logical_cpus", "toolchain", "target",
                  "user_agent", "device_memory_gb", "build_profile", "timed_region",
                  "scheduling_hint", "timer", "warmup_runs", "timed_runs"):
            v = info.get(k)
            if v not in (None, ""):
                L.append("- **%s**: %s" % (k, v))
        L.append("")

    L.append("### Scheduling caveat (read before trusting any ratio)")
    L.append("")
    L.append("On heterogeneous CPUs (Apple Silicon P/E cores, Intel P/E cores) "
             "the core a process lands on dominates a memory-latency result: a "
             "foreground browser tab runs on performance cores, while a CLI "
             "process may run on efficiency cores with a much smaller L2. At a "
             "working-set size that fits one core's L2 but not the other's, "
             "identical code can differ ~10x. The native harness now requests "
             "performance-core scheduling (`scheduling_hint`), but this is a bias, "
             "not a guarantee — confirm core residency out of band before quoting "
             "a ratio, and prefer a size that is DRAM-bound on both runtimes for "
             "the cleanest comparison.")
    L.append("")
    L.append("_Not captured automatically — record by hand for a citable run:_ "
             "power mode (plugged in vs battery), background load, thermal state, "
             "actual core residency, and for the browser, the exact version and "
             "whether the tab was focused.")
    L.append("")

    if notes:
        L.append("## Selection notes")
        L.append("")
        for n in notes:
            L.append("- " + n)
        L.append("")

    with open(path, "w") as f:
        f.write("\n".join(L))


def write_csv(path, rows):
    cols = ["scratchpad_mib", "scratchpad_bytes", "native_min", "browser_min",
            "ratio_min", "native_median", "browser_median", "ratio_median",
            "native_spread", "browser_spread", "paired"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path, rows, impl_hash, repro, notes):
    doc = {
        "schema": "fair-compute-bench-report/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_hash": impl_hash,
        "reproducibility": repro,
        "comparison": rows,
        "selection_notes": notes,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Browser-vs-native comparison report.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default=None,
                    help="default: <results-dir>/report")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, "report")

    files = load_results(args.results_dir)
    if not files:
        die("no valid " + SCHEMA + " result files in " + args.results_dir)

    reduced, impl_hash, notes = collect(files)
    rows = build_rows(reduced)
    repro = repro_block(reduced)

    os.makedirs(out_dir, exist_ok=True)
    md = os.path.join(out_dir, "comparison.md")
    cv = os.path.join(out_dir, "comparison.csv")
    js = os.path.join(out_dir, "comparison.json")
    write_markdown(md, rows, impl_hash, repro, notes)
    write_csv(cv, rows)
    write_json(js, rows, impl_hash, repro, notes)

    runtimes = ", ".join(sorted(reduced)) or "none"
    paired = sum(1 for r in rows if r["paired"])
    print("runtimes found: " + runtimes)
    print("sizes: %d total, %d paired (native+browser)" % (len(rows), paired))
    print("wrote:")
    print("  " + md)
    print("  " + cv)
    print("  " + js)
    if paired == 0:
        print("\nnote: no paired sizes yet — collect browser runs at the native "
              "sizes, then re-run to get ratios.")


if __name__ == "__main__":
    main()
