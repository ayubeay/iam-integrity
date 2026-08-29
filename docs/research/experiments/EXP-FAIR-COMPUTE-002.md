# EXP-FAIR-COMPUTE-002 — Fair Useful Compute, Phases 2B / 2C / 3

Status: `PROPOSED` · Phase 2B `BLOCKED` (capital)
Registered: 2026-08-29
Migrated from: `docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md` · C2
Reserve: `docs/browser-fair-compute-reserve.md`
Prior phases: Phases 1 and 2A are banked in `docs/research/EXPERIMENT_LEDGER_2026-08-27.md` as E8/E9.

## Hypothesis

The hardware-advantage transition region observed on Apple Silicon appears on materially
different architectures, and survives participant economics.

E8 validated narrowly: Fair Compute found *where* the browser/native transition sits, not
who wins. This asks whether that boundary is a property of the workload or of the machine.

## Phases, concluding separately

**2B — cross-architecture.** Does a comparable transition region reproduce on a non-Apple
accelerator? **`BLOCKED`: requires hardware access not currently available.** The blocker
is capital, and it is recorded rather than worked around.

**2C — parameter surface.** Does the transition map as a surface rather than a single
point across workload parameters? Runnable on existing hardware.

**3 — participant economics.** Under simulated heterogeneous participants, does reward
concentration stay within a stated bound? Runnable on existing hardware.

## Accept / reject

**2B rejected** if the transition is machine-specific — which would be a substantive
finding about E8's evidence boundary, not a failure.

**3 rejected** if one operator can convert capital into identities or hardware and recover
a disproportionate share regardless of workload design. That result would constrain the
whole Fair Useful Compute direction and is the most decision-relevant of the three.

## Evidence boundary

Simulation and benchmarking only. **No live network, no token, no participants.** Phase 3
models participant behaviour; it does not observe it.

## Activation

Per the reserve's activation list. 2C and 3 need no new resources; 2B stays `BLOCKED`
until accelerator access exists.

## Provenance

    source artifact:       docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md (C2);
                           E8/E9 in EXPERIMENT_LEDGER_2026-08-27.md
    registered:            2026-08-29
    implementation commit: bench/fair-compute-bench, bench/fair-compute-bench-gpu
    evidence boundary:     one machine class for E8/E9; 2B unrun
    conclusion date:       pending
