# RESERVE — Execution Path Viability (+ JANUS cross-reference)

Status: RESERVED / NOT ACTIVE BUILD.
Parent: execution-governance measurement primitive (feeds vLOID/OROS/receipts). JANUS is NOT owned here — see cross-reference.
Captured: 2026-08-23. Motivating evidence: the 2026-08-23 Momentum inspection (positive mean / negative median / fat tail; no prospective MAE/MFE telemetry). Momentum is cited as motivation only — **this reserve must not alter or pre-judge the Path A experiment.**

## What is genuinely new

Evaluating an execution by the **state trajectory required to reach the outcome**, not only its terminal result; and the deeper **execution-envelope / survival-margin** abstraction beneath domain-specific excursion metrics. No existing reserve owns this.

## Terminology rules (preserve exactly)

- **Conventional MAE/MFE remain the raw, interoperable terminology** (Maximum Adverse / Favorable Excursion) in interfaces and data.
- **MADE / MAFE (Maximum Adverse/Favorable Dynamic Excursion) remain PROVISIONAL** — they become canonical only if a later review establishes that "dynamic" adds genuine semantics rather than renaming established metrics. Do not treat MADE/MAFE as canonical merely because the names are appealing.
- **Path Viability is separate from JANUS.**

## Doctrine

A terminally successful execution is not necessarily *viable* if reaching success required passing through states exceeding the available risk/capital/latency/safety/resource/policy/operational envelope; a terminal failure may have contained large uncaptured favorable excursion. Distinguish: outcome truth · adverse-path truth · favorable-path truth · survival truth (stayed inside the permissible envelope?) · capture truth (how much favorable path was realized). Generalized question: *between authorization and terminal outcome, what states did the execution traverse, and could the governing system tolerate that path?* Domain-general, not trading-only (trading is a forcing function): trading, agent execution, transaction routing, network/provider execution, logistics, robotics, fulfillment, financing.

**Execution-envelope abstraction (the deeper primitive):** express path viability as **distance from an acceptable, multidimensional execution envelope** (capital, latency, memory, queue depth, throughput, error rate, retries, liquidity, slippage, network quality, energy, physical safety, policy/authorization limits, deadline, resource availability). Do not force all domains into percentage-based MAE/MFE: financial→price/PnL excursion; network→latency/loss/retries; event infra→queue/memory/latency/loss; robotics→force/trajectory/energy/collision margin/timing; fulfillment→deadline/inventory/capacity. Common primitive = trajectory relative to the permissible envelope; survival margin = distance from boundaries. Example: Strategy A final +18% with −42% adverse path vs Strategy B +14% with −7% adverse path — terminal ranking prefers A; path viability may prefer B (A may be impossible to capitalize/survive/authorize under the available envelope). This should eventually inform sizing/admissibility/policy/learning, not merely reporting.

Candidate receipt fields (reserve only): execution_id, start/terminal_state, terminal_outcome, mae, mfe, made/mafe (provisional), adverse/favorable_peak_ts, operating_envelope, envelope_breached, survival_margin, recovery_distance, favorable_capture_ratio, path_viable, path_viability_reason.

## JANUS — cross-reference, DO NOT redefine

The canonical JANUS is the existing **"JANUS dual-read validator"** reserve (`docs/RESERVED.md`): an adversarial reinterpretation / opposing-side second-read mechanism tied to ORA/OROS — the strongest admissible *competing* interpretation of the same evidence (not generic devil's advocate; agreement, disagreement, missing evidence, asymmetric confidence are all information). Path Viability may **provide evidence to** JANUS (e.g., Face B: "38% of those successes crossed today's available resource envelope") but does not own JANUS, and JANUS does not own MADE/MAFE. **Naming-collision note (do not fix in this batch):** `helixjanus/` and one staging line use "JANUS = classify the operating environment" (a regime-classifier), which collides with the canonical dual-read meaning; recorded as a documented architecture issue for a later bounded cleanup — not renamed here.

## Relationship (cross-reference, do not duplicate)

vLOID (admissibility may consume path evidence) · OROS (execution) · VERITY (source/evidence confidence) · DRIFT (envelope/viability changes with regime) · Receipts/HelixScan (reconstruct trajectory where receipts carry enough state — a receipt proves an action occurred; a *history* proves how it evolved; HelixScan exposes trajectory without becoming the governor) · Momentum Sniper/finance (forcing function; do not assign JANUS ownership without repo evidence) · JANUS dual-read validator (RESERVED.md) · Stream Execution Fabric (high-throughput event example is annotated there as evidence — batch 2).

## Non-goals

Not a trading-only primitive; MADE/MAFE not canonical yet; do not redefine JANUS; do not fix the HELIX-JANUS naming collision here; do not alter/pre-judge the Path A experiment; no implementation.

## Activation

Before building: (1) confirm the canonical JANUS reserve (done — RESERVED.md dual-read validator); (2) determine whether HELIX-JANUS implements part of the concept; (3) find existing MAE/MFE or path-state telemetry (Momentum currently has none — that's a gap, not a build order); (4) determine whether MADE/MAFE add genuine semantics; (5) identify a real consumer with a measurable path-viability problem; (6) extend an existing primitive rather than duplicate. Until then: RESERVE ONLY.
