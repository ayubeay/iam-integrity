# RESERVE - Provider Qualification & Workload-Aware Routing

**Status:** Reserved architectural refinement. Not a standalone project.
**Extends:** survivor-oracle/docs/RESERVE_infrastructure_provider_intelligence.md and
api-connect RESERVE.md section 12. This file adds workload-awareness and the qualification
model; it does not restate those.

Two separate reserve drafts arrived describing this - one framed as provider benchmarking,
one as normalized capability access. They are the same architecture from two angles, merged
here rather than filed twice.

## Purpose
External providers should progressively become measurable, replaceable, routable execution
substrates rather than hard-coded dependencies. Covers Solana JSON-RPC, WebSockets,
Yellowstone gRPC, transaction relays, Jito or equivalent, indexers, AI inference, cloud and
edge infrastructure, and data APIs.

Named providers are examples, never architectural dependencies.

## Responsibility boundaries
    API CONNECT     how do I reliably access this capability?
                    normalized interfaces, adapters, caching, freshness metadata,
                    health and latency telemetry, capability discovery
    KONIGO CONNECT  which viable path or provider should carry it now?
                    continuity, failover, regional routing, fallback ordering
    DRIFT           has the environment changed enough to invalidate the assumption?
    VERITY          how much should this provider or result be trusted?
    OROS / vLOID    may this execute at all?

API Connect does not decide admissibility. Routing does not bypass governance. Do not
collapse these.

## Qualification model
Median latency, p95 and p99 tail, error rate, timeout rate, availability, throughput, stream
stability, reconnect behaviour, slot lag and freshness, dropped events, quota state, rate
limits, region, regional performance, supported capabilities, cost per workload unit,
historical reliability.

Extensible - different capability classes need different criteria.

## Workload-aware, not one global score
    MomentumSniper / HELIX   latency, freshness, stream stability, submission performance
    SURVIVOR / VERITY        completeness, reliability, provenance, consistency
    AI workloads             capability, inference latency, context limits, cost, region

    workload requirements -> eligible providers -> normalized observations
    -> VERITY and DRIFT signals -> continuity selection -> admissibility -> execution
    -> receipt

## AI inference as a capability family
Prefer, eventually:

    inference.request(capability="reasoning", latency_budget=..., jurisdiction=...)

over provider-specific calls scattered through application code. But do not force
lowest-common-denominator abstractions - expose provider-specific capability where
abstracting it would destroy something useful.

## Jurisdiction as a routing dimension
A request may eventually carry required or prohibited jurisdictions, residency
requirements, latency budget and trust floor. That allows future local infrastructure -
including Nigeria-based - to become an eligible substrate without redesigning anything
above it. Not a commitment to build hardware.

## What the quota incident actually showed
    RPC quota exhaustion -> SURVIVOR degraded with retries -> mint-path latency rose
    -> OROS /events slowed -> MomentumSniper crossed its 12s timeout -> governance failures

Changing the provider restored scoring and cut the failures. Verified: BONK 67, scoring
0.5.3, coverage 65%.

**It removed roughly 2.6 seconds. Approximately 4.9 seconds of intrinsic mint-path work
remained.** So this does NOT prove routing solves the latency problem. It proves something
narrower and still important: a provider dependency can degrade several downstream systems,
and replacing it can restore them without touching the applications.

The intrinsic latency is a separate engineering problem and must not be assumed solved by
failover.

## Implementation doctrine
A capability becomes eligible for normalization when there is evidence: multiple credible
providers exist, provider failure creates real downstream risk, quota exhaustion is
operationally relevant, switching recurs, jurisdiction matters, cost or latency differences
are material, or provider coupling is creating maintenance burden.

Evidence-driven extraction, not architecture-first rewriting.

## Non-goals
No standalone AI router. Do not rename API Connect around AI, merge it with KONIGO, move
governance into it, hard-code any provider, prematurely migrate working providers, or
rewrite stable services to satisfy this reserve.

## Principle
    DO NOT OWN EVERY RAIL.
    OWN THE INTELLIGENCE, GOVERNANCE, CONTINUITY AND EVIDENCE
    THAT DETERMINE WHICH RAIL SHOULD BE USED.

---

## Extension — 2026-08-27: Intelligence Supply Continuity + Social-Evidence Provider Validation

This reserve already owns provider evaluation and workload-aware routing. Two mechanisms
arrived that belong here rather than under a new parent.

### Intelligence Supply Continuity

Provider availability is an execution-continuity concern, not only a procurement concern.

    provider unavailable / degraded → KONIGO evaluates available path
    → alternative provider selected → VERITY evaluates its evidence independently
    → Information Admissibility reassesses → CONTINUE / DEGRADE / DEFER / ESCALATE / DENY

**A fallback provider does not inherit another provider's trust or evidentiary status.**
Continuity of supply must not become continuity of assumed credibility.

### Social-Evidence Provider Validation Case

Testbed: Momentum Sniper. When a consequential agent first requires social intelligence
beyond its existing sources, evaluate external agent-native social retrieval providers
**before** building proprietary ingestion infrastructure.

The objective is not "social listening." It is to determine whether an additional
information supplier provides measurable decision value while preserving sufficient
evidence integrity for governed use.

    social / community sources → authorized provider
    → normalized representation + raw-evidence / provenance reference
    → VERITY → Information Admissibility → domain intelligence → vLOID / policy
    → authorized action → receipt

**A provider's normalized response must not silently become primary truth.** Preserve or
make retrievable: source/platform · source object ID or URL · author identity ·
observed_at · source timestamp · thread relationships · edit/deletion state · engagement
context · provider identity · retrieval method · normalization/transformation history ·
raw evidence reference · freshness. Normalization reduces tokens and simplifies reasoning,
but compression can destroy information relevant to consequential decisions. Prefer
**normalized representation + provenance pointer** over normalized representation treated
as evidence. See `evidence-lifecycle-state-provenance-envelope.md`.

Epistemic distinctions to preserve:

    POST DELETED           ≠ POST NEVER OBSERVED
    NORMALIZED COPY EXISTS ≠ UNDERLYING CLAIM VERIFIED
    HIGH ENGAGEMENT        ≠ HIGH RELIABILITY
    MULTIPLE POSTS         ≠ MULTIPLE INDEPENDENT SOURCES
    PROVIDER RETURNED DATA ≠ DATA ADMISSIBLE FOR EXECUTION

### Validation contract

Do not add a social feed merely because more information is available. **A proposed source
must earn its edge.** Compare baseline against baseline + candidate evidence on measures
appropriate to the active strategy — win rate, expected and median return,
catastrophic-loss rate, entry precision, false-positive rate, regime-specific performance,
timing advantage, signal redundancy, coverage, latency, freshness, cost.

    BASELINE → outcome distribution
    CHALLENGER (baseline + candidate evidence) → outcome distribution
    COMPARE → out-of-sample edge?  NO → reject / remain external
                                   UNCERTAIN → continue observation
                                   YES → candidate architectural edge

Also determine *where* an edge exists — a source with no global improvement may still
carry information under a particular regime, asset class, liquidity condition or event
stage. Do not promote that observation beyond the evidence.

### Four evaluation dimensions

**Capability** (sources and operations supplied) · **Evidence integrity** (traceability of
normalized results) · **Operational quality** (freshness, latency, availability, coverage,
cost, continuity) · **Decision contribution** (measurable improvement to the consequential
system). Provider capability alone does not establish architectural value.

### Invariants

**Everything may be connected. Every connection must earn its edge.**
**More information is not automatically more intelligence.**
**Normalized information is not automatically admissible evidence.**

### Activation

Activate the validation case only when a consequential agent genuinely requires evidence
beyond its existing source; a candidate provider exposes an authorized integration
surface; sufficient receipts exist to evaluate incremental decision value; and provenance
characteristics can be inspected rather than assumed. No provider integration, Momentum
Sniper modification, new module or production dependency follows from this entry.
