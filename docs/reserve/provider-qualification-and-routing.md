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

---

## Extension 2026-08-29 — Dependency Trust Degradation & Obligation Preservation

Status: Reserved architectural refinement. Not a standalone project.

### Why this belongs here and not in its own file

This reserve already owns dependency trust and already states that *a fallback provider
does not inherit another provider's trust or evidentiary status.* What it did not carry is
what happens to **obligations already accepted** when a dependency degrades. That is a
consequence of the dependency relationship this reserve governs, so it is recorded here.

The graduated-degradation pattern itself is **not new and is not restated**. It exists in
three domains already:

    emaa-external-machine-action-admissibility.md   TRUSTED → STANDARD → FLAGGED
                                                    → RESTRICTED → QUARANTINED
    counterfactual-execution-governor.md            NORMAL → RESTRICTED → SAFE_MOTION_ONLY
                                                    → LOCAL_CONTROL_ONLY
                                                    → HUMAN_CONFIRMATION_REQUIRED
                                                    → PROTECTIVE_STOP
    iam-external-identity-risk-signals.md           NORMAL → WATCH → ELEVATED
                                                    → RESTRICTED → COMPROMISED → RECOVERY

What this extension adds is the same shape applied to a **consumed dependency**, plus the
attribution rule below.

### Three states that must not collapse

    AVAILABLE  ≠  TRUSTED  ≠  ADMISSIBLE

A dependency may respond and be untrustworthy. It may be trustworthy and inadmissible for
this execution under current policy. Availability is the weakest of the three and the
easiest to measure, which is why it is the one most often mistaken for the others.

### Protective pause is not shutdown

`counterfactual-execution-governor.md` already states the principle for embodied systems —
*do not treat every anomaly as requiring total shutdown* — and it holds here. Reducing the
affected execution surface, preserving reversible activity, resolving trust, then
restoring or restricting is a different operation from stopping.

    reduce affected surface → preserve reversible activity → resolve trust
    → RESTORE / RESTRICT / TERMINATE

A system that can only continue or halt will halt too much, and the cost of halting is
borne by whoever depended on it.

### Obligation preservation

**A dependency failure is a state-transition attribution question, and the attribution is
frequently made wrongly.**

    provider outage
    → execution unavailable
    ≠ participant default
    ≠ participant misconduct
    ≠ participant abandonment

**Do not convert infrastructure failure into participant failure.** Where an obligation was
already accepted and the infrastructure that would discharge it becomes unavailable, **the
obligation is not silently erased**, and our inability to execute must not by itself be
interpreted as participant breach, misconduct or abandonment.

    INFRASTRUCTURE FAILURE  ≠  EVIDENCE OF PARTICIPANT FAILURE

This is deliberately narrower than a finding about the participant. A participant may hold
independently assessable obligations under the governing policy or agreement. **Those are
evaluated from their own evidence — neither inferred from our dependency failure, nor
excused by it.** Two questions, separately answered:

    failure attribution arising from our own dependency
    any independent obligation the participant holds

The distinction that carries the operational weight:

    a new discretionary action    may be denied while trust is unresolved
    an existing obligation        is not extinguished by our inability to execute it

These are different decisions and a degraded posture should not silently apply the first
rule to the second. Grace, deferral and eventual fulfilment are dispositions; treating our
own outage as a mark against the counterparty is an attribution error.

`computable-accountability.md` records the causal evidence — obligation, dependency
failure, affected capability, inability to execute, attribution, disposition, eventual
fulfilment — so that the reason a participant was **not** attributed our failure remains
reconstructable. **The operational rule lives here; the causal record lives there. It is
not written twice.**

### Relationship to this reserve's invariants

*Every connection must earn its edge.* A degraded dependency is an edge that has stopped
earning it — but the participants on the other side of our obligations did not choose that
edge, and must not pay for its failure.

RESERVED. DO NOT BUILD.
