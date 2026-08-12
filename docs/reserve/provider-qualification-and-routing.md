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
