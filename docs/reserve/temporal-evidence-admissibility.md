# RESERVE — Temporal Evidence Admissibility / Signal Half-Life

Status: RESERVED — research/architecture. NOT an active build. **Not validated alpha.**
Captured: 2026-08-28.
**Child of `evidence-lifecycle-state-provenance-envelope.md` (ELSPE).** Subordinate and
complementary, not a competing owner.

## Boundary with ELSPE

    ELSPE  what state is this evidence in, where did it come from,
           and what did it establish at time T?

    THIS   how quickly does its decision-relevant usefulness decay,
           and at what future point is it no longer admissible for a given action?

ELSPE owns lifecycle state, provenance envelope, observability capability and temporal
*claim* semantics. This reserve owns decay rate, persistence, and the point of
inadmissibility. Neither restates the other.

## Core thesis

A signal is not trustworthy merely because strong historical evidence once supported it.
Market relationships, correlations, behavioural signals, liquidity conditions,
counterparties and regimes decay.

    "Was this relationship historically demonstrated?"
    is not
    "Was there sufficient evidence, available at this exact decision time,
     that the relationship was still alive?"

**Historical validity ≠ present admissibility.**

Origin: a pairs-trading result in which apparently strong relationships largely lost
predictive usefulness once acceptance decisions were reconstructed using only information
available during each historical period. The lesson generalizes far past pairs trading.

## Signal lifecycle

    DISCOVERED → CONFIRMED → ACTIVE → WEAKENING → DECAYING → INVALID

Transitions are evidence-driven, not fixed expiry. State variables: signal age · evidence
timestamp · last confirmation · recent confirmation strength · historical strength ·
recency-weighted strength · decay velocity · regime consistency · structural-break
probability · persistence probability · observation density · contradictory evidence ·
source freshness · confidence trajectory.

## Signal half-life

The expected period over which evidence supporting a relationship loses enough predictive
relevance that continued execution requires renewed confirmation.

**Not necessarily a fixed clock.** It may depend on regime, volatility, liquidity,
participant behaviour, asset class, signal type, structural events, information arrival
rate, and adversarial adaptation.

## Five separated times

    EVENT TIME        when something happened
    OBSERVATION TIME  when the system observed it
    KNOWLEDGE TIME    when it became legitimately usable
    DECISION TIME     when execution was authorized
    EXECUTION TIME    when the action actually occurred

Every decision must be reconstructable using only what was knowable at its decision
timestamp. **No future information may leak backward** through revised datasets,
classifications, universe membership, rankings, thresholds, regime labels, acceptance
criteria, liquidity statistics, model selection or parameter tuning.

Receipt fields: decision_timestamp · signal_timestamp · source_observation_timestamp ·
data_availability_timestamp · model_version · signal_state · signal_age · regime_state ·
recent_confirmation · decay_score · admissibility_decision · reason_codes.

## Relationship to existing canonical reserves

ELSPE (parent) · DRIFT and `regime-evidence-engine.md` (is the supporting environment
changing?) · VERITY (evidence confidence) · JANUS (regime interpretation) · vLOID
(present admissibility) · OROS · `executable-capacity-thinnest-leg.md` (the complementary
execution-physics question) · `computable-accountability.md`.

## Research questions

Can relationship persistence be modelled directly rather than repeatedly applying binary
discovery tests? Which decay measurements identify dying signals earliest without
overreacting to noise? Should signal families carry different half-life models? Can
survival or hazard models estimate the probability an edge remains alive? Can change-point
detection identify structural death earlier? How should contradictory evidence affect
admissibility? Can receipts expose hidden look-ahead bias during research audits?

## Boundary

This is an **evidence-governance mechanism**, not a claim that any statistical technique
produces trading profits. It applies wherever previously valid information may become
stale: fraud intelligence, counterparty trust, cybersecurity, procurement, robotics,
infrastructure, market intelligence, agent decision systems, and Signal Drift scenario
design.

RESERVED — DO NOT BUILD.
