# RESERVE — Executable Capacity / Thinnest-Leg Constraint

Status: RESERVED — research/architecture. NOT an active build. **Not validated alpha.**
Captured: 2026-08-28.
**This reserve owns the thinnest-leg primitive.** Other reserves consume it —
`demand-sovereign-market-infrastructure.md` most directly.

## Core thesis

**A valid opportunity is not necessarily an executable opportunity.**

Aggregate volume, average liquidity, quoted liquidity and historical capacity can
materially overstate what can actually be executed at the moment an opportunity appears.

## The primitive

**A system's executable capacity at a moment is bounded by the scarcest required
resource, not by aggregate capacity elsewhere in the system.**

    Executable Capacity(t) ≈ MIN(
        capacity of required component A,
        ratio-adjusted capacity of B,
        …,
        capacity of required component N
    )

subject to location, timing, qualification, price, policy, risk and execution
dependencies. Implementation must account for conversion and hedge ratios.

## Distinctions that collapse in ordinary analysis

    daily volume        ≠ executable depth
    average liquidity   ≠ instantaneous liquidity
    signal strength     ≠ execution capacity
    theoretical edge    ≠ realizable edge
    registered users    ≠ available qualified counterparties

## Capacity is time-local

A capacity estimate calculated hours earlier may itself be stale. Capacity observations
carry timestamp · TTL · confidence · route · venue/source · depth assumptions · slippage
assumptions · maximum admissible size.

    opportunity detected → signal admissible → route discovery
    → instantaneous capacity assessment → proposed sizing → impact/slippage simulation
    → weakest-path constraint → execution admissibility
    → execute / throttle / defer / deny → receipt → realized slippage feedback

A candidate may carry HIGH signal confidence and LOW executable capacity, and correctly
receive THROTTLE, DEFER or DENY rather than ALLOW at the proposed size.

## Receipt extension

proposed_size · approved_size · capacity_estimate · capacity_timestamp · capacity_TTL ·
binding_constraint · expected_slippage · realized_slippage · route · liquidity_source ·
reason_for_throttle · execution_result.

This separates *what signal did we see* from *how much of that opportunity could actually
have been captured*.

## Generalization beyond trading

Payments → weakest settlement rail. Network routing → weakest path or provider. Supply
chains → constrained supplier or logistics step. Robotics → unavailable capability or
resource. Marketplaces → scarce side of *local* liquidity at the required time,
qualification and place. Procurement → constrained supplier, capital or approval step.
Agent workflows → least available required tool. Compute → scarcest admissible node.

Twenty thousand registered handymen nationally create no liquidity for a customer needing
one qualified plumber within ten miles tomorrow at 2 PM if none is available.

This is a candidate general execution-capacity primitive for OROS-style multi-resource
execution, not a trading-specific measure.

## Relationship to existing canonical reserves

`execution-liquidity-intelligence.md` · `execution-path-viability.md` ·
`execution-placement-engine.md` · `provider-qualification-and-routing.md` ·
`temporal-evidence-admissibility.md` (the complementary epistemic question) ·
`demand-sovereign-market-infrastructure.md` (consumer) · OROS · vLOID.

## Relationship to Temporal Evidence Admissibility

Two different questions, deliberately separate:

    TEMPORAL EVIDENCE ADMISSIBILITY   is the evidence supporting this still alive?
    EXECUTABLE CAPACITY               even if valid, how much can execute right now?

A system can be entirely correct about an opportunity and still lose because it
misunderstood capacity. Conversely, enormous liquidity is worthless if the signal has
already died.

## Activation

Reserve only. Revisit when a live or shadow execution system requires capacity-aware
sizing, or when a multi-resource execution path exhibits a bottleneck that aggregate
metrics fail to predict.

RESERVED — DO NOT BUILD.
