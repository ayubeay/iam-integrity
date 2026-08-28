# RESERVE — Instrument Admissibility Envelope (IAE)

Status: RESERVED — research / future architecture. NOT an active build.
**NOT authorization to deploy capital. NOT a directive to build an options exchange.**
Captured: 2026-08-27.

Distinct from `capital-admissibility-framework.md`, which governs whether *capital and
project execution* are admissible. IAE asks whether the **financial instrument and the
proposed obligation itself** are machine-understandable and admissible.

## Core question

How should an autonomous or human-directed execution system determine whether a
financial instrument — and a specific proposed position in it — is admissible *before*
execution?

## Pipeline

    intent → identify instrument → identify underlying exposure
    → reconstruct economic obligation → verify market/protocol
    → verify pricing/oracle inputs → collateral requirements
    → bounded/unbounded loss → liquidity → expiry/exercise mechanics
    → settlement → counterparty/protocol risk → jurisdiction/policy
    → portfolio exposure → vLOID admissibility
    → ALLOW / THROTTLE / DEFER / DENY → execution → receipt
    → settlement verification

## Envelope fields (schema deliberately unfrozen)

instrument identity/type · underlying · venue/protocol · network · contract identifiers ·
direction · strike · expiry · exercise style · premium · size · collateral · maximum
theoretical loss · maximum modelled loss · payoff structure · liquidation mechanics ·
settlement method · oracles · oracle freshness · oracle disagreement · liquidity/depth ·
expected slippage · counterparty exposure · contract risk · concentration · correlation ·
jurisdiction/policy · execution permissions · confidence · timestamp · provenance.

## Defined-risk principle

**Transaction value ≠ economic exposure.**

Two $100 transactions can create radically different risk. Governance must reason about
payoff topology, maximum loss, conditional obligations, time dependence, collateral,
liquidity, settlement dependencies and failure modes — not asset names and amounts.

## Relationship to existing canonical reserves

`capital-admissibility-framework.md` · `governed-capital-eligibility.md` ·
`domain-aware-capital-intelligence.md` · `execution-liquidity-intelligence.md` ·
`private-agent-trading-infrastructure.md` · `tokenized-securities-authority.md` ·
`executable-asset-semantics.md` (the asset's own lifecycle semantics; IAE consumes them)
· VERITY (venue, oracle, instrument metadata provenance) · DRIFT (market, liquidity,
volatility, oracle regime changes invalidating a prior assessment) · IAM · OROS ·
vLOID (final boundary) · Universal Money Router (settlement, later).

## Research questions

Can arbitrary instruments be normalized into a common machine-readable risk envelope?
How much true economic exposure is determinable before execution? How should oracle
uncertainty propagate into admissibility? When should changing conditions invalidate a
previously approved envelope? Can maximum-loss guarantees be independently verified? How
should protocol-failure risk combine with market risk? Can one framework cover spot,
options, futures, lending, swaps, prediction markets, structured products and tokenized
instruments? Can receipts prove not merely that a transaction occurred but **why its risk
was considered admissible at that moment**?

## Boundary

Does not authorize real-money derivatives trading, leverage, options deployment, copying
another protocol, adding options to Momentum Sniper, building an exchange, or exposing
proprietary architecture to obtain external beta access.

## Activation

Revisit when HELIX expands beyond spot execution; external protocol integrations require
instrument-level governance; Momentum Sniper has prospective evidence justifying research
into alternative exposure structures; Universal Money Router interacts with instruments
rather than transfers; an agent requires governed authority over complex financial
obligations; or a partner provides a useful sandbox.

## Strategic principle

The opportunity is not "build every financial product." It is *"build infrastructure
capable of understanding what an execution actually obligates us to, before deciding
whether that execution is admissible."*

RESERVED. NO CAPITAL DEPLOYMENT. DO NOT BUILD.
