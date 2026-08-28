# RESERVE — SportGPT Intelligence Layer + Market-Divergence Intelligence

Status: RESERVED — product/architecture direction. NOT an active build.
Captured: 2026-08-27.
**Staging boundary:** the separately staged EventPulse and SportGPT material in
`staging/reserves-2026-07-13.md` retains staging status and is NOT promoted by this file.
Discovery is not promotion.

## Positioning

SportGPT Intelligence is not EventPulse with a sports skin, not "AI sports picks," and not
a sports chatbot. EventPulse may provide underlying event/signal infrastructure; SportGPT
Intelligence interprets those signals in sporting context, compares them with models and
markets, explains disagreement, tracks what was knowable at decision time, and measures
whether the intelligence remained useful prospectively.

## Origin observation

A soccer-model builder preserved a genuinely untouched holdout season, evaluated ~1,700
matches, and found **de-vigged bookmaker probabilities beat every tested model** on log
loss, Brier and RPS, with frozen strategies remaining negative (~−12.5% ROI). Rather than
adding variables until a backtest looked good, they proposed running prospectively with
timestamped odds, injuries and confirmed lineups, and asking where a model disagrees with
the market **for a defensible reason**.

The architectural question that survives:

    When our model disagrees with a strong market baseline, can we explain why using
    only information actually available at that moment, and can that disagreement
    survive prospective testing?

## Core loop

    event → baseline → market state → context → new events/signals → admissibility
    → model state → market/model divergence → explanation → confidence
    → decision-time receipt → outcome → calibration

## Capability areas

**Market** — implied and de-vigged probabilities, consensus, dispersion, opening/current/
closing movement, repricing rate, cross-market inconsistencies, whether SportGPT leads,
follows or lags. *Bookmaker disagreement does not mean the bookmaker is wrong; markets are
powerful aggregators and benchmarks.*

**Team / player** — injuries, suspensions, confirmed and expected lineups, goalkeeper
changes, workload, rest, transfers, manager changes, availability uncertainty. Distinguish
CONFIRMED · PROBABLE · RUMORED · INFERRED · UNKNOWN; do not flatten evidence qualities
into one feature.

**Matchup** — pressing vs buildup, transition profiles, aerial and set-piece mismatches,
formation interactions, defensive line behaviour, style conflicts. Why *these* opponents
may interact differently than generic strength estimates imply.

**Environment** — travel, rest differential, congestion, weather, altitude, surface,
venue, referee tendencies, tournament incentives. These do not automatically become model
variables; they earn inclusion through evidence.

**Events** — source event → EventPulse → timestamp → provenance → classification →
relevance → admissibility → affected entity → intelligence-state update.

## Divergence intelligence

Where does SportGPT disagree with the market, by how much, and why — with contribution
attribution across signal families and an explicit unexplained residual.

**DIVERGENCE ≠ OPPORTUNITY.** A divergence is interesting only if material, explainable,
evidence-backed, timestamp-valid, calibrated and robust. Never imply "model disagrees with
bookmaker, therefore bet."

## Explanation, counterfactual and model intelligence

Why the forecast moved; what information caused it; how much each family contributed; what
remains unexplained. What the forecast would be without a given injury, with the expected
lineup, with the goalkeeper available; what evidence would reverse the position. And
SportGPT's own calibration, Brier, log loss, RPS, confidence-bucket and league-specific
performance, model vs market, model vs closing market, signal-family usefulness,
degradation and drift. **Do not hide poor performance;** a model that loses to the market
still generates useful research information.

## No retrospective intelligence leakage

A historical intelligence state must use only information available at that timestamp.
Never reconstruct "what SportGPT would have known" using later information — **even when
doing so would make the model look better.**

    prediction 13:00 · injury confirmed 14:10 · lineup 17:30 · kickoff 18:00
    The 13:00 receipt must not contain the 14:10 or 17:30 information.

## Information Advantage Half-Life

New sporting information does not remain novel indefinitely: NEW → PARTIALLY KNOWN →
MARKET REACTION → BROAD ABSORPTION → STALE. If the market reprices a confirmed lineup
immediately, SportGPT must not continue describing it as undiscovered advantage. Question:
*how long did this information remain non-consensus?*

## Prospective validation

Freeze model/version → record prediction → timestamp evidence → snapshot market → preserve
rationale → wait → record outcome → compare → append receipt. Avoid historical feature
mining presented as validated intelligence. Commitment mechanism:
`prospective-claim-commitment.md`.

## Negative evidence

Preserve failures — wrong divergences, immediate market corrections, features that worked
historically and failed prospectively, signals already fully priced. Do not optimize them
out of history.

## Safety / product boundary

Intelligence and betting execution remain conceptually distinct. A probability, a market
comparison and an explanation do **not** imply "place a bet." Any wagering functionality
requires separate legal, regulatory, risk and jurisdictional analysis.

## Anti-patterns

Feature stuffing until backtests look profitable · hindsight leakage · survivorship bias ·
cherry-picked wins · hidden failed predictions · treating bookmaker disagreement as proof
of edge · correlation as causation · rumors presented as confirmed · stale intelligence
presented as novel · silently changing historical predictions · confusing prediction
accuracy with betting profitability.

## Relationship to existing canonical reserves

`prospective-claim-commitment.md` (commitment ledger) · `proof-before-promotion.md` ·
`evidence-lifecycle-state-provenance-envelope.md` (source state over time) · VERITY
(source trust: official club statement vs journalist vs anonymous account) · Information
Admissibility Governor (ADMIT / DEFER / DOWNWEIGHT / REJECT / ESCALATE) · DRIFT /
`regime-evidence-engine.md` (signal decay, market adaptation, rule changes) ·
`staging/reserves-2026-07-13.md` (EventPulse/SportGPT — staging, unpromoted).

## Doctrine

**A prediction is not intelligence by itself.** Intelligence requires context, evidence,
timing, uncertainty, comparison, explanation and subsequent accountability.

## Activation

Revisit when SportGPT has sufficient access to reliable sports/event data, timestamped
market data, team/player information, EventPulse events, source provenance, model outputs
and prospective observation capacity.

RESERVED. DO NOT OVERBUILD. DO NOT CLAIM MARKET EDGE WITHOUT PROSPECTIVE EVIDENCE.
