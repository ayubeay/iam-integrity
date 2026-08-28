# RESERVE — Underwater Duration & Edge-Decay Admissibility (UDEA)

Status: RESERVED — research / measurement primitive. **DO NOT alter live or shadow
strategy parameters merely because this reserve exists.**
Captured: 2026-08-28.
Primary home: Momentum Sniper / JANUS.
Supporting architecture: DRIFT, VERITY, vLOID execution-governance receipts.

**Do not promote any external trading anecdote or forum statistic into system doctrine
without independent evidence.**

## Origin

Systematic-trading research on an often-undermeasured distinction:

    DRAWDOWN DEPTH ≠ DRAWDOWN DURATION

Maximum drawdown measures how far an equity curve fell from a previous peak. Underwater
duration measures how long the system stayed below that peak before recovering. A
positive-expectancy strategy can spend substantial time below its previous high.

The operational problem: **a statistically ordinary adverse period feels like edge decay
in real time.** That perception causes premature parameter changes, abandonment, risk
escalation and overfitting — interventions that destroy otherwise valid systems.

UDEA exists to distinguish *normal strategy suffering* from *possible edge decay or
regime change*, as rigorously as the available evidence permits.

## Core principle

Do not ask only "how much are we down?" Ask "how long have we been underwater?" and,
more importantly, **"how much relevant opportunity has occurred while we were
underwater?"**

Candidate measurements (fields will evolve after research):

    underwater_trade_count · underwater_wall_clock_duration
    underwater_eligible_opportunity_count · underwater_regime_duration
    bars_since_equity_high · trades_since_equity_high
    eligible_setups_since_equity_high · executed_setups_since_equity_high
    denied_or_skipped_setups_since_equity_high
    current_drawdown_depth · maximum_drawdown_depth · recovery_duration

## Why multiple clocks matter

"50 trades underwater" is not intrinsically meaningful — it may be ten hours for a
high-frequency system, fifty days for another, or several months for a selective one.
Equally, "30 days without a new equity high" indicates nothing about degradation if
almost no historically admissible opportunity occurred in those thirty days. Momentum
Sniper is deliberately selective, so wall-clock duration alone produces false alarms.

## Counterfactual underwater duration

Reserve the stronger concept. Distinguish:

**A — strategy failure despite sufficient relevant opportunity.** 30 days underwater,
400 historically comparable admissible setups, a large number executed, performance
materially worse than the historical conditional distribution. This may indicate genuine
degradation.

**B — absence of the opportunities the strategy was designed to exploit.** 30 days
underwater, 6 genuinely admissible setups, a regime that rarely produced the strategy's
required conditions. This primarily indicates opportunity scarcity.

Therefore **NO NEW EQUITY HIGH must never automatically imply EDGE DECAY.**

## Conditional distribution principle

Historical underwater behaviour is not one unconditional distribution. Condition on JANUS
regime, market-cap regime, liquidity, volatility, opportunity frequency, setup type,
signal strength, execution environment, market breadth, risk posture, eligible-candidate
count and any other demonstrated explanatory variable.

The question becomes *"is the present underwater state anomalous relative to historically
comparable conditions?"* rather than *"is the strategy currently losing?"*

## Governed response loop

    OBSERVE → MEASURE UNDERWATER STATE → IDENTIFY CURRENT REGIME
    → MEASURE ELIGIBLE OPPORTUNITY → COMPARE WITH HISTORICAL CONDITIONAL DISTRIBUTION
    → ESTIMATE ANOMALY / CONFIDENCE → DRIFT ASSESSMENT → GOVERNED RESPONSE

Candidate responses (taxonomy subject to research):

- `PRESERVE` — no sufficient evidence of degradation
- `WATCH` — unusual behaviour emerging, evidence insufficient for intervention
- `THROTTLE` — reduce exposure while collecting additional evidence
- `SHADOW` — stop or reduce capital exposure, continue counterfactual observation
- `INVESTIGATE` — run explicit degradation / regime diagnostics
- `RETIRE` — only after sufficient evidence that the original edge no longer satisfies
  the system's admissibility requirements

## Parameter-intervention guard

**Do not change parameters merely because a strategy feels stale while its underwater
state remains within its historically expected conditional range.** Parameter
modification must require evidence beyond psychological discomfort, recent losses, lack
of new equity highs, or arbitrary calendar duration.

Every future intervention should preserve a receipt containing: trigger · observed
anomaly · historical comparison · regime context · opportunity count · confidence ·
proposed intervention · authorization · before/after configuration · subsequent outcome.
This prevents hindsight rewriting.

## Anti-complacency guard

The opposite failure must also be prevented. Historical drawdown tolerance must **never**
become justification for indefinitely preserving a genuinely degraded strategy. *"This
happened before"* is not evidence that *"this is still normal."* If current behaviour
becomes materially anomalous conditional on opportunity and regime, DRIFT should escalate
investigation.

The purpose of UDEA is neither `KEEP TRADING` nor `STOP TRADING`. It is
**improve the evidence used to decide.**

## Relationship to DRIFT

DRIFT should eventually distinguish expected variance, temporary opportunity drought,
execution degradation, liquidity change, market-regime transition, signal degradation and
structural edge decay. UDEA is one measurement family feeding DRIFT, not an independent
strategy controller. **DRIFT must not infer regime change solely from underwater
duration.**

## Relationship to Executable Capacity

`executable-capacity-thinnest-leg.md` asks *could the historical opportunity actually have
been executed at the assumed size and conditions?* UDEA asks *is the current adverse
period abnormal enough to indicate possible degradation?* Together they attack two
backtest-to-live failure classes:

1. **Fake executable capacity** — a backtest assumes liquidity or size that never existed.
2. **Premature strategy intervention** — a valid system is modified or abandoned during
   statistically ordinary adversity.

## Momentum Sniper application

Longitudinal evidence should eventually measure more than win rate, average and median
return, large winners, catastrophic losses, skip rate and total return. Also: time below
previous equity high · trades below previous equity high · eligible opportunities while
underwater · regime transitions while underwater · setup frequency during underwater
periods · recovery characteristics · depth × duration · duration × opportunity density ·
parameter stability through adverse periods.

The objective is to learn **what normal suffering looks like for this strategy** before
meaningful live capital is exposed.

## Research questions

What constitutes an equity high for the relevant accounting model? Should underwater
duration be measured by trades, bars, wall-clock time, eligible opportunities, or all?
Which measure best predicts genuine degradation? How should skipped opportunities be
incorporated? How do changing JANUS regimes alter expected underwater distributions? How
much history is required before declaring a duration abnormal? How do we avoid using
future information when defining acceptable ranges? Can change-point or regime-detection
methods improve the variance-versus-decay distinction? How is execution deterioration
separated from signal deterioration? When should abnormal duration trigger `WATCH`,
`THROTTLE`, `SHADOW`, `INVESTIGATE` or `RETIRE`? How are multiple strategies evaluated
when portfolio diversification masks one strategy's degradation? Can historical
intervention simulations quantify how much performance premature parameter changes would
have destroyed?

## Activation

Activate when Momentum Sniper / JANUS possesses enough chronologically clean shadow or
live-equivalent evidence to estimate meaningful underwater distributions without
manufacturing confidence from inadequate sample sizes.

Until then: log the necessary state · preserve raw observations · do not optimize against
underwater duration · do not change parameters to make historical drawdowns look better ·
do not treat external trading anecdotes as ground truth.

The reserve exists so that future live-capital decisions can distinguish **pain** from
**evidence of failure**.

RESERVED — DO NOT BUILD.
