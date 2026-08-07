# RESERVE - FORGE: Forward Outcome & Regime Graph Engine

**Status:** Reserved capability shared across the stack. Not a standalone product.
**Urgency:** LOW to build. The robustness framing is usable in decisions today.

## What it is not
Not prediction. The question is not what will happen but:

    given what we know now, what futures are plausible, how confident are we in each,
    what would change them, and which actions remain safe across all of them?

## Model
    observations -> current state -> future branches -> consequences -> action

    present ---- A  48%  favourable
             |-- B  29%  deteriorating
             |-- C  15%  severe disruption
             '-- ?   8%  unresolved / unknown

The unknown branch matters most. A serious system must not force everything into futures it
already understands - the same doctrine the SURVIVOR work reached when it separated
UNRESOLVED from OBSERVED.

## Position
FORGE executes nothing.

    DRIFT   -> FORGE    regime is changing; where could it lead?
    VERITY  -> FORGE    how trustworthy are the observations? weak evidence should widen
                        uncertainty rather than produce false precision
    FORGE   -> vLOID    supplies possible consequences; vLOID still decides admissibility
    OROS                coordinates whatever was authorised
    HelixAtlas          renders branches spatially, probability mass visibly moving as
                        evidence arrives

## The capability that matters more than probability
Not which future is most likely but which action performs acceptably across the most
plausible futures.

    action   future A   future B   future C
    X          +100        -80       -100
    Y           +40        +25        -10
    Z           +20        +18        +12

A prediction system picks X because A is most likely. A resilience-oriented system prefers
Y or Z, because they survive being wrong.

## Future robustness score
Expected outcome, downside across branches, reversibility, uncertainty, cost of waiting,
information expected to arrive, catastrophic-tail exposure, recovery options.

## Applications
KONIGO Connect - satellite failure, mobile degradation, ISP recovery, data-centre problems;
choose a route robust across conditions. Earthwise - rainfall, commodity prices, logistics,
crop conditions, export demand. Commerce Sniper - supplier failure, price movement, demand,
counterparty reliability. Backpack/EHFC - competing interpretations of forced-flow
completion rather than predicting price.

And for autonomous agents generally: before an irreversible action, simulate plausible
consequences. That is likely the strongest application, and it connects directly to the
Autonomous Finance Control Plane reserve.

## Forecast receipt
Observed state, evidence sources, timestamp, each scenario with probability and supporting
evidence, the unknown probability, recommended posture, robustness score, assumptions,
invalidating conditions, model version, hash.

Six months later: what did the system believe, why, and what actually happened? That gives
forecasting something it usually lacks - accountability. Historical receipts become
calibration data.

## Grading
Building FORGE is far downstream. But the robustness question - which action survives being
wrong? - is usable in decisions today, including product ones. It is the formal version of
preferring reversible moves under uncertainty.
