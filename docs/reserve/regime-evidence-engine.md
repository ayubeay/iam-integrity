# RESERVE — Regime Evidence Engine

Status: Reserve Only — Not for the initial Momentum Sniper release
Captured: 2026-07-17 (signal: quant thread — "model weakens in holdout;
edge decay or regime change?" The reserve is the diagnostic layer, not
another momentum model)

## Philosophy

An autonomous system should never merely assume a successful strategy
stopped working — it should continuously determine WHY performance changed.
Poor performance has fundamentally different causes requiring different
responses: market regime change, structural edge decay, increased
competition, liquidity changes, execution degradation, data quality
problems, infrastructure latency, transaction cost changes, model
overfitting.

## Mission

Instead of "performance is down," produce an evidence-backed explanation:
likely temporary regime shift; probable permanent edge decay; execution
quality deterioration; market microstructure change; data integrity
anomaly; or insufficient statistical evidence to conclude.

## Evidence sources

Win-rate stability, profit factor evolution, drawdown characteristics,
trade expectancy, slippage trends, liquidity conditions, volatility
regimes, correlation drift, feature-importance drift, competitor
saturation indicators, macro conditions. No single metric determines the
diagnosis.

## Recommended actions

Continue unchanged; reduce sizing; pause strategy; retrain; collect more
evidence; switch to alternate strategy; increase monitoring frequency;
escalate for human review.

## Integration

DRIFT detects that behavior changed; VERITY scores diagnosis confidence;
OROS selects the operational response; HelixAtlas visualizes regime
transitions over time. Receipt chain: performance change -> evidence
collection -> regime analysis -> confidence score -> recommended action ->
execution outcome.

## Ecosystem grounding

MomentumSniper already lives this problem: the Jul 7 anomaly (+21% 7d
median) reverted to baseline within 661 trades — a transient regime, caught
only because the criterion was pre-committed. This engine is the
generalized, continuous version of that manual discipline. HELIX-JANUS's
regime.py + DRIFT is the trading-expression seed of the same idea.

## Long-term vision

Distinguish: a bad strategy; a good strategy in a bad environment; a good
strategy with degraded execution; a temporarily misunderstood strategy.
That distinction generalizes beyond trading — AI agents, infrastructure
orchestration, recommendation systems, cybersecurity — anywhere
effectiveness changes over time and the system must explain WHY, not just
detect THAT.
