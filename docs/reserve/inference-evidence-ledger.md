# RESERVE - Inference Evidence Ledger (Hypothesis Research & Evidence Ledger)

**Status:** Canonical reserve. Fills the slot marked pending in
docs/reserve/staging/reserves-2026-07-24.md line 68 and reserves-2026-07-25.md line 53.
**Urgency:** MEDIUM. Closer to home than it looks - see grading.

## Purpose
Governed, reproducible research for strategy development, so every refinement, experiment
and conclusion has an auditable history. The goal is distinguishing genuine discovery from
accidental overfitting.

Treat research as a sequence of HYPOTHESES, not a sequence of optimised parameters.

## Per research cycle
Original hypothesis, expected market behaviour, supporting reasoning, dataset, period,
instruments, entry and exit logic, parameters, filters, statistical results, failure and
success reasons, confidence, researcher notes, AI observations, human decisions, and the
evidence linking one iteration to the next.

## Research receipt per modification
Parent hypothesis, child hypothesis, what changed, why, expected impact, measured impact,
statistical significance, and the decisive question:

    did this refinement strengthen the original hypothesis,
    or merely improve historical performance?

## Refinement taxonomy
    HYPOTHESIS_STRENGTHENING     RISK_MANAGEMENT_IMPROVEMENT
    MARKET_REGIME_ADAPTATION     EXECUTION_IMPROVEMENT
    DATA_QUALITY_IMPROVEMENT     TRANSACTION_COST_ADJUSTMENT
    POSSIBLE_CURVE_FIT           UNSUPPORTED_OPTIMIZATION
    INCONCLUSIVE_EVIDENCE

## Research integrity score
Rewards research that generalises across markets and periods, survives out-of-sample
testing, stays logically consistent, preserves the original hypothesis and reproduces.

Penalises excessive parameter tuning, unsupported filters, data leakage, look-ahead bias,
survivorship bias, and optimisation without explanatory evidence.

## Integration
VERITY evidence confidence and provenance. OROS governed research workflows. DRIFT regime
changes that legitimately explain strategy evolution. Shield Router validating artifacts
before promotion. SURVIVOR immutable research receipts. HelixAtlas visualising hypothesis
evolution as a branching graph. Momentum Sniper and future trading systems as CONSUMERS of
validated research, not components of the ledger.

## Grading
Not abstract. MomentumSniper has been through multiple regime iterations, and the question
"did that change strengthen the hypothesis or just improve the backtest?" is exactly what
those iterations needed and did not have. The Liq/MC admission gate, the SL/TP rules and
the poll-interval change were each defensible in argument; none has a receipt saying which
category it belonged to.

The cheapest first step is retrospective: write the existing MomentumSniper iterations into
the taxonomy above. That would show immediately whether the strategy evolved by hypothesis
or by tuning - and it requires no new code.

## Doctrine
Move from strategy optimisation toward evidence-driven hypothesis evolution, where every
production strategy traces back through an auditable chain of research decisions supported
by explicit evidence rather than unexplained performance improvements.
