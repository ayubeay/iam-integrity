# EXP-PRECURSOR-001 — Counterfactual Execution Governor, Phase Zero

Status: `PROPOSED`
Registered: 2026-08-29
Migrated from: `docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md` · C1
Reserve: `docs/reserve/counterfactual-execution-governor.md`

## Hypothesis

Observable precursor states can predict a dangerous or policy-violating agent execution
early enough to intervene, at an acceptable false-intervention rate.

This is a proposition about whether such prediction is *possible*, not about whether any
current implementation does it. That is why it is an experiment rather than a test.

## Preconditions

A controlled agent sandbox with known ground truth. **No real funds, no real-world
autonomous execution.**

## Claims, concluding separately

**Claim A — detectability.** Dangerous-action recall is materially above chance.

**Claim B — preventive utility.** Avoided expected harm exceeds intervention cost plus
false-positive cost. A predictor that achieves recall by blocking broadly satisfies A and
fails B, which is why they must not share a verdict.

**Claim C — calibration.** Predicted-probability bands match observed frequency. An 80–89%
danger band resolving near 37% rejects C regardless of A.

## Accept / reject

**Validated** requires A, B and C together. Recall alone is not the result.

**Rejected** if the predictor achieves recall only by over-blocking, or if calibration
fails while recall looks good — the configuration most likely to be mistaken for success.

## Evidence boundary

Synthetic scenarios in one sandbox. **Establishes nothing about production agents or about
human behaviour.** Ground truth is constructed, so the danger labels are the experimenter's
definition of danger, not the world's.

## Relationship to other specs

The 2026-08-27 candidates file asserted that C1 is a "close relative" of ledger candidate 1
(stale-but-valid artifact). **That relationship does not survive inspection.** C1 concerns
precursor prediction of agent behaviour; `TEST-STALE-ARTIFACT-001` concerns whether a
consumer distinguishes a stale artifact from a current one. They share only the theme of
governance catching something before harm. They are recorded here as distinct, and the
earlier assertion is superseded rather than inherited.

## Activation

vLOID needs pre-execution consequence forecasting, or enough OROS traces exist for
retrospective testing.

## Provenance

    source artifact:       docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md (C1)
    registered:            2026-08-29
    implementation commit: none — no implementation authorized
    evidence boundary:     one synthetic sandbox
    conclusion date:       pending
