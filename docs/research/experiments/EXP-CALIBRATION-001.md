# EXP-CALIBRATION-001 — Agent Calibration and Epistemic Yield

Status: `PROPOSED`
Registered: 2026-08-29
Migrated from: `docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md` · C4
Reserve: `docs/reserve/agent-metacognition-calibration-layer.md`

## Two claims, concluding separately

**Claim A — calibration.** An agent's stated confidence corresponds to observed outcomes:
predicted-probability bands track observed frequency within a stated tolerance.

**Claim B — epistemic yield.** Additional reasoning or retrieval measurably reduces
decision-relevant uncertainty, and a detectable point exists beyond which it stops doing so.

A can hold while B fails, and vice versa. An agent may be well-calibrated and yet gain
nothing from thinking longer; or gain a great deal while being systematically overconfident.
Forcing one verdict would lose whichever half it did not name.

## Preconditions

Prediction receipts emitted *before* execution, postconditions observed *after*, and
sufficient volume for banding. Without pre-execution receipts this measures recollection,
not calibration.

## Accept / reject

**A rejected** if confidence is uninformative about outcomes — the common and important
case, and the one that makes a confidence field actively misleading rather than merely
useless.

**B inconclusive** if epistemic yield cannot be distinguished from noise at available
volume. Inconclusive is the honest verdict here; low volume is not evidence of no effect.

## Evidence boundary

One agent, one task class, one environment regime. **Historical competence does not transfer
across regimes** — a calibration result earned in one regime says nothing about the agent's
calibration after a regime change, which is precisely when calibration matters most.

## Activation

When an agent emits prediction receipts routinely. Consumes the same receipt substrate as
`docs/reserve/computable-accountability.md`.

## Provenance

    source artifact:       docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md (C4)
    registered:            2026-08-29
    implementation commit: none — no implementation authorized
    evidence boundary:     one agent, one task class, one regime
    conclusion date:       pending
