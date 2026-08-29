# docs/research/experiments — falsifiable propositions

An **experiment** asks whether a proposition about the world is true. It carries a
hypothesis, a pre-registered acceptance and rejection condition, and an evidence boundary
stating the scope within which any conclusion holds.

This is not the same question a test asks. See `docs/research/tests/INDEX.md`.
The rules governing both are in `docs/research/EVIDENCE_DISCIPLINE.md`.

## Status vocabulary

    PROPOSED       registered, not yet run
    ACTIVE         running
    VALIDATED      the hypothesis was supported within the stated evidence boundary
    REJECTED       the hypothesis was not supported — a result, not a failure
    INCONCLUSIVE   the run could not discriminate between hypotheses
    BLOCKED        cannot proceed; the blocker is named in the spec

**Status applies at claim level, not only at experiment level.** A single run may validate
one claim while rejecting another. Forcing one global verdict destroys information.

## Register

| ID | Proposition | Status |
|---|---|---|
| [EXP-VALUE-PROVENANCE-001](./EXP-VALUE-PROVENANCE-001.md) | Value provenance survives transformation through computation; local admissibility does not imply collective admissibility | `PROPOSED` |
| [EXP-COMPUTE-PLACEMENT-001](./EXP-COMPUTE-PLACEMENT-001.md) | Execution placement across heterogeneous/mobile compute is a governed decision that cost-performance scheduling cannot represent | `PROPOSED` |
| [EXP-GENEALOGY-001](./EXP-GENEALOGY-001.md) | Data integrity is not epistemic integrity — genealogy and retrieval structure alone can distort agent belief | `PROPOSED` |
| [EXP-MEMORY-001](./EXP-MEMORY-001.md) | An agent can be compromised upstream of execution with weights, credentials and executor untouched | `PROPOSED` |
| [EXP-PRECURSOR-001](./EXP-PRECURSOR-001.md) | Precursor states can predict dangerous agent execution early enough to intervene, at acceptable false-intervention cost | `PROPOSED` |
| [EXP-FAIR-COMPUTE-002](./EXP-FAIR-COMPUTE-002.md) | The browser/native transition region generalizes across architectures and survives participant economics | `PROPOSED` · 2B `BLOCKED` |
| [EXP-CALIBRATION-001](./EXP-CALIBRATION-001.md) | Agent confidence tracks outcomes; additional reasoning has a detectable yield ceiling | `PROPOSED` |
| [EXP-SELECTION-001](./EXP-SELECTION-001.md) | Pre-outcome commitment plus opportunity-universe accounting resists cherry-picking where commitment alone does not | `PROPOSED` |
| [EXP-PROVIDER-AGREEMENT-001](./EXP-PROVIDER-AGREEMENT-001.md) | Two providers serving the same feed give the same answer | `PROPOSED` |
| [EXP-PERSISTENCE-001](./EXP-PERSISTENCE-001.md) | A latent-state formulation beats a drift baseline at predicting relationship persistence | `READY` |
| [EXP-NARRATIVE-ROTATION-001](./EXP-NARRATIVE-ROTATION-001.md) | Capital rotates from expanded leaders toward under-expanded constituents detectably before price expresses it | `BLOCKED` |

## Rules

**Pre-registration.** The hypothesis and its acceptance/rejection conditions are written
before observation, so an unfavourable result can be accepted without the criterion being
renegotiated afterward.

**Rejection is preferred to inflation.** An experiment whose acceptance criterion is *"the
existing reserves already handle this, so create nothing"* has been designed correctly.

**One mechanism, one spec.** Where several cases exercise the same mechanism, the spec is
written once and each case references it. Two specs for one question is how the same
experiment gets run twice under two names.

**Registration is not build authorization.** Nothing in this directory authorizes
implementation, deployment, procurement or capital.
