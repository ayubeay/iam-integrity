# docs/research/experiments — falsifiable propositions

An **experiment** asks whether a proposition about the world is true. It carries a
hypothesis, a pre-registered acceptance and rejection condition, and an evidence boundary
stating the scope within which any conclusion holds.

This is not the same question a test asks. See `docs/research/tests/INDEX.md`.

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

## Rules

**Pre-registration.** The hypothesis and its acceptance/rejection conditions are written
before observation, so an unfavourable result can be accepted without the criterion being
renegotiated afterward.

**Rejection is preferred to inflation.** An experiment whose acceptance criterion is *"the
existing reserves already handle this, so create nothing"* has been designed correctly.

**Registration is not build authorization.** Nothing in this directory authorizes
implementation, deployment, procurement or capital.
