# EXP-SELECTION-001 — Prospective Commitment and Selection Integrity

Status: `PROPOSED`
Registered: 2026-08-29
Migrated from: `docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md` · C5
Reserve: `docs/reserve/prospective-claim-commitment.md`
Parent mechanism: `docs/reserve/evidence-commitment-and-anchoring.md`

## Hypothesis

Committing a claim before outcome visibility, **plus** accounting for the opportunity
universe, produces a track record that resists cherry-picking — where commitment alone
does not.

The second half is the whole point. Immutable individual claims can still compose into a
misleading aggregate if the set of claims that were *not* published is unaccounted for.

## Preconditions

A system making enough prospective decisions that longitudinal claims matter, with
`eligible` / `evaluated` / `predicted` / `abstained` states recordable. Abstention must be
recorded at the moment of abstention, not reconstructed.

## Accept / reject

**Validated:** a record where published predictions cannot be reconstructed as a favourable
subset — selection ratio and abstention accounting are themselves verifiable.

**Rejected:** immutable individual claims still permit a misleading aggregate.

**This rejection is the expected result and is a finding, not a failure.** It would
establish that claim integrity alone is insufficient and that opportunity-universe
accounting is load-bearing rather than decorative. The experiment is designed so that the
likely outcome is informative.

## Evidence boundary

The specific ledger tested. **Establishes nothing about the underlying predictive skill** —
a perfectly selection-honest record of poor predictions still passes.

## Cases

### Case 001 · Purple-Ad6867 external prediction board

An external collaborator opened an independent prediction board. This provides the piece
the experiment most needs and cannot supply for itself: **a pre-outcome timestamp under
someone else's control.**

    internal prediction receipt
      → external immutable pre-outcome board record
      → real-world outcome
      → reconciliation

Protocol constraints agreed with the collaborator, and binding on this case:

- Use an **existing model with its existing decision rules.** Do not tune toward the board.
- Submit only naturally qualifying predictions. **Abstention is data, not failure.**
- Measure coverage and cadence separately from accuracy.
- Begin with a small manual sample; consider their API only after the reconciliation
  mechanics are understood.

Status: `READY` · next action owned by us. See
`docs/research/collaborations/REGISTER.md`.

## Relationship to ledger candidate 5

The 2026-08-27 candidates file paired C5 with ledger Top-5 candidate 5 (complete Epoch 2 to
a stated N with at least one real deferral). **That pairing does survive inspection.**
Epoch 2's deferral receipts are exactly the abstention accounting this experiment requires,
from a system already running. Epoch 2 remains E2 in the ledger and is not duplicated as a
spec; this experiment may consume its receipts as a second case.

## Provenance

    source artifact:       docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md (C5);
                           EXPERIMENT_LEDGER_2026-08-27.md Top-5 #5 (E2, ACTIVE)
    registered:            2026-08-29
    implementation commit: none — no implementation authorized
    evidence boundary:     one ledger per case
    conclusion date:       pending
