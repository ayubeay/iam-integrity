# EXP-PERSISTENCE-001 — Relationship Persistence: Drift Baseline vs Latent State

Status: `READY` · next action owned by us
Registered: 2026-08-29
Origin: external collaboration. See `docs/research/collaborations/REGISTER.md`.

## Hypothesis

A latent-state or hazard formulation predicts pair-relationship persistence better,
out of sample, than a simple recent hedge-ratio-drift baseline.

**Target is relationship survival, not P&L.** P&L stays out of this experiment entirely;
introducing it changes the question and makes the result unusable as a comparison.

## The collaborator's baseline

An external collaborator established, on their own universe: low recent hedge-ratio drift
survives better than high drift, by a few percentage points, with the direction positive in
most eras.

## Reproduction doctrine — load-bearing

**We are not required to reproduce their exact decimals or base rate.** Our universe and
windows differ. What should reproduce, if the effect is real enough in our data, is the
*shape*:

    LOW_DRIFT   → higher future survival
    HIGH_DRIFT  → lower future survival

**Reproduction can mean preserving the effect structure rather than matching someone
else's numbers.** Demanding decimal agreement across different universes would manufacture
a failure that means nothing.

If drift does not clear its own noise floor on our data, **that is a valid result** and the
collaborator has explicitly asked to be told.

## Protocol, in order

1. Preserve the collaborator's target definition and censoring logic.
2. Reproduce the drift baseline on our universe and windows.
3. **Document the result even if negative — this step is not conditional on success.**
4. Only then test the latent-state / hazard formulation.
5. Compare out of sample.

Do not reorder. Testing the richer model before the baseline is established would leave no
reference against which "better" could mean anything.

## Accept / reject

**Validated** if the latent-state formulation beats the drift baseline out of sample on the
same target under the same censoring.

**Rejected** if it does not, or if the baseline itself does not clear noise on our data —
in which case the comparison has no floor and that fact is the finding.

## Evidence boundary

Our universe, our windows, our censoring implementation. **Does not establish anything
about the collaborator's result on their data**, and does not transfer to P&L.

A later robustness question, deliberately out of scope for this run: pair-eras sharing
instruments, sectors, macro periods or structural relationships may not be independent
observations. That analysis must not be added mid-experiment.

## Provenance

    source artifact:       external collaboration protocol, agreed 2026-08
    registered:            2026-08-29
    implementation commit: none yet
    evidence boundary:     our universe and windows only
    conclusion date:       pending
