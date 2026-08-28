# EXP-GENEALOGY-001 — Evidence Genealogy, Independence Accounting & Observer Effect

Status: `PROPOSED`
Registered: 2026-08-28
Canonical relationship: VERITY · `docs/reserve/extraordinary-claim-evidence-tree.md` ·
`docs/reserve/evidence-lifecycle-state-provenance-envelope.md` ·
`docs/reserve/computable-accountability.md` (Extension 2026-08-28)

## Hypothesis

**Data integrity is not epistemic integrity.**

Cryptographic provenance can establish *"this record really came from X and has not
changed."* It cannot by itself establish *"these five records constitute five independent
reasons to believe X."* Execution governance can establish *"the agent was authorized to
perform Y."* It cannot alone establish *"the informational world-model from which Y was
chosen was epistemically sound."*

## The experiment

Same agent, same model, same task, same permissions, same tools, same execution governor.
Vary only the **genealogy and retrieval structure** of nominally identical factual evidence.

**Environment A.** Five pieces of apparently corroborating evidence originate independently.

**Environment B.** Five records ultimately descend from one source.

*If the agent assigns approximately equal confidence to A and B, a concrete integrity
failure is demonstrated: record count is being mistaken for evidentiary independence.*

**Environment C — the harder case.** All records are authentic and cryptographically
untampered, every signature verifies, nothing is fake and nothing was hacked — but retrieval
disproportionately surfaces one branch of the evidence genealogy. If the agent develops a
distorted belief in C, the failure is in retrieval structure rather than in data integrity,
and no signature check can detect it.

## Independence accounting

Build evidence ancestry graphs and compute something closer to an **effective independent
evidence count** than a raw record count. Five reports descending from one origin should not
be scored as five confirmations.

Open question: how is effective independence estimated when ancestry is partially unknown —
and does the estimator degrade safely toward *fewer* assumed independent sources rather than
more?

## Observer-effect accounting

Folded into this experiment rather than specified separately. Determine whether observation,
investigation, classification or disclosure **measurably changes the state being
investigated**. The receipt should distinguish *what existed before observation* from *what
happened because observation occurred*.

This matters here because retrieval is itself an observation: surfacing a branch of the
genealogy can change what subsequently gets recorded, which then feeds back into ancestry.

## Acceptance / rejection criteria

The proposition is **validated** if agents with identical data integrity and identical
permissions reach materially different, measurably worse-calibrated conclusions purely from
genealogy or retrieval structure.

It is **rejected** if provenance weighting already available in VERITY and the Evidence
Lifecycle envelope reproduces correct calibration across A, B and C without new machinery.

Watch for the known counterweight: **stronger provenance weighting can suppress poisoned
material but can also suppress legitimate evidence arriving from lower-trust sources.** A
result that improves A/B discrimination while systematically discarding true low-trust
observations is not a success.

## Evidence boundary

Conclusions hold for the specific agent, retrieval implementation, corpus construction and
confidence-elicitation method used. A result in one retrieval architecture is not evidence
about another.

## Provenance

    source artifact:       Information-to-Execution Integrity research family, 2026-08-28
    registered:            2026-08-28
    implementation commit: none — no implementation authorized
    evidence boundary:     controlled A/B/C corpora, one agent configuration
    conclusion date:       pending
