# RESERVE — Prospective Claim Commitment (PCC)

Status: RESERVED — architecture primitive. NOT an active build.
Captured: 2026-08-27.
Child of `evidence-commitment-and-anchoring.md`, which owns the anchoring mechanism. PCC
adds the experimental doctrine that mechanism does not itself supply: **committing a
hypothesis before outcome visibility**, plus selection integrity.

## Core primitive

Before an uncertain future state resolves, an actor commits to:

    CLAIM → ACTOR / MODEL IDENTITY → TIMESTAMP → EVIDENCE AVAILABLE → CONTEXT
    → CONDITIONS → CONFIDENCE → VERSION → COMMITMENT → FUTURE RESOLUTION
    → OBJECTIVE SETTLEMENT → SCORE → PERMANENT RECEIPT

    "I believed X at time T, using evidence E, under model M, with confidence C."
    Later: "Reality resolved as Y."

The original claim is not rewritten. The mechanism prevents hindsight from rewriting
claimed competence.

## Applications

Momentum Sniper BUY/SKIP decisions · trading and model benchmarking · agent forecasts ·
risk assessments · VERITY judgments · research hypotheses · operational forecasts ·
human-vs-agent benchmarking · SportGPT forecasts (see below).

## No retroactive mutation

Once committed, do not silently change probability, confidence, model version, rationale;
insert later evidence; delete losing forecasts; reclassify predictions; rewrite
timestamps; or change settlement criteria. Corrections occur through **superseding
receipts**. Never erase historical state.

## Selection integrity — the critical extension

Recording submitted predictions alone is insufficient. A system can maintain a perfectly
immutable record while cherry-picking *what enters* the record.

Preserve the **opportunity universe**:

    ELIGIBLE → EVALUATED → PREDICTED → ABSTAINED → DEFERRED
    → INVALIDATED/CANCELLED → RESOLVED

If a system examines 10,000 candidates and publishes 200 that perform well, the public
record does not establish predictive competence without knowing how those 200 were
selected.

    IMMUTABILITY ≠ COMPLETENESS
    CLAIM INTEGRITY ≠ SELECTION INTEGRITY

Claim integrity asks whether the recorded prediction was altered. Selection integrity asks
how this prediction came to be part of the displayed record. **Both matter.**

## Abstention receipts

An intelligent system may say "I don't know." Abstention must not become an invisible
mechanism for retrospective selection, so abstention and non-action are accounted for
explicitly where appropriate.

## Attack surface

Hash-locking alone does not establish trustworthy competence. Multiple identities; hidden
models; submitting many variants and promoting the winner; selective participation and
disclosure; prediction spam; correlated flooding; strategic abstention; changing
confidence semantics or settlement rules; ambiguous outcome definitions; illiquid-market
exploitation; benchmark shopping; version proliferation; identity reset after poor
performance.

## Identity continuity and model lineage

An actor should not accumulate losses, discard identity, and present a clean record where
policy recognizes continuity. Model versions v1 → v2 → v3 should not silently merge or
disappear: what changed, when, why, whether the new model was chosen after observing prior
outcomes, and how each version performed prospectively.

## Resolution integrity

A commitment is only as trustworthy as its resolver. Resolvers should be objective,
timestamped, traceable and reproducible where possible. Disputed outcomes must not be
silently forced into binary settlement: RESOLVED · DISPUTED · VOID · CANCELLED ·
UNRESOLVABLE · SUPERSEDED.

## Competence receipts

Over time, individual prospective receipts become longitudinal evidence across
calibration, coverage, abstention quality, confidence reliability, domain and regime
performance, consistency and market-relative performance. **Do not collapse this into a
single universal reputation number by default.**

## Application — SportGPT Prediction Commitment Ledger

Applies to `sportgpt-intelligence-layer.md`: event identified → admissible evidence
snapshot → market snapshot → model version → forecast → confidence → divergence →
rationale → commitment → event occurs → outcome → settlement → calibration → immutable
receipt. **This does not promote the separately staged EventPulse/SportGPT material**
(`staging/reserves-2026-07-13.md`), which retains staging status.

## Doctrines

    PROSPECTIVE EVIDENCE > RETROSPECTIVE STORYTELLING
    IMMUTABILITY ≠ COMPLETENESS
    CLAIM INTEGRITY ≠ SELECTION INTEGRITY
    ABSTENTION IS A DECISION AND MAY REQUIRE ACCOUNTING
    NEGATIVE RESULTS ARE EVIDENCE
    A TRUSTWORTHY TRACK RECORD PRESERVES WHAT WAS BELIEVED
    BEFORE THE ANSWER WAS KNOWN

## Relationship to existing canonical reserves

`evidence-commitment-and-anchoring.md` (parent mechanism) · `proof-before-promotion.md` ·
`extraordinary-claim-evidence-tree.md` (falsification receipts) · VERITY (claim
provenance, identity continuity, resolver trustworthiness, record completeness) ·
Information Admissibility Governor (no retrospective intelligence leakage) · DRIFT
(distinguishing genuine degradation from retrospective rewriting) ·
`computable-accountability.md` · `agent-metacognition-calibration-layer.md` (prediction
receipts and calibration).

## Activation

Revisit when SportGPT, Momentum Sniper or another system makes enough prospective
decisions that longitudinal performance claims matter. Investigate commitment mechanism,
identity model, model lineage, opportunity-universe accounting, resolver design, receipt
structure, selection-integrity measurement, privacy, storage, cost and legal implications.

Experiment definition: `docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md`.

RESERVED. DO NOT BUILD.
