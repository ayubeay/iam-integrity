# Experiment Candidates — 2026-08-27

**Status:** PROPOSED. None of these has been run.
**Purpose:** falsifiable questions extracted from the 2026-08-27 reserve batch.

These are **pointers, not specifications**. Each references a canonical reserve that holds
the architecture and intention; this file holds only the empirical question, acceptance
and rejection conditions, evidence boundary, status and activation condition. Reserve
prose is deliberately not duplicated — that is how the two records drift apart.

**Promotion path.** A candidate moves into `EXPERIMENT_LEDGER_<date>.md` only after it has
actually been run, carrying its result and evidence boundary. Inclusion here is not
implementation authorization.

---

## C1 · Counterfactual Execution Governor — Phase Zero sandbox

**Reserve:** `../reserve/counterfactual-execution-governor.md`
**Question:** can observable precursor states predict a dangerous or policy-violating agent
execution early enough to intervene, at an acceptable false-intervention rate?
**Preconditions:** a controlled agent sandbox with known ground truth; no real funds, no
real-world autonomous execution.
**Accept:** dangerous-action recall materially above chance **with** preventive utility
positive — avoided expected harm exceeding intervention and false-positive cost — and
calibration where predicted-probability bands match observed frequency.
**Reject:** a predictor achieving recall only by blocking broadly; or calibration where an
80–89% danger band resolves near 37%.
**Evidence boundary:** synthetic scenarios in one sandbox; establishes nothing about
production agents or human behaviour.
**Status:** PROPOSED · **Activation:** vLOID needs pre-execution consequence forecasting,
or enough OROS traces exist for retrospective testing.

---

## C2 · Fair Useful Compute — Phases 2B / 2C / 3

**Reserve:** `../browser-fair-compute-reserve.md` (Phases 1 and 2A banked as ledger E8/E9)
**Question:** does the hardware-advantage transition region observed on Apple Silicon
appear on materially different architectures, and does it survive participant economics?
**Preconditions:** 2B requires non-Apple accelerator access — **blocked on capital**. 2C
and 3 are runnable on existing hardware.
**Accept:** a comparable transition region reproduces across architectures (2B); a
parameter surface maps rather than a single point (2C); reward concentration under
simulated heterogeneous participants stays within a stated bound (3).
**Reject:** the transition is machine-specific; or one operator converts capital into
identities/hardware and recovers a disproportionate share regardless of workload design.
**Evidence boundary:** simulation and benchmarking only; no live network, no token.
**Status:** PROPOSED (2B BLOCKED on capital) · **Activation:** per the reserve's list.

---

## C3 · Default-State Admissibility — adversarial default/inaction cases

**Reserve:** `../reserve/default-state-admissibility.md`
**Question:** where a system records a value that was inherited rather than chosen, can
downstream consumers distinguish DEFAULT_INHERITED from EXPLICITLY_CONFIRMED — or does
inaction silently become evidence of intent?
**Preconditions:** at least one existing surface where a default is submitted without
interaction; no production change required to observe.
**Accept (defect confirmed):** the record is indistinguishable from a deliberate selection,
or a downstream consumer treats it as confirmed intent.
**Reject:** provenance already distinguishes them, or the consumer degrades confidence.
**Evidence boundary:** the specific surfaces tested; says nothing about defaults generally.
**Status:** PROPOSED · **Activation:** when a consequential surface carries a default whose
misreading would matter.

---

## C4 · AMCL — calibration and epistemic yield

**Reserve:** `../reserve/agent-metacognition-calibration-layer.md`
**Question:** does an agent's stated confidence correspond to observed outcomes, and does
additional reasoning measurably reduce decision-relevant uncertainty?
**Preconditions:** prediction receipts emitted before execution and postconditions observed
after; sufficient volume for banding.
**Accept:** predicted-probability bands track observed frequency within a stated tolerance;
and a detectable point exists beyond which further retrieval or reasoning stops reducing
uncertainty.
**Reject:** confidence is uninformative about outcomes; or epistemic yield cannot be
distinguished from noise at available volume.
**Evidence boundary:** one agent, one task class, one environment regime. Historical
competence does not transfer across regimes.
**Status:** PROPOSED · **Activation:** when an agent emits prediction receipts routinely.

---

## C5 · Prospective Claim Commitment — selection integrity

**Reserve:** `../reserve/prospective-claim-commitment.md`
(parent mechanism: `../reserve/evidence-commitment-and-anchoring.md`)
**Question:** does committing a claim before outcome visibility, **plus** accounting for
the opportunity universe, produce a track record that resists cherry-picking — where
commitment alone does not?
**Preconditions:** a system making enough prospective decisions that longitudinal claims
matter; eligible/evaluated/predicted/abstained states recordable.
**Accept:** a record where published predictions cannot be reconstructed as a favourable
subset — i.e. selection ratio and abstention accounting are themselves verifiable.
**Reject:** immutable individual claims still permit a misleading aggregate, confirming
that claim integrity alone is insufficient. *This is the expected result and is a finding,
not a failure.*
**Evidence boundary:** the specific ledger tested; establishes nothing about the underlying
predictive skill.
**Status:** PROPOSED · **Activation:** SportGPT or Momentum Sniper reaching sustained
prospective decision volume.

---

## C6 · Extraordinary Claim Evidence Tree — UNKNOWN preservation

**Reserve:** `../reserve/extraordinary-claim-evidence-tree.md`
(parent doctrine: `../reserve/proof-before-promotion.md`)
**Question:** when evidence supports neither TRUE nor FALSE, does the system terminate at
UNKNOWN — or does it manufacture a verdict?
**Preconditions:** one narrowly falsifiable claim, competing hypotheses defined, evidence
requirements and a falsification receipt fixed **before** investigation.
**Accept:** the system preserves UNKNOWN where warranted, atomizes the claim rather than
evaluating a narrative as one proposition, and distinguishes source count from independent
evidence count.
**Reject:** it collapses to TRUE/FALSE under insufficient evidence, propagates evidence for
one atomized claim onto others, or treats a repost lineage as independent corroboration.
**Evidence boundary:** one claim, one evidence corpus; an adversarial test of the
Information Admissibility Governor, not a finding about the claim's subject matter.
**Status:** PROPOSED · **Activation:** Information Admissibility Governor testing, VERITY
evaluation, or Signal Drift scenario design.

---

## Relationship to the Top 5 pre-Zircon candidates

These six are additions to, not replacements for, the ranked Top 5 in
`EXPERIMENT_LEDGER_2026-08-27.md`. C1 and C5 are close relatives of ledger candidates 1
and 5 respectively and should be reconciled before either is run, so the same question is
not executed twice under two names.

**Zircon remains reserve-only.**

---

## MIGRATION — 2026-08-29

All six candidates have been migrated to individual specifications. **The text above is
preserved unchanged as the dated record of how these questions were first framed.** Where
a later classification differs, the spec states why rather than silently superseding.

| Candidate | Migrated to | Classification |
|---|---|---|
| C1 · Counterfactual Execution Governor | [`experiments/EXP-PRECURSOR-001.md`](./experiments/EXP-PRECURSOR-001.md) | experiment |
| C2 · Fair Useful Compute 2B/2C/3 | [`experiments/EXP-FAIR-COMPUTE-002.md`](./experiments/EXP-FAIR-COMPUTE-002.md) | experiment · 2B `BLOCKED` |
| C3 · Default-State Admissibility | [`tests/TEST-DEFAULT-001.md`](./tests/TEST-DEFAULT-001.md) | **test** |
| C4 · AMCL calibration | [`experiments/EXP-CALIBRATION-001.md`](./experiments/EXP-CALIBRATION-001.md) | experiment |
| C5 · Prospective Claim Commitment | [`experiments/EXP-SELECTION-001.md`](./experiments/EXP-SELECTION-001.md) | experiment |
| C6 · Extraordinary Claim Evidence Tree | [`tests/TEST-ATOMIZATION-001.md`](./tests/TEST-ATOMIZATION-001.md) | **test**, scoped — see below |

### Reconciliations performed during migration

**C6 split three ways.** Deliberate declining to resolve was already owned by
`tests/TEST-RESTRAINT-001.md`; source count versus independent evidence count by
`experiments/EXP-GENEALOGY-001.md`. `TEST-ATOMIZATION-001` owns only what remained — claim
atomization and verdict restraint under insufficient evidence — and states both boundaries
rather than restating either mechanism.

**C5 ↔ ledger candidate 5 confirmed.** Epoch 2's deferral receipts are exactly the
abstention accounting `EXP-SELECTION-001` requires. Epoch 2 remains E2 in the ledger and is
**not** duplicated as a spec.

**C1 ↔ ledger candidate 1 not confirmed.** This file asserted the two were close relatives.
On inspection they share only the theme of governance catching something before harm: C1
concerns precursor prediction of agent behaviour, ledger candidate 1 concerns whether a
consumer distinguishes a stale artifact from a current one. They are recorded as distinct.
The earlier assertion is superseded, not inherited.

### Top-5 pre-Zircon candidates

Also migrated from `EXPERIMENT_LEDGER_2026-08-27.md`:

| Ledger candidate | Migrated to | Classification |
|---|---|---|
| 1 · Stale-but-valid artifact | [`tests/TEST-STALE-ARTIFACT-001.md`](./tests/TEST-STALE-ARTIFACT-001.md) | **test**, two independent invariants |
| 2 · Consumers honour `INCOMPLETE` | [`tests/TEST-INCOMPLETE-001.md`](./tests/TEST-INCOMPLETE-001.md) | **test** |
| 3 · Replay harness reproduces baselines | [`tests/TEST-REPLAY-001.md`](./tests/TEST-REPLAY-001.md) | **test** |
| 4 · Two providers agree | [`experiments/EXP-PROVIDER-AGREEMENT-001.md`](./experiments/EXP-PROVIDER-AGREEMENT-001.md) | experiment |
| 5 · Complete Epoch 2 with a real deferral | **no spec created** | already E2, `ACTIVE` in the ledger |

The ledger itself is **not** modified. It is a dated snapshot; adding forward references to
later work would break that property. The specs cite the ledger, not the reverse.
