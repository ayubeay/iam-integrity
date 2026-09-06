# TEST-STALE-ARTIFACT-001 — Staleness Detection and Source Attribution

Status: INVARIANT A `FAIL` · INVARIANT B `PASS` · was **BLOCKING candidate** — pre-Zircon.
Gate disposition: **SATISFIED** — PRE-ZIRCON GATE RECALCULATION, 2026-09-06. The decision this candidate existed to unlock has been answered; see the dated addendum at the end of this file.
The question, procedure, controls and pass/fail conditions below are preserved UNCHANGED. Lifecycle status and execution date are recorded in the dated addendum, not on this line — following the form used by `TEST-INCOMPLETE-002`.
Registered: 2026-08-29
Migrated from: `docs/research/EXPERIMENT_LEDGER_2026-08-27.md` · Top-5 candidate 1
Related reserves: `docs/reserve/temporal-evidence-admissibility.md` ·
`docs/reserve/evidence-lifecycle-state-provenance-envelope.md`

## Why this is a test and not an experiment

The ledger recorded this as an experiment with `VALIDATED = defect confirmed`. On the
experiment/test line drawn in `docs/research/EVIDENCE_DISCIPLINE.md`, the object here is **our own
implementation** and the question is whether it satisfies a required behaviour. That is a
test. The reframing is recorded rather than silent, because the ledger's framing was
defensible and a later reader should be able to see the change was deliberate.

## Two invariants, concluding independently

**Invariant A — temporal admissibility.** The consumer can distinguish a stale artifact
from a current one.

**Invariant B — source attribution.** The receipt's cited source corresponds to the source
the data actually came from.

**A and B conclude on separate lines.** One run can pass one and fail the other, and a
combined verdict would destroy that. This is the same structure as E1, where a ledger was
internally perfect and its market evidence inadmissible.

## Machinery already available

The VERITY local/production pair, divergent by five months and both readable.
`litmus-firewall/pow_fetcher.py` as **negative control** — it runs a scorer locally, reads
a local artifact, and sets the receipt's source to a production endpoint.
HELIX-JANUS `Observation` as **positive control** — a system in the same portfolio that
already defers correctly.

Both controls are required. The negative control proves the test can detect the defect; the
positive control proves it does not fire on correct behaviour.

## Procedure

1. Run the Litmus VERITY path against the current stale local artifact.
2. Capture the emitted receipt.
3. **Invariant A:** does anything in it distinguish a months-old artifact from a current
   one — a timestamp, a freshness bound, a degraded confidence?
4. **Invariant B:** does the cited source correspond to the data actually read?
5. Run the HELIX-JANUS fetch path against an unavailable source and compare what each
   records.

## Pass / fail

**A fails** (defect confirmed) if nothing distinguishes stale from current. **A passes** if
the path independently flags staleness or degrades confidence.

**B fails** (defect confirmed) if the cited source does not match the actual source.
**B passes** if it matches, or if the receipt records both.

## Why now

Zircon inherits an evidence and receipt model. **If freshness and source attribution are
not part of admissibility, every receipt inherits the hole.** This was found by accident; it
would not have been found by design.

## Boundary

The specific paths tested, at the commit tested. Establishes nothing about receipt
architecture generally.

## Decision unlocked

Whether freshness bounds and source attribution must be first-class receipt fields before
Zircon defines a receipt schema.

---
---

# DATED ADDENDUM — 2026-09-06 · EXECUTION AND GATE RECONCILIATION

**Everything above this marker is the specification as registered 2026-08-29, preserved
unchanged.** Amending a test's question after running it is what preservation exists to
prevent.

## Result

    LIFECYCLE STATUS                        EXECUTED 2026-09-06
    INVARIANT A — temporal admissibility    FAIL   (defect confirmed)
    INVARIANT B — source attribution        PASS
    NEGATIVE CONTROL                        VALID
    POSITIVE CONTROL                        PASS
    EXECUTION RECEIPT                       TEST-STALE-ARTIFACT-001-EXECUTION-RECEIPT-2026-09-06.md

Two invariants, two independent conclusions. **No global verdict.**

Ledger vocabulary, for the immutable 2026-08-27 record: Claim A `VALIDATED` (defect
confirmed) = Invariant A `FAIL`. Claim B `REJECTED` = Invariant B `PASS`, under the
ledger's own clause *"or if it records both"*.

## Evidence boundary

The Litmus/VERITY path at `litmus-firewall 0503ccd`, one stale artifact
(`verity data/leaderboard.json`, `generatedAt 2026-03-14T01:33:34.123Z`), executed
2026-09-06 under redirected test-harness isolation with the subject byte-identical before
and after. Positive control `helixjanus 8306baa`. No network, no credentials, no
application-code modification.

**Establishes nothing about receipt architecture generally.**

    OLD TIMESTAMP PRESENT ≠ STALENESS ADJUDICATED
    A TIMESTAMP REQUIRES AN AGE / FRESHNESS SEMANTIC
      TO ESTABLISH TEMPORAL ADMISSIBILITY

## The requirement the FAIL exposed is already owned — by composition

    docs/reserve/temporal-evidence-admissibility.md
    docs/reserve/evidence-lifecycle-state-provenance-envelope.md   (ELSPE, its parent)

The bridge is written into the documents themselves: temporal-evidence-admissibility
declares itself *"Child of ELSPE … Neither restates the other"* and partitions ownership —
ELSPE holds lifecycle state, provenance envelope and temporal claim semantics; the child
holds decay rate, persistence and the point of inadmissibility.

Load-bearing specifics, not vocabulary overlap: the child's **five separated times**
(event / observation / knowledge / decision / execution), its lifecycle
`DISCOVERED → CONFIRMED → ACTIVE → WEAKENING → DECAYING → INVALID`, and its **receipt
field list**, which already names `signal_age`, `decay_score` and `admissibility_decision`.
ELSPE supplies `observed_at` / `published_at` with the rule that the distinction must
survive normalization, and the section **"State is not admissibility."**

A receipt carrying those fields could not exhibit the observed defect.

    DEFECT DISCOVERED           ≠ NEW ARCHITECTURE REQUIRED
    REQUIREMENT ALREADY OWNED   ≠ REQUIREMENT IMPLEMENTED

**Both reserves are RESERVED — DO NOT BUILD.** The requirement is owned as specification
and is implemented nowhere. Nothing here asserts otherwise, and nothing new is created.

## What this does not establish

Repairing `litmus-firewall` is an independent downstream job. No document in this
repository states a dependency from Zircon to Litmus.

    LITMUS FAILS THE REQUIREMENT ≠ ZIRCON MUST WAIT FOR LITMUS TO BE REPAIRED

Also unestablished: whether an unredirected production invocation behaves identically;
whether any deployed instance exhibits this; anything about the IAM branch.

## Gate disposition

**SATISFIED.** The candidate's stated purpose was to prevent Zircon inheriting an
unexamined evidence model. The model was examined, the hole was found in a real path, and
the requirement that closes it already exists in doctrine.

    A GATE THAT PRODUCED ITS INFORMATION IS SATISFIED,
      NOT PENDING UNTIL THE IMPLEMENTATION IT EXPOSED IS REPAIRED

Execution receipt: `TEST-STALE-ARTIFACT-001-EXECUTION-RECEIPT-2026-09-06.md`.
Reconciliation: `docs/research/PRE-ZIRCON-GATE-RECONCILIATION-2026-09-06.md`.
