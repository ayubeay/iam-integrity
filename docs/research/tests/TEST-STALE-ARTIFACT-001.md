# TEST-STALE-ARTIFACT-001 — Staleness Detection and Source Attribution

Status: `RESERVED` · **BLOCKING candidate** — pre-Zircon
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
