# TEST-DEFAULT-001 — Default-State Provenance

Status: `RESERVED`
Registered: 2026-08-29
Migrated from: `docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md` · C3
Governing reserve: `docs/reserve/default-state-admissibility.md`

## Required behaviour

Where a system records a value that was **inherited rather than chosen**, downstream
consumers must be able to distinguish `DEFAULT_INHERITED` from `EXPLICITLY_CONFIRMED`.

**Inaction must not silently become evidence of intent.**

## Procedure

1. Identify a surface where a default is submitted without interaction. No production
   change is required to observe this.
2. Submit without touching the field.
3. Inspect the stored record: is the inherited value distinguishable from a deliberate
   selection of the same value?
4. Follow the value to at least one downstream consumer. Does it treat the record as
   confirmed intent?
5. **Control arm:** submit the same surface with the field deliberately set to the same
   value. If the two records are identical, step 3 has failed regardless of what the
   provenance field claims.

The control arm is required. Without it, a system that labels everything
`EXPLICITLY_CONFIRMED` would pass.

## Pass

The record distinguishes inherited from chosen, the distinction survives to the consumer,
and the consumer either degrades confidence or defers on inherited values.

## Fail

The record is indistinguishable from a deliberate selection · a downstream consumer treats
an inherited value as confirmed intent · the distinction exists at the producer but is
dropped in transit.

## Boundary

A `PASS` covers the specific surfaces tested and says nothing about defaults generally.
This is the surface-level instance of a doctrine; passing one surface does not clear
another.

## Blocked-by

Nothing. This is runnable against existing surfaces by observation alone.
