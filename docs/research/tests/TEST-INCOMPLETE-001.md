# TEST-INCOMPLETE-001 — Do Consumers Honour `INCOMPLETE`?

Status: `RESERVED` · **BLOCKING candidate** — pre-Zircon
Registered: 2026-08-29
Migrated from: `docs/research/EXPERIMENT_LEDGER_2026-08-27.md` · Top-5 candidate 2

## Required behaviour

The validated-five design emits `score_status: INCOMPLETE` when a required signal fails to
resolve. **Every downstream consumer must gate on it** — degrade, defer, or refuse — rather
than consuming the score silently.

## Why now

E3 already established that missing data can *inflate* a score: any denominator policy is a
different way of dividing by a hole. If `INCOMPLETE` is not enforced downstream, the
guarantee exists only at the producer and is **unenforceable** — which is the same as not
existing, from the consumer's side.

Dependency and finality semantics are inherited directly by Zircon.

## Machinery reused

`survivor-oracle` scorer · `survivor-shield-sdk` · `agentguard` · `poi-engine` consumer
path. All present. **None modified.**

## Procedure

1. Static trace of `score_status` from producer to every consumer. Enumerate consumers
   first — a consumer omitted from the trace is a consumer that was never tested.
2. One probe with a deliberately unresolvable signal.
3. Observe each consumer's behaviour on the `INCOMPLETE` record.
4. **Control arm:** the same probe with a fully resolved signal. Consumers must proceed
   normally — otherwise a system that refuses everything would pass.

## Pass

Every enumerated consumer degrades, defers, or refuses on `INCOMPLETE`, and proceeds
normally on the control.

## Fail

**Any consumer proceeds on `INCOMPLETE` without degrading or deferring.** One is enough.
A guarantee honoured by three of four consumers is not honoured.

## Boundary

The consumers enumerated at the commit tested. A new consumer added later is untested by
construction — which is itself an argument for enforcing at the boundary rather than
per consumer.

## Decision unlocked

Whether admissibility must be enforced at the boundary or can be delegated to producers.
