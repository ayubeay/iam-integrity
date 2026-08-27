# docs/research — what measurement has taught us

This directory holds the portfolio's **experimental knowledge**: hypotheses that were tested, results that were validated or rejected, and the boundaries within which each conclusion holds.

It is one of three distinct records. Confusing them is how a portfolio loses track of what it knows.

| Record | Question it answers | Location |
|---|---|---|
| **Implementation status** | What exists and runs? | `IMPLEMENTATION_STATUS.md` |
| **Reserves** | What may exist — decisions preserved for possible future work? | `RESERVED.md`, `docs/reserve/` |
| **Research** | What has measurement actually taught us? | `docs/research/` (this directory) |

A system can appear in all three at once and mean something different in each. Implementation status records that HELIX-JANUS runs. A reserve records that JANUS-ORA might one day exist. This directory records that HELIX-JANUS's Epoch 1 ledger was internally perfect and its market evidence inadmissible — a fact that belongs to neither of the others.

## Canonical snapshot

**[`EXPERIMENT_LEDGER_2026-08-27.md`](./EXPERIMENT_LEDGER_2026-08-27.md)** — current canonical research snapshot.

Ledgers are dated. New experiment epochs get new dated files rather than being appended into one growing document, so that no single file becomes unreadable the way an undated accumulation would.

## Conventions

**Claim-level status.** `VALIDATED` / `REJECTED` / `INCONCLUSIVE` / `ACTIVE` / `PROPOSED` / `BLOCKED` apply to individual claims, not necessarily to whole experiments. A single run can validate one claim while rejecting another; forcing one global verdict destroys information.

**Negative results are first-class.** A rejected hypothesis is retained whenever it narrows the design space or overturns an architectural assumption. Rejection is a result, not a failure to produce one.

**Every conclusion carries its evidence boundary.** The provenance footer records `source artifact → experiment date → implementation commit → evidence boundary → conclusion date`. The boundary states the scope within which the conclusion holds — one machine, one workload, one population, one provider. A conclusion recorded without its boundary is not preserved, only remembered, and will eventually be applied where it does not hold.

**Pre-registration.** Hypotheses and acceptance conditions are written down before observation. This is not ceremony: it is the reason an unfavourable result can be accepted without the criterion being renegotiated after the fact.

## Build boundary

Inclusion in this directory is **not** implementation authorization. A documented experiment, a promising result, and a ranked candidate are all still research. Build activation is a separate decision made elsewhere.

**Zircon remains reserve-only.**
