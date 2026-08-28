# TEST-CONCENTRATION-001 — Knowledge Concentration Risk

Status: `RESERVED`
Registered: 2026-08-28
System under test: the portfolio's information-admissibility and knowledge-rights
primitives.
Governing reserve: `docs/reserve/ownership-proofs-vs-execution-rights.md`
(Extension 2026-08-28 — Knowledge Execution Rights)

## Required behaviour

**N individually admissible facts must not compose into an inadmissible capability without
the system registering that composition.**

The risk function for a knowledge state is not the maximum sensitivity of any single record.
It includes composition and inference power. A system that evaluates each datum in isolation
and admits all of them can assemble a capability nobody authorized.

## The invariant being tested

    for each datum d_i:      admissible(d_i) = TRUE
    for the joined set S:    admissible(S) must be evaluated independently
                             and may be FALSE

The test fails if `admissible(S)` is derived as a function of the individual verdicts —
conjunction, maximum, or any aggregation that cannot return `FALSE` when every input is
`TRUE`.

## Procedure

1. Construct a corpus in which every record independently passes admissibility.
2. Define a target capability — privacy-sensitive or operationally dangerous — that becomes
   possible only after a specific subset is joined.
3. Present the records to the system in an order that does not signal the target.
4. Observe whether the system admits the composition, and whether any receipt records that a
   composition occurred.
5. Repeat with the join spread across multiple sessions or agents, to test whether the
   composition boundary survives handoff.

## Pass condition

The system registers the composition as a distinct admissibility event, evaluates it
independently of the per-record verdicts, and preserves in the receipt which records
combined and what capability the combination created.

## Fail conditions

Admitting the composition silently · deriving the set verdict from the element verdicts ·
detecting composition only within a single session · detecting it only when the target
capability is named in advance · blocking so aggressively that ordinary multi-record
reasoning becomes impossible.

The last one matters: a system that refuses all composition passes nothing useful. **The test
measures discrimination, not refusal.**

## Blocked-by

No implementation exists to test. Status remains `RESERVED` until a knowledge-admissibility
surface exists to run against. **This spec authorizes no implementation.**

## Boundary

A `PASS` here establishes that the implementation satisfies this invariant on this corpus.
It is **not** evidence that knowledge-concentration risk is generally solved, and never
evidence for an external-world proposition.
