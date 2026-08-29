# docs/research/tests — implementation invariants

A **test** asks whether a system satisfies a required behaviour. It names the system under
test, the invariant, the procedure, and the exact conditions under which it passes or fails.

This is a different question from the one an experiment asks. See
`docs/research/experiments/INDEX.md`. The rules governing both are in
`docs/research/EVIDENCE_DISCIPLINE.md`.

## Status vocabulary

    RESERVED   the invariant is specified; no implementation exists to run against
    READY      an implementation exists and the test can be run
    RUNNING    execution in progress
    PASS       the implementation satisfied the invariant under the stated procedure
    FAIL       it did not
    BLOCKED    cannot proceed; the blocker is named in the spec

## Register

| ID | Invariant | Status |
|---|---|---|
| [TEST-CONCENTRATION-001](./TEST-CONCENTRATION-001.md) | N individually admissible facts must not compose into an inadmissible capability unregistered | `RESERVED` |
| [TEST-RESTRAINT-001](./TEST-RESTRAINT-001.md) | `PRESERVE_UNKNOWN` survives classification pressure and stays distinct from `UNKNOWN` | `RESERVED` |
| [TEST-DEFAULT-001](./TEST-DEFAULT-001.md) | An inherited default must stay distinguishable from a deliberate selection, all the way to the consumer | `RESERVED` |
| [TEST-ATOMIZATION-001](./TEST-ATOMIZATION-001.md) | A narrative is evaluated as its constituent propositions; no verdict is manufactured under insufficient evidence | `RESERVED` |
| [TEST-STALE-ARTIFACT-001](./TEST-STALE-ARTIFACT-001.md) | **BLOCKING** — a stale artifact is distinguishable from a current one, and a receipt's cited source is its actual source | `RESERVED` |
| [TEST-INCOMPLETE-001](./TEST-INCOMPLETE-001.md) | **BLOCKING** — every downstream consumer gates on `score_status: INCOMPLETE` | `RESERVED` |
| [TEST-REPLAY-001](./TEST-REPLAY-001.md) | The replay runner reproduces its own recorded baselines | `RESERVED` |

## The boundary that must not be crossed

**A `PASS` is never evidence that an external-world hypothesis is validated.**

A test establishes that an implementation behaves as required, on the inputs used, at the
commit tested. It says nothing about whether the proposition motivating the invariant is
true of the world. Promoting a `PASS` into a validated hypothesis is the specific error this
directory split exists to prevent.

The reverse also holds: an experiment's `VALIDATED` does not mean any system implements the
consequence correctly. That is what a test is for.

## Rules

**`RESERVED` is honest.** Most specs here have no implementation to run against, and saying
so is preferable to a test that passes because it exercises nothing.

**Every test names a control.** A test that cannot fail proves nothing. Where a naive
always-refuse or always-accept implementation would pass, the spec must include the arm that
catches it.

**Specification is not implementation authorization.**
