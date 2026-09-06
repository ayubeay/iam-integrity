# Pre-Zircon Gate Reconciliation — 2026-09-06

A dated current-status record. It does **not** amend
`EXPERIMENT_LEDGER_2026-08-27.md`, which stays as written. The ledger records what was
believed on 2026-08-27; this records what the evidence establishes on 2026-09-06.

    HISTORICAL STATEMENTS REMAIN HISTORICAL.
    WAS BLOCKING CANDIDATE → CURRENT DISPOSITION

**This is not Zircon activation.** Nothing here authorizes a build, an organization, a
schema implementation, or any execution authority.

---

## Gate-disposition vocabulary

These tokens describe a **candidate's role in the pre-Zircon gate**. They are defined here
because this record uses them and nothing else in the repository does.

    BLOCKING       work cannot proceed past the gate until this concludes
    NON_BLOCKING   informative; the gate does not wait on it
    SATISFIED      the decision this candidate existed to unlock has been answered
    DISSOLVED      the candidate's gating rationale no longer holds; the underlying
                   test or experiment is unaffected and may still be run
    SUPERSEDED     replaced by another candidate; preserved as evidence
    DEFERRED       still open, but nothing currently waits on it
    UNRELATED      carries no dependency to the gate
    UNESTABLISHED  the evidence does not settle the disposition either way

**A gate disposition is not a test status.** The test register in
`tests/INDEX.md` defines its own vocabulary — `RESERVED · READY · RUNNING · PASS · FAIL ·
BLOCKED · SUPERSEDED · INCONCLUSIVE` — for whether an implementation satisfies an
invariant. The two answer different questions and must not be merged.

    A CANDIDATE'S GATING ROLE ≠ AN IMPLEMENTATION'S INVARIANT STATUS

A `SATISFIED` gate does not mean its test passed; `TEST-STALE-ARTIFACT-001` is `SATISFIED`
as a gate while carrying `INVARIANT A FAIL`. A `DISSOLVED` gate implies no result at all.

---

## Top-5 current dispositions

| # | Candidate | 2026-08-27 role | Current disposition |
|---|---|---|---|
| 1 | Stale-but-valid artifact | **BLOCKING candidate** | **SATISFIED as a gate** · Invariant A `FAIL`, Invariant B `PASS`; the exposed requirement is already owned by `temporal-evidence-admissibility` + ELSPE (owned as specification, **implemented nowhere**) |
| 2 | Do consumers honour `INCOMPLETE` | **BLOCKING candidate** | `TEST-INCOMPLETE-001` **SUPERSEDED** (premise rejected, scope mismatch) · `TEST-INCOMPLETE-002` A `PASS` · B `INCONCLUSIVE`, a distinct tested surface of `default-state-admissibility`, **not a new doctrine** · the enforcement-locus question is **unresolved / deferred** |
| 3 | Replay harness | *cheapest; gates the cost of the rest* | **Pre-Zircon gating purpose DISSOLVED** · the test is unrun and no result is implied |
| 4 | Provider agreement | *informative, non-blocking* | **PROPOSED · NON-BLOCKING** — unchanged, not executed |
| 5 | Epoch 2 | *informative, non-blocking* | **ACTIVE / open · NON-BLOCKING** · completion **UNESTABLISHED** |

**PRE-ZIRCON TOP-5 BLOCKERS: NONE ESTABLISHED.**

---

## Notes that must not be lost

**#1.** `TEST-DEFAULT-001` remains `RESERVED` and unrun; it gains no `PASS` or `FAIL` from
any execution recorded here.

    DEFECT DISCOVERED         ≠ NEW ARCHITECTURE REQUIRED
    REQUIREMENT ALREADY OWNED ≠ REQUIREMENT IMPLEMENTED

**#3.** What dissolved is the *gating role*, not the test and not the concern. The
apparatus is still not under version control, and that remains untested.

    A CLAIM INSIDE A SPEC ABOUT ANOTHER DOCUMENT
      ≠ THAT DOCUMENT CONTAINING THE CLAIMED CONTENT

**#5 · Epoch 2.** The ledger's acceptance condition is *"N cycles with zero unattributable
receipts and at least one genuine `DATA_UNAVAILABLE` receipt."* **No value of N appears in
the inspected source.** None is invented here, and E2 is not closed.

The HELIX-JANUS deferral tests executed 2026-09-06 as a positive control **do not satisfy
E2**. They are the pre-existing unit tests, and E2's own record already states *"The
deferral path is unit-tested but not production-exercised."* The unmet condition is
production exercise.

    RE-RUNNING AN EXISTING UNIT TEST
      ≠ NEW EVIDENCE FOR AN EXPERIMENT THAT ALREADY RECORDS UNIT-TESTING AS INSUFFICIENT

No external failure was or will be manufactured; the ledger places that out of scope.

---

## Activation layers — kept distinct

| | Layer | Status per inspected source |
|---|---|---|
| A | Dependency-intelligence **method** use | **Permitted** — `zircon.md`: *"The METHOD is usable immediately"*; *"can be applied to current work today"* |
| B | Research-program organization / registering | **Permitted as research activity only.** Registration is explicitly not implementation authorization. Not organizational activation |
| C | Conceptual foundational receipt / admissibility **requirements** | **May be researched and reconciled.** No source inspected establishes that this layer depends on D or E, and none is asserted here |
| D | Infrastructure / application **build** | **RESERVE-ONLY · NOT ACTIVATED** |
| E | **Organizational** activation | **NOT ACTIVATED** — *"far downstream"* per source |
| F | Execution / actuation **authority** | **UNESTABLISHED · NOT GRANTED** |

    RESEARCHING A REQUIREMENT
      ≠ DEFINING PRODUCTION ARCHITECTURE
      ≠ IMPLEMENTING IT
      ≠ ACTIVATING THE SYSTEM
      ≠ GRANTING EXECUTION AUTHORITY

---

## Unresolved, carried forward

1. **Enforcement locus** — must admissibility be enforced at the boundary, or may it be
   delegated to producers? The 2026-08-27 ledger posed it; no completed work answers it.
   A design decision, not missing evidence.
2. Whether a **failed measurement** requires a state distinct from `DEFAULT_INHERITED`.
   Inside `default-state-admissibility`'s own reserved vocabulary scope.

Neither is resolved by assertion here.

---

## Boundary of this reconciliation

Derived from `iam-integrity` at `10525da` plus the uncommitted supersession changes, and
from executions recorded in their own receipts. One Zircon-mentioning line from the
enumeration run remains unread; other repositories were not searched for Zircon references.

No reserve was created. No doctrine was created. No application code was touched. No test
or experiment was run by this operation.
