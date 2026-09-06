# TEST-INCOMPLETE-002 — Enforced-Model Incompleteness / Missing-Evidence Admissibility

Status: `FROZEN — NOT YET RUN`
Registered: 2026-09-06
Supersedes: `TEST-INCOMPLETE-001` (PREMISE_REJECTED — scope mismatch, not implementation
failure). The superseded test is preserved intact; this one does not inherit its requirement.

**ID rationale.** `-002` follows the `EXP-FAIR-COMPUTE-002` precedent: a later iteration on the
same subject, keeping the lineage legible. If house convention prefers a distinct subject name
(`TEST-ENFORCED-INCOMPLETENESS-001`), rename before registration — the content is unaffected.

---

## THIS DOCUMENT IS FROZEN BEFORE OBSERVATION

Acceptance and rejection conditions below were written before the governing path was
identified and before any fixture was run. No outcome is predicted; all outcomes in §9 are
admissible.

    DO NOT INSPECT THE RUNTIME ANSWER AND THEN CHOOSE THE ACCEPTANCE CRITERIA

---

## 1 · WHY THIS TEST EXISTS

`TEST-INCOMPLETE-001` asked whether production consumers gate on
`validated_five_shadow.score_status`. Phase 1.5 established that field was declared
`enforced: false`, inside a model whose doctrine document stated *"Production remains 0.5.3
unchanged"*, 25 days before the obligation was written. The test asked a valid-sounding
question about a component that carried no such obligation.

The underlying pre-Zircon concern survives: **if the model that actually governs cannot
represent missing evidence, every receipt Zircon inherits carries that hole.**

This test asks that question of the model that governs.

---

## 2 · TWO INVARIANTS, CONCLUDING INDEPENDENTLY

**INVARIANT A — ENFORCED PRODUCER INCOMPLETENESS SEMANTICS.**
When the governing production path cannot evaluate a required signal, does it explicitly
represent that incompleteness in a way capable of reaching a consumer?

**INVARIANT B — ENFORCED CONSUMER ADMISSIBILITY.**
If such a state exists and can reach consumers, does every enumerated production consumer
handle it according to the enforced model's own contract?

**A and B conclude on separate lines.** A may fail while B is inapplicable. A may pass while
B fails. A combined verdict destroys the distinction.

---

## 3 · TARGET — TO BE ESTABLISHED, NOT ASSUMED

The subject is the model/path that **actually governs** production scoring at the tested
commit.

    PRODUCTION-LABELLED     ≠ VERIFIED EXECUTION PATH
    ENFORCED:true SOMEWHERE ≠ THIS SCORE GOVERNS THE CONSUMER

The governing path must be identified from code before probing. Version labels are not
evidence.

`validated_five_shadow` is **NOT** the enforcement subject. It may be retained as a
comparison control, because it demonstrates a known explicit incompleteness representation.

---

## 4 · INVARIANT A — WHAT COUNTS

Vocabulary is not required. Any mechanism counts if it is **proven** to represent
incompleteness in a way that could govern:

    · an explicit status field
    · a null / absent score
    · a degraded confidence that changes admissibility
    · a coverage state that actually changes the score or the verdict
    · refusal to score
    · any other explicit representation, demonstrated

**Guards, binding on this invariant:**

    LEXICAL ABSENCE ≠ SEMANTIC ABSENCE
    A COVERAGE FIELD EXISTS ≠ INCOMPLETENESS GOVERNS THE SCORE
    A NULL SIGNAL SILENTLY CONVERTED INTO A NUMERIC CONTRIBUTION
      ≠ INCOMPLETENESS REPRESENTED
    A DIFFERENT NUMERIC SCORE ≠ INCOMPLETENESS DETECTION
      unless the model explicitly establishes that the difference represents
      missing-evidence admissibility

**A PASSES** if an incompleteness representation exists that is capable of reaching a
consumer and is distinguishable from a fully-observed result.

**A FAILS** if missing evidence is absorbed into the score with no representation a consumer
could act on.

**A INCONCLUSIVE** if a candidate representation exists but whether it governs cannot be
established from the frozen corpus.

---

## 5 · INVARIANT B — CONDITIONAL, AND NOT INHERITED FROM THE SUPERSEDED TEST

B is evaluated **only if** A exposes a state capable of reaching consumers.

The superseded test's requirement — *"every downstream consumer must gate on it"* — is **NOT**
imposed. B must first establish what consumer behaviour the enforced model's own contract
requires, then test against that.

**B PASSES** if every enumerated production consumer handles the state per the enforced
contract, and behaves ordinarily on the resolved control.

**B FAILS** if any consumer treats an incomplete result as fully observed, contrary to the
enforced contract.

**B NOT_APPLICABLE** if A establishes that no incompleteness state reaches any consumer.
**This is not a PASS.** It means the question cannot arise, which is itself the finding.

---

## 6 · CONSUMER SET — inherited from Phase 1, frozen

Production consumers of `calculateSurvivalScore` in `survivor-oracle` at `078ac95`:

    src/index.js:121 · src/monitor.js:62 · src/rescorer.js:31 · src/attest.js:37

Research callers (9) are reference only and never substitute for a production consumer.
`survivor-shield-sdk`, `agentguard`, `poi-engine` contain no reference and are not consumers.

---

## 7 · FIXTURES

    UNRESOLVED ARM   synthetic tokenData with one genuinely unresolvable input that the
                     GOVERNING model requires
    RESOLVED CONTROL identical shape with that input resolved

Smallest fixture that traverses the real scoring path. Synthetic and local only.

**Forbidden:** network · RPC · trading · capital · credentials · external APIs · third-party
infrastructure · production writes · application-code modification.

Application code may be **executed** locally as required. It may not be **changed**.

---

## 8 · CONTROL AGAINST A FALSE POSITIVE

The resolved control must produce ordinary behaviour. **A system that refuses or degrades on
both arms does not pass** — it would be indistinguishable from one that refuses everything.

Both arms are required for either invariant to conclude.

---

## 9 · ADMISSIBLE OUTCOME SPACE — no outcome is predicted

    A PASS / B PASS
    A PASS / B FAIL
    A PASS / B INCONCLUSIVE
    A FAIL / B NOT_APPLICABLE
    A INCONCLUSIVE / B NOT_APPLICABLE
    BLOCKED

    THE SHADOW MODEL HAVING BETTER MISSING-EVIDENCE SEMANTICS
      ≠ THE PRODUCTION MODEL HAVING A DEFECT
    — until the production model is tested against its OWN contract.

---

## 10 · BOUNDARY

The governing path and consumers enumerated at `survivor-oracle 078ac95`, on the fixtures
used. Establishes nothing about scoring architecture generally, nothing about runtime
behaviour under real token data, and nothing about any deployed instance.

**A `PASS` is never evidence that an external-world hypothesis is validated.**

---

## 11 · WHAT THIS TEST DOES NOT DO

Does not fix any defect it finds · does not modify application code · does not promote any
shadow model · does not satisfy `TEST-STALE-ARTIFACT-001` · does not investigate claim-level
promotion · does not activate Zircon.

    THE TEST MAY DISCOVER A DEFECT. THIS OPERATION DOES NOT FIX THE DEFECT.

---
---

# POST-FREEZE ADDENDUM — 2026-09-06, after execution

**Everything above this marker is the specification as frozen at 2026-09-06T16:34:23Z,
sha256 `dff27c5cb60820dfaff634c073315a5e1b1b6a76d938595c4af1eabdad0602ba`, byte-identical.
It is preserved unchanged, including its `FROZEN — NOT YET RUN` status line, because
amending a specification after observation is what freezing exists to prevent.** Current
lifecycle status is recorded here instead.

    LIFECYCLE STATUS   EXECUTED 2026-09-06
    INVARIANT A        PASS
    INVARIANT B        INCONCLUSIVE
    EXECUTION RECEIPT  TEST-INCOMPLETE-002-EXECUTION-RECEIPT-2026-09-06.md

---

## CLASSIFICATION — surface instance, not a new mechanism

    SURFACE INSTANCE OF AN OWNED DOCTRINE
    Governing reserve:  docs/reserve/default-state-admissibility.md
    Nearest test owner: TEST-DEFAULT-001

**This test discovers no new doctrine.** It was written while correcting the scope mismatch
in `TEST-INCOMPLETE-001`, and the collision check performed after execution found the
mechanism already owned.

    A NEUTRAL DEFAULT FOR A FAILED MEASUREMENT
      IS A VALUE INHERITED RATHER THAN ESTABLISHED

    A SECOND SURFACE OF AN OWNED DOCTRINE ≠ A NEW MECHANISM

`TEST-DEFAULT-001`'s fail condition — *"the record is indistinguishable from a deliberate
selection · a downstream consumer treats an inherited value as confirmed intent · the
distinction exists at the producer but is dropped in transit"* — describes the observed
behaviour in all three clauses.

## WHY A DISTINCT TEST IS NEVERTHELESS JUSTIFIED

`TEST-DEFAULT-001`'s own boundary: *"passing one surface does not clear another."*

    ONE DOCTRINE MAY OWN MULTIPLE INDEPENDENTLY TESTABLE SURFACES

**Anti-proliferation rule applied.** A second surface justifies a distinct test only where it
introduces a materially different origin, propagation path, consumer boundary, enforcement
behaviour, or consequence. This surface differs on four of five:

    origin              measurement failure, not human inaction at a form
    propagation         neutral numeric substitution inside a scorer
    consumer boundary   an attestation that is cryptographically signed
    consequence         a signed verdict crosses a trust boundary without its
                        completeness state

`TEST-DEFAULT-001`'s procedure — submit a form without touching a field — cannot exercise
this path.

## TEST-DEFAULT-001 IS UNCHANGED

    A FINDING ON ANOTHER SURFACE OF THE SAME DOCTRINE
      ≠ EVIDENCE FOR A TEST WHOSE PROCEDURE WAS NOT RUN

`TEST-DEFAULT-001` remains `RESERVED` and gains no PASS or FAIL from this execution.

---

## OBSERVED SURFACE — narrow statement

> The enforced scoring path can substitute a neutral/default value for a failed measurement
> while separately retaining evidence that the measurement was unavailable; the inspected
> attestation path does not carry that evidence-completeness distinction with the resulting
> verdict.

Established at `survivor-oracle 078ac95`, on the fixtures used, for the paths inspected.
**No generalization beyond that path and commit.**

`INVARIANT B` is **not** upgraded on static evidence. Three consumers were shown by static
reading never to read `coverage`; that is not an executed observation of their behaviour, and
the enforced contract (*"reported only"*) imposes no obligation to test them against.

---

## PRE-ZIRCON REQUIREMENT CANDIDATE

**Correction to the execution receipt §7.** That section's wording risks reading as though
Zircon already has a receipt schema. It does not — Zircon is reserve-only and unbuilt. The
supported formulation:

    PRE-ZIRCON REQUIREMENT CANDIDATE
    A future Zircon receipt / admissibility design must not silently inherit the tested
    pattern, in which a verdict can cross a boundary without the evidence-completeness
    state needed to interpret that verdict.

This is a **requirement candidate** until the pre-Zircon gate is reconciled. No architecture
is created from it.

## RESEARCH-METHOD FINDING — recorded, not canonized

    MECHANISM NOVELTY DETERMINES DOCTRINE OWNERSHIP
    SURFACE NOVELTY DETERMINES WHETHER ANOTHER EMPIRICAL TEST IS JUSTIFIED

One instance. Held as a candidate. Promotion only if it prevents duplicate doctrine while
preserving necessary testing across several further cases.
