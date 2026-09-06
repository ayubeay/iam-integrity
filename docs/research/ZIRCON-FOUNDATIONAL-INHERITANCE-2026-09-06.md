# Zircon Foundational Inheritance — 2026-09-06

**FOUNDATIONAL INHERITANCE SET FOR A FUTURE ZIRCON EVIDENCE / RECEIPT DESIGN.**

Not a Zircon receipt schema. No production schema exists, and none is defined here.
The inheritance set **constrains** a future design; it is not that design.

---

## 1 · Purpose and boundary

This record answers one question: *what requirements have already been earned elsewhere
that a future Zircon evidence or receipt design must inherit rather than reinvent?*

It does **not** answer where each requirement must be enforced, and it does not design,
implement, or authorize anything.

    REQUIREMENT INHERITANCE
      ≠ MODULE OWNERSHIP TRANSFER
      ≠ IMPLEMENTATION
      ≠ PRODUCTION SCHEMA DEFINITION
      ≠ ENFORCEMENT LOCUS
      ≠ ZIRCON ACTIVATION

**Ownership does not move.** Each requirement below stays with its canonical owner. A
future Zircon design inherits the obligation; the reserve keeps the doctrine.

    RESEARCHING A REQUIREMENT
      ≠ DEFINING PRODUCTION ARCHITECTURE ≠ IMPLEMENTING IT
      ≠ ACTIVATING THE SYSTEM ≠ GRANTING EXECUTION AUTHORITY

---

## 2 · Baseline

    commit   950bfa53d8529a0b218a41b12d66683659f05629
    parent   10525dabf9b5aae8099c9d486d993c39c6a7b9eb
    worktree clean · unpushed · no application code involved

---

## 3 · Owner search and controls

Two read passes.

    PASS 1   4 files reused a prior full read, proven byte-identical by
             `git diff 10525da..950bfa5` returning empty for each
             6 files read in full
             1 file — `computable-accountability.md` — read to line 300 of 354
               and marked INCOMPLETE

    PASS 2   12 further files read in full
             `computable-accountability.md` completed

    DISTINCT 4 identity + 6 pass-1 full + 1 completed across passes + 12 pass-2 full
             = 23

A full read at one baseline plus proven byte identity at another is the same content
basis, not recollection. **Zero blocks left INCOMPLETE at the end of pass 2.**

    A FILE PARTIALLY READ IN PASS 1 AND COMPLETED IN PASS 2
      ≠ A FILE READ IN FULL IN PASS 1

Controls each pass: a known owner had to resolve; a runtime-generated token had to return
zero. Owner discovery reported matching **files** per concept rather than matching lines,
because a search hit is a candidate to read, not an owner.

    KEYWORD OVERLAP ≠ MECHANISM OWNERSHIP
    A CROSS-REFERENCE TO AN OWNER ≠ DEMONSTRATION THAT THE OWNER OWNS THE MECHANISM
    A FILE MATCHING A CONCEPT SEARCH ≠ AN OWNER

**Selection is itself a search with an opportunity space.** The map offered 44 / 61 / 50 /
92 files on four concepts; twelve were read. Exclusions were made on sourced grounds —
indexes (`EVIDENCE_DISCIPLINE` §8: *"Reconciliation reads files, not the index"*), staging
(self-declared pre-canonical), tests as owners (a test names a governing reserve), and
domain applications (IVCP: *"Domain instances consume this; they do not restate it"*). The
last is the weakest; `underwater-duration-edge-decay-admissibility` and
`regime-evidence-engine`, both named by `temporal-evidence-admissibility` itself, would be
the first to promote if C3 proves to have a competing owner.

The highest-density match in the entire map — `operational-sovereignty-dependency-independence`,
39 hits — was **read rather than excluded on its title**, and its own scope boundary
disclaims the evidence side, naming IVCP's `SHARES_SOURCE_WITH` as the analogue.

---

## 4 · Inheritance matrix

| # | Requirement | Canonical owner | Type | Enforcement locus |
|---|---|---|---|---|
| C1 | provenance envelope survives transformation | ELSPE | DIRECT | UNSPECIFIED |
| C2 | event / observation / knowledge / decision / execution times stay distinguishable | temporal-evidence-admissibility + ELSPE | BY_COMPOSITION | UNSPECIFIED |
| C3 | historical validity is not present admissibility | temporal-evidence-admissibility | DIRECT | UNSPECIFIED |
| C4 | lifecycle state is not admissibility | ELSPE | DIRECT | UNSPECIFIED |
| C5 | the recorded source corresponds to the source actually consumed | **UNESTABLISHED** | CONDITIONAL | UNESTABLISHED |
| C6 | inherited values stay distinguishable from established ones | default-state-admissibility | DIRECT | UNSPECIFIED |
| C7 | missing evidence must not silently become established evidence | agent-metacognition-calibration-layer | DIRECT | UNSPECIFIED |
| C8 | UNKNOWN / UNRESOLVED remain admissible epistemic states | anomaly-handling-protocol | DIRECT | UNSPECIFIED generally; specified for the anomaly domain only |
| C9 | later source change does not rewrite historical observation | ELSPE | DIRECT | UNSPECIFIED |
| C10 | upstream invalidation obliges reconsideration, not reversal | computable-accountability | DIRECT | **ALREADY_SPECIFIED** |
| C11 | a later auditor can reconstruct a decision's evidence basis | computable-accountability | DIRECT | UNSPECIFIED |
| C12 | independence is not inferred from source or agent count | independent-validation-capability-promotion | DIRECT | UNSPECIFIED |
| C13 | evidence used to generate a hypothesis is not confirmation of it | EVIDENCE_DISCIPLINE §6.5 | DIRECT | NOT_APPLICABLE |
| C14 | multiplicity and selection opportunity space survive into interpretation | EVIDENCE_DISCIPLINE §6.5 | DIRECT | NOT_APPLICABLE |
| C15 | status attaches at claim level where claims conclude independently | EVIDENCE_DISCIPLINE §3 | DIRECT | NOT_APPLICABLE |

**Implementation status is uniform and is not inferred.** Every owner above is
`RESERVED — DO NOT BUILD`, or, for `EVIDENCE_DISCIPLINE`, a research discipline that is
explicitly *not a gate*. Owned as specification; implemented nowhere.

    REQUIREMENT ALREADY OWNED ≠ REQUIREMENT IMPLEMENTED

---

## 5 · Requirements inherited directly

Thirteen. Each is stated by a single owner in a form that requires no composition to
produce the obligation.

**C1 · C4 · C9 — ELSPE.** Provenance envelope with transformation history and the rule
that `published_at ≠ observed_at` must survive normalization; the section *State is not
admissibility*; the historical-evidence principle that a later source change must not
silently rewrite the observation record. *Adjacent, no bridge established:*
`verity-provenance-resilience` owns provenance survival under marker loss and neither file
references the other.

**C3 — temporal-evidence-admissibility.** *"Historical validity ≠ present admissibility"*,
verbatim, with the signal lifecycle and the point of inadmissibility.

**C6 — default-state-admissibility.** Absence of intervention must not be read as
affirmative intent; the inaction-as-state vocabulary; the `UNTOUCHED → DEFAULT → FACT`
promotion failure.

**C7 — agent-metacognition-calibration-layer.** States the obligation directly:
`NOT_MEASURED ≠ MEASURED_SAFE`, `NOT_OBSERVED ≠ OBSERVED_NEGATIVE`, *"uncertainty must
propagate rather than silently becoming success"*, and *"an agent must never silently
transform absence of evidence into evidence of success."* Supporting adjacent surfaces —
not co-owners — are `default-state-admissibility` and `verity-provenance-resilience`.

    MULTIPLE SURFACES SUPPORTING A REQUIREMENT
      ≠ OWNERSHIP BY COMPOSITION WHEN ONE OWNER ALREADY STATES IT DIRECTLY

**C8 — anomaly-handling-protocol.** States the requirement almost verbatim: *"UNKNOWN and
UNRESOLVED are legitimate, valid governed states"*, alongside the epistemic separation
`OBSERVED / INFERRED / HYPOTHESIZED / REPRODUCED / VERIFIED / FALSIFIED / UNRESOLVED` that
must never silently collapse. `proof-before-promotion` (*"UNKNOWN is load-bearing"*) and
`extraordinary-claim-evidence-tree` (*"UNKNOWN IS A VALID COMPUTATIONAL OUTPUT"*) are
**supporting doctrinal lineage**, showing how the requirement arose in claim evaluation —
not co-owners required to produce it.

    SUPPORTING ANCESTRY ≠ REQUIRED COMPOSITION
      WHEN ONE OWNER ALREADY STATES THE REQUIREMENT DIRECTLY

**C10 · C11 — computable-accountability.** The reconsideration obligation and its three
boundaries; the causal-chain reconstruction invariant. Its own boundary with
`invariant-precomputation` is written into the text: that reserve governs invalidation
checked *on reuse*; this one names the gap where a conclusion has already authorized an
action and has no further reuse event.

**C12 — independent-validation-capability-promotion.** `N_effective ≤ N_raw`, the Evidence
Independence Graph, and the explicit refusal to define a formula. Three reserves declare
themselves consumers of it rather than restating it.

**C13 · C14 · C15 — EVIDENCE_DISCIPLINE.** §6.5 and §3. Enforcement locus is
`NOT_APPLICABLE`: this document states of itself that it is *not a gate* — later work
inherits the discipline rather than passing through it.

---

## 6 · Requirements inherited by composition

**One.** Its semantic bridge is **written into the documents**, not constructed by this
reconciliation.

**C2 — temporal-evidence-admissibility + ELSPE.** The child declares itself *"Child of
ELSPE … Neither restates the other"* and partitions ownership explicitly: lifecycle state
and provenance envelope to the parent; decay, persistence and the point of inadmissibility
to the child.

    TWO OWNERS NAMING A COMMON THIRD ≠ A BRIDGE BETWEEN THEM
    A CROSS-REFERENCE ≠ A COMPOSITION BRIDGE

Those tests are why C7 and C8 are direct rather than composed, and why the C1 /
`verity-provenance-resilience` relationship is recorded as adjacency rather than
composition. C8 was first drafted as a three-owner composition resting partly on a
cross-reference; direct ownership removes the need to establish whether that
cross-reference would have qualified as a bridge at all.

---

## 7 · Conditional and unresolved

**C5 — the recorded source corresponds to the source actually consumed.**

`OWNER UNESTABLISHED.` Four independent phrasings returned no reserve stating this
invariant in a general cross-system form.

    FIELDS REQUIRED TO RECONSTRUCT SOURCE
      ≠ EXPLICIT REQUIREMENT THAT THE RECORDED SOURCE
        CORRESPOND TO THE SOURCE ACTUALLY CONSUMED

Adjacent ownership is substantial and constrains the same failure mode: `context-integrity`
requires source identity and a receipt that reconstructs what was actually supplied;
`computable-accountability` requires reconstructing where information originated and what
source asserted it; `provider-qualification-and-routing` preserves provider identity,
retrieval method, source object reference, transformation history and raw-evidence
reference, and states that a normalized response must not silently become primary truth;
`EVIDENCE_DISCIPLINE` §4 requires source-artifact provenance.

**This is recorded as an ambiguity, not resolved.** It may be an unstated invariant already
implied by those owners rather than a genuine gap. **No reserve is created, and none is
proposed.**

Worth preserving: this is the invariant that **passed** in `TEST-STALE-ARTIFACT-001`
(Invariant B), while the invariant that **failed** (A, temporal admissibility) is the one
fully owned. Passing a test and having a doctrinal owner are unrelated properties.

---

## 8 · Enforcement locus — explicitly left open

Not answered here, and not answerable here: **must admissibility be enforced at a shared
boundary, or may some requirements be delegated to producers or consumers?**

    A REQUIREMENT BEING OWNED ≠ ITS ENFORCEMENT LOCUS BEING SETTLED

**Material for:** C1 · C2 · C3 · C4 · C5 · C6 · C7 · C9 · C11 · C12.

**Already specified, and only for its own scope:** C10. `computable-accountability` rules
that *reconsideration is horizontal; consequence is domain-owned* — the obligation is
recorded centrally and routed to the named domain owner.

**Specified for one domain, NOT generally:** C8. `anomaly-handling-protocol` specifies an
enforcement posture — capability restriction while a cause is unresolved — for the anomaly
domain. That does not settle where the general rule *"UNKNOWN and UNRESOLVED are admissible
epistemic states"* would be enforced across a Zircon design.

    A DOMAIN-SPECIFIC ENFORCEMENT ANSWER ≠ A GLOBAL ENFORCEMENT LOCUS
    DIRECT OWNERSHIP OF A RULE
      ≠ GLOBAL SCOPE FOR THAT OWNER'S ENFORCEMENT POSTURE

C8's owner is direct (§5). That settles where the semantic rule comes from; it does not
extend AHP's anomaly-domain posture to every surface a Zircon design would touch.

**Not applicable:** C13 · C14 · C15, which are research-selection discipline rather than
runtime obligations.

---

## 9 · Tested and not inherited

**None.** Fifteen candidates were tested; none dissolved and none fell out of scope.
Fourteen resolved to an existing owner; one (C5) resolved to `UNESTABLISHED`.

**No requirement was found that justifies a new doctrine, a new reserve, or a new module.**

---

## 10 · What this record does not authorize

    Zircon infrastructure or build            NOT AUTHORIZED — reserve-only
    Zircon organizational activation           NOT AUTHORIZED — far downstream per source
    a production receipt or evidence schema    NOT DEFINED — none exists
    implementation of any requirement above    NOT AUTHORIZED — all owners say DO NOT BUILD
    enforcement-locus resolution               NOT ANSWERED
    execution or actuation authority           NOT GRANTED, and not inferable from research
    any new reserve, doctrine or module        NONE CREATED, NONE PROPOSED
    ownership transfer into Zircon             NONE — owners are unchanged

**The ownership shape is federated, deliberately.** ELSPE holds evidence state and
provenance; temporal-evidence-admissibility holds temporal admissibility;
agent-metacognition-calibration-layer holds epistemic incompleteness;
default-state-admissibility holds inherited-versus-established semantics;
independent-validation-capability-promotion holds independence; computable-accountability
holds decision reconstruction and reconsideration; EVIDENCE_DISCIPLINE holds
research-selection discipline.

**A future Zircon design inherits these without swallowing them.** No single monolithic
owner is required, and consolidating them would destroy the boundaries each file was
written to protect.

---

## 11 · Evidence boundary

Derived from `iam-integrity` at `950bfa5`. **Twenty-three distinct files**: nineteen read
in full across two passes, and four reused by proven byte identity to a full read at
`10525da`. Zero blocks left INCOMPLETE.

The two read passes reported eleven and thirteen files respectively;
`computable-accountability.md` appears in both — truncated in the first, completed in the
second — so the distinct total is twenty-three, not twenty-four.

Establishes what the inspected files state. It establishes nothing about systems, nothing
about any implementation, and nothing about repositories not searched.

**Known imprecision, carried forward.** The `2026-09-06` gate reconciliation and the
`TEST-REPLAY-001` addendum describe *"zircon.md's complete 92-line text"*. The file is 117
lines total, 92 non-empty; the earlier label used non-empty units. All 117 lines were
printed and read, so the finding those records rest on is unaffected — the figure is in the
wrong unit and is corrected here rather than by amending a clean commit.

`EXP-GENEALOGY-001` is named by `computable-accountability` as the falsifiable form of
source independence and was **not** read. It is a `PROPOSED`, unrun experiment: it can test
a mechanism, not establish ownership of one.

---

## 12 · Next permitted operations

Two independent paths. **Neither is a prerequisite for the other**, and this operation
found no source dependency between them.

**PATH B — dependency-intelligence research registration and application.** Permitted now
by `zircon.md`'s own words. Records produced under it may cite this inheritance set.

**PATH C2 — enforcement-locus adjudication.** Narrower than the 2026-08-27 framing implied:
open for ten requirements, already settled for C10, and settled for C8 only within the
anomaly domain.

Neither path activates Zircon. Neither grants execution authority.

RESEARCH RECORD. NO BUILD. NO ACTIVATION. NO NEW DOCTRINE.
