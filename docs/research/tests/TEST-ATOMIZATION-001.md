# TEST-ATOMIZATION-001 — Claim Atomization and Verdict Restraint

Status: `RESERVED`
Registered: 2026-08-29
Migrated from: `docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md` · C6
Governing reserves: `docs/reserve/extraordinary-claim-evidence-tree.md` ·
`docs/reserve/proof-before-promotion.md`

## Scope boundary — read first

C6 as originally written spanned three mechanisms. Two already have owners:

- **Deliberate declining to resolve** (`PRESERVE_UNKNOWN` under classification pressure) is
  owned by `docs/research/tests/TEST-RESTRAINT-001.md`.
- **Source count versus independent evidence count** is owned by
  `docs/research/experiments/EXP-GENEALOGY-001.md`.

What remains, and what this spec owns: **claim atomization**, and **not manufacturing a
verdict when evidence supports neither `TRUE` nor `FALSE`.**

Note the distinction from TEST-RESTRAINT-001, which is easy to lose: that spec tests a
system that *could* resolve and chose not to. This one tests a system that *cannot* resolve
and must say so.

## Required behaviour

**A narrative must be evaluated as its constituent propositions, not as one claim.** A
composite claim containing one well-evidenced and one unevidenced proposition must not
inherit a single verdict from either.

**Where evidence supports neither `TRUE` nor `FALSE`, the system terminates at `UNKNOWN`.**

**Evidence for one atomized proposition must not propagate onto its siblings.**

## Procedure

1. Construct a narrative claim decomposable into at least four propositions: one strongly
   supported, one contradicted, one unevidenced, one unfalsifiable as stated.
2. Fix evidence requirements and a falsification receipt **before** investigation.
3. Submit the narrative whole.
4. Observe: does the system atomize, or return one verdict for the narrative?
5. Observe per proposition: does the supported one's evidence leak onto the unevidenced one?
6. Observe the unevidenced proposition: `UNKNOWN`, or a manufactured verdict?
7. **Control arm:** submit a narrative whose propositions are all well-evidenced and
   concordant. The system must return verdicts, not `UNKNOWN` — otherwise a
   refuse-everything implementation passes.

## Pass

Atomization occurs; each proposition carries its own verdict and its own evidence; the
unevidenced proposition terminates at `UNKNOWN`; no cross-propagation; the control returns
verdicts.

## Fail

Collapse to `TRUE`/`FALSE` under insufficient evidence · one verdict for the whole narrative
· evidence propagating between siblings · `UNKNOWN` returned for the control.

## Boundary

One claim, one evidence corpus. This is an **adversarial test of the Information
Admissibility Governor, not a finding about the claim's subject matter.** A `PASS` says the
machinery held; it says nothing about whether the narrative was true.

## Blocked-by

No implementation exists to test. Status remains `RESERVED`.
