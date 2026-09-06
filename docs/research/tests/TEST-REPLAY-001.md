# TEST-REPLAY-001 — Does the Replay Harness Reproduce Its Own Baselines?

Status: `RESERVED` — **the test has not been run.**
Pre-Zircon gating purpose: **DISSOLVED** 2026-09-06 (PRE-ZIRCON GATE RECALCULATION). The test is preserved; what dissolved is its role as a pre-Zircon gate. See the dated addendum at the end of this file.
Registered: 2026-08-29
Migrated from: `docs/research/EXPERIMENT_LEDGER_2026-08-27.md` · Top-5 candidate 3

## Required behaviour

The replay runner, executed against its recorded baseline, reproduces its recorded results.

## Why now

The Zircon gate names a replay model explicitly, and replay apparatus already exists: a
replay runner with ten baseline and result JSON files across four versions.

**It is not under version control.** Nothing guarantees the baselines still correspond to
the runner. That is the entire reason to run this before relying on replay for anything
else.

## Procedure

1. Record the current hashes of runner and baseline files **before** execution, so a
   post-hoc change cannot be mistaken for a reproduction.
2. Run the runner against the recorded baseline.
3. Diff against the recorded results.
4. Record which of the four versions were tested and which were not.

## Pass

Results reproduce within the tolerance the baselines themselves declare. If they declare
none, that absence is recorded as part of the result rather than resolved by choosing one
now.

## Fail

Results do not reproduce. **This is a dated determinism finding and is itself the answer** —
it establishes that replay is not available machinery and must be rebuilt before any
experiment depends on it.

## Boundary

The versions actually executed, at the file hashes recorded in step 1. Untested versions
remain untested and must not be described as reproducing.

## Decision unlocked

Whether replay is available machinery or must be rebuilt. **This gates the cost of
`TEST-STALE-ARTIFACT-001` and `TEST-INCOMPLETE-001`**, which is why it runs first despite
being the least consequential of the three.

---
---

# DATED ADDENDUM — 2026-09-06 · PRE-ZIRCON GATING PURPOSE DISSOLVED

**Everything above is preserved unchanged.** This test was never run, and nothing here
claims a result.

## Two independent reasons, either sufficient

**1 · The cost it existed to reduce has already been paid.** Its own *Decision unlocked*
states the purpose: *"This gates the cost of `TEST-STALE-ARTIFACT-001` and
`TEST-INCOMPLETE-001`, which is why it runs first despite being the least consequential of
the three."* Both have since been adjudicated without relying on it —
`TEST-STALE-ARTIFACT-001` executed 2026-09-06, and `TEST-INCOMPLETE-001` superseded on a
premise adjudication.

    A COST-REDUCTION GATE ON WORK ALREADY COMPLETED IS SPENT, NOT PENDING

**2 · Its "Why now" rests on a claim the inspected source does not carry.** This spec and
the 2026-08-27 ledger both state *"The Zircon gate names a replay model explicitly."*
`docs/reserve/zircon.md`, read in full (92 lines), contains no replay model, no baseline,
no reproducibility requirement and no determinism clause. An uncapped search of
`docs/reserve` and `docs/research` returned no Zircon-owning document naming one.

    A CLAIM INSIDE A SPEC ABOUT ANOTHER DOCUMENT
      ≠ THAT DOCUMENT CONTAINING THE CLAIMED CONTENT

*Search boundary:* `docs/reserve` and `docs/research` in this repository. Other
repositories were not searched, and one Zircon-mentioning line from the enumeration run
remains unread.

## What this explicitly does NOT establish

    replay is useless in general                      NOT ESTABLISHED
    deterministic replay is unnecessary               NOT ESTABLISHED
    the replay harness works                          NOT ESTABLISHED
    the replay harness fails                          NOT ESTABLISHED
    the baselines still correspond to the runner      NOT ESTABLISHED

The concern that motivated it stands on its own: the apparatus is not under version
control. That remains true and remains untested. This test stays available, unrun, and may
be run whenever a purpose justifies it — but not as a pre-Zircon gate.

Reconciliation: `docs/research/PRE-ZIRCON-GATE-RECONCILIATION-2026-09-06.md`.
