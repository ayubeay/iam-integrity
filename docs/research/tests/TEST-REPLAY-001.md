# TEST-REPLAY-001 — Does the Replay Harness Reproduce Its Own Baselines?

Status: `RESERVED`
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
