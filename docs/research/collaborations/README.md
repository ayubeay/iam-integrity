# docs/research/collaborations — external collaboration governance

This directory holds **operational records**, not reserves and not specifications.

## Why it is separate

A reserve says *this may exist someday*. A specification says *this is how the question
would be answered*. A record here says *we owe a named person work*.

Those have different failure modes. A stale reserve is inert. **A stale obligation is a
broken commitment to someone who is waiting.** Filing live counterparty state in
`docs/reserve/` would let the second decay quietly with the properties of the first.

Nothing in this directory is architecture. Nothing here authorizes a build.

## The boundary that keeps this useful

    canonical spec/test   the reusable falsifiable protocol   → experiments/ or tests/
    case record           one run of it, with a counterparty  → here

When two collaborators exercise the same mechanism, **one canonical spec exists and each
collaboration references it as a separate case.** Two specs for one mechanism is how the
same question gets executed twice under two names.

**Where no reusable protocol exists yet, the canonical-spec field is left unresolved.**
It is not filled by inventing a spec to make the table look complete. An unresolved field
is accurate; a manufactured one is not.

## A record is not evidence

**A collaborator record is not evidence that their claim is true.** These stay separate:

    claim  →  proposed test  →  evidence received  →  result

A counterparty's statement enters as `claim`. It becomes `evidence received` only when
something arrives that could have contradicted it. Self-reported performance is
`SELF_REPORTED`, never `VERIFIED`.

## Status vocabulary

    PROPOSED                a lead; no commitment made by either side
    WAITING_ON_US           we owe the next action
    AWAITING_COUNTERPARTY   they owe the next action
    READY_TO_RUN            protocol agreed, resources available, not started
    ACTIVE                  running
    BLOCKED                 cannot proceed; the blocker is named
    COMPLETED               concluded, result recorded
    CLOSED                  ended without a result; the reason is recorded

`NEXT-ACTION OWNER` is mandatory on every record. It is the field that prevents an active
obligation from disappearing into a long conversation.

## Confidentiality and IP

Every record carries an explicit IP boundary in **both** directions.

- Do not copy a counterparty's proprietary protocol, chemistry, strategy logic or source
  into this repository. Generalizable findings may inform doctrine; their implementation
  may not enter canon.
- Do not disclose internal architecture, methodology, tooling or repository contents to a
  counterparty beyond what the collaboration requires.
- Access granted is not authorization. An invitation to observe an environment is not
  permission to scrape it, republish it, or treat what is posted there as verified.

## Doctrine

**Once an exploratory conversation creates a concrete experimental commitment, it becomes
an execution obligation and should remain visible until completed, explicitly abandoned,
superseded, or blocked.**

**A collaborative experiment succeeds when it resolves uncertainty honestly — not when it
produces the result either participant hoped for.**

**Silence, abstention, negative evidence and inconclusive evidence are valid outcomes when
the protocol supports them.** Do not manufacture activity to produce a record.

## Review

`REGISTER.md` is a live document with a review cadence, not an archive. A record whose
`LAST INTERACTION` is stale relative to its status is itself a finding — particularly any
row reading `WAITING_ON_US`.
