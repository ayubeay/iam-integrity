# RESERVE — Contributor Continuity / Handoff Gate (CCHG)

Status: RESERVED — cross-stack organizational continuity primitive. NOT an active build.
Captured: 2026-08-27.
Origin: a real early-stage coordination problem — a contributor preparing to leave while
holding implementation context nobody else had.

## Core principle

**Departure is a human event. Continuity completion is an operational event.**
The architecture must not confuse the two.

A contributor's departure, replacement, reassignment or role transition is not
operationally complete merely because the person stops working. It is complete when the
relevant execution state has been transferred, reconciled, revoked or preserved as
appropriate, and receipted.

## Canonical continuity sequence

    assigned work → current state → source / code / artifacts → documentation
    → credentials / access → unresolved decisions → dependencies
    → successor context → continuity verification → receipt

## The gate

Before transition is marked COMPLETE, evaluate:

1. **Assigned work** — tasks, systems, projects, deliverables, responsibilities.
2. **Current state** — completed / in progress / blocked / abandoned / awaiting review /
   awaiting external dependency / unknown.
3. **Source, code, artifacts** — repositories, branches, documents, designs, datasets,
   prompts, configurations, deployment artifacts, media, research, working files
   available to the organization rather than solely under the departing contributor.
4. **Documentation** — enough for another authorized contributor or agent to understand
   what exists, how it works, what changed, why decisions were made, known limitations
   and unfinished work.
5. **Credentials and access** — what can be accessed, what belongs to the organization,
   what must be revoked, what ownership must transfer, whether shared credentials need
   rotation, whether service accounts or infrastructure would become inaccessible.
   **Do not preserve passwords or secrets inside continuity receipts.** Record state and
   required governance actions instead.
6. **Unresolved decisions** — architectural, product, operational, security, design.
7. **Dependencies** — work depending on the contributor, systems depending on their
   artifacts, external parties, blocked tasks, undocumented assumptions.
8. **Successor context** — enough for the incoming party to continue without
   reconstructing the previous contributor's work from scratch.
9. **Continuity verification** — a handoff does not pass because someone claims files
   were transferred. Where appropriate the successor or responsible authority
   acknowledges the package is usable.
10. **Receipt** — durable transition record.

## Dispositions

    HANDOFF_COMPLETE · HANDOFF_COMPLETE_WITH_EXCEPTIONS · HANDOFF_PENDING
    HANDOFF_BLOCKED · HANDOFF_INCOMPLETE · SUCCESSOR_UNASSIGNED · KNOWLEDGE_AT_RISK

## Risk-sensitive rigour

Do not make perfect documentation a universal prerequisite for departure.

    handoff rigour ∝ dependency + privilege + criticality + knowledge concentration
                   + replacement difficulty + operational impact

A contributor changing a marketing graphic needs a light handoff. A contributor holding
production infrastructure, cryptographic authority, financial operations or irreplaceable
institutional knowledge needs substantially stronger verification.

## Failure mode addressed

Without the gate: person disappears → knowledge disappears → ownership ambiguous →
credentials remain active or inaccessible → unfinished work hard to reconstruct →
dependencies surface later → replacement rebuilds context → execution slows or fails.

## Agent continuity extension

The same primitive governs replacement of autonomous agents. An agent should not simply
terminate or be replaced while holding unresolved execution state.

    agent identity → active objectives → execution state → working memory / context
    → artifacts → pending commitments → external dependencies → permissions
    → successor agent → state transfer → verification → receipt

This is what makes CCHG more than employee offboarding: the identical failure occurs when
an agent is killed, upgraded, swapped for another model, loses authorization, or is
reassigned mid-objective.

## Separation of concerns on departure

Distinguish IDENTITY · AUTHORITY · ACCESS · ACTIVE RESPONSIBILITY · HISTORICAL
CONTRIBUTION · ECONOMIC ENTITLEMENT. Removing operational authority must not erase
legitimate historical contribution; historical contribution must not preserve
credentials or operational authority.

## Relationship to existing canonical reserves

- IAM — identity, permissions, ownership transfer, revocation, successor authorization.
- OROS — transition sequencing, task reassignment, handoff execution.
- VERITY — confidence that handoff requirements were actually satisfied.
- `computable-accountability.md` — durable evidence of what transferred.
- `intent-continuity-primitive.md` — adjacent continuity semantics.
- `governed-work-attribution.md` — attribution survives the handoff.
- `organizational-state-transition-governor.md` — departure as an organizational event.

## Activation

Reserve only. Revisit when contributor, contractor, employee, vendor or autonomous-agent
transitions require governed continuity. Implementation should first determine whether
CCHG belongs as a cross-stack protocol, an OROS workflow primitive, an IAM lifecycle
extension, or a shared capability. Do not prematurely force module ownership.

RESERVED. DO NOT BUILD.
