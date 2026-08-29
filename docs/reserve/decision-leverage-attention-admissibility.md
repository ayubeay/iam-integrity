# RESERVE — Decision Leverage & Attention Admissibility

Status: RESERVED — DO NOT BUILD. Governance primitive, not a product, not a dashboard,
not an alerting system.
Captured: 2026-08-29.
Parent: vLOID governance surface. Consumes EMAA and Computable Accountability.

## Scope boundary — read first

Two reserves own adjacent territory and are **not restated here**:

- `emaa-external-machine-action-admissibility.md` owns receiver-side machine-action
  admissibility and the human-in-the-loop doctrine — `HUMAN-DEFINES-AUTHORITY +
  MACHINE-ENFORCES-BOUNDARIES + HUMAN-ESCALATION-FOR-EXCEPTIONS`, and the non-goal that
  it *"does not assume per-action human approval scales."*
- `computable-accountability.md` owns influence provenance, the Consequential
  Recommendation object and the `LOW / MODERATE / HIGH / CRITICAL` recommendation
  escalation ladder. Its research questions include, verbatim: *"How is uncertainty
  represented without creating alert fatigue?"*

EMAA states that per-action approval does not scale. **This reserve owns the mechanism
that statement presumes**: which events earn human attention, in what order, and what
happens to governance quality as the human channel saturates. It is the answer to
Computable Accountability's alert-fatigue question, and exists to be that answer rather
than a second governance surface.

## Core invariant

    EVENT IMPORTANCE  ≠  ATTENTION PRIORITY

An event may be objectively significant and still be the wrong thing to surface now.
Priority is a function of importance, decision leverage, reversibility, time remaining,
and **the current state of the attention channel** — not of importance alone.

**Noise can look responsible.** A governance surface that escalates everything has not
increased safety; it has converted review into ritual while appearing more diligent.

## The degradation ladder

Human review is a finite resource that degrades under load, and it degrades *silently*:

    ATTENTIVE     reviewer evaluates evidence and can change the outcome
    STRAINED      evaluation shortens; approvals track presentation rather than evidence
    SATURATED     queue exceeds capacity; triage becomes arrival order
    CEREMONIAL    approval is recorded without evaluation
    ABSENT        approval is delegated, batched or auto-confirmed in practice

**The receipt is identical at every rung.** A `CEREMONIAL` approval and an `ATTENTIVE`
one produce the same signed artifact, which is why the degradation is invisible to any
system that counts approvals. Governance quality can fall to zero while every audit
metric improves.

## Decision leverage

Not every decision changes an outcome. Leverage is the degree to which a human's
available choices alter the reachable outcome set:

    HIGH    the choice materially changes what happens, and the reviewer can affect it
    LOW     the outcome is substantially determined before the choice is presented
    NIL     the only admissible response is the one already proposed

**A `NIL`-leverage escalation is not governance. It is an audit artifact.** Presenting one
consumes attention that a `HIGH`-leverage decision needed, so leverage is an admissibility
input rather than a reporting field.

## Attention admissibility loop

    event → significance assessment → decision leverage → reversibility and time window
    → current channel state → attention admissibility
    → SURFACE_NOW / QUEUE / BATCH / DELEGATE / SUPPRESS_WITH_RECORD / DEFER
    → human evaluation → decision → receipt including channel state at decision time

**`SUPPRESS_WITH_RECORD` is never silent.** Suppression that leaves no trace is
indistinguishable from a system that never observed the event. The record is what makes
suppression a governed act rather than a gap.

## Channel state as a receipt field

A decision receipt should preserve the state of the attention channel at the moment of
decision — queue depth, elapsed reviewer time, concurrent escalations, degradation rung.
Without it, a later reconstruction cannot distinguish a considered approval from a
saturated one, and Computable Accountability's causal chain terminates at an approval
whose quality is unknowable.

## Attention as a governed resource

Attention is a resource constraint in the sense `hanoi-planner.md` already uses — its
Resource constraint class covers "budget, compute, time windows, rate limits, staff."
Human review capacity belongs to that class. This reserve does not restate the planner's
constraint model; it supplies the admissibility rule that decides what consumes the
budget.

## Anti-patterns

Escalating everything and calling it caution · measuring governance by approval count ·
treating approval latency as the only quality signal · suppressing signal to reduce load
without recording the suppression · surfacing `NIL`-leverage decisions to demonstrate
oversight · assuming a signed approval implies an evaluated one · adding a reviewer as
the remedy for a saturated channel without changing what reaches it.

## Relationship to existing canonical reserves

`emaa-external-machine-action-admissibility.md` (human-in-loop doctrine, authority
budgets) · `computable-accountability.md` (influence provenance; consumes the channel-state
receipt field) · `hanoi-planner.md` (Resource constraint class) ·
`strategic-admissibility.md` (`ADVANCE / HOLD / EXPERIMENT / GATHER_EVIDENCE / DEFER /
ABANDON` — a strategic decision may itself be low-leverage) · `human-recovery-mesh.md`
(governed human intervention) · vLOID · OROS · receipts.

## Research questions

How is decision leverage estimated before the decision is made? Can channel degradation
be detected from receipts alone, or does detection require measuring evaluation itself?
At what queue depth does approval quality measurably fall, and is that threshold
per-reviewer? How should suppression records be reviewed so that suppression does not
become an unexamined filter? Does batching preserve evaluation quality or merely relocate
the saturation?

## Non-goals

Not an alerting product, not a notification router, not a workload-management tool, not a
reviewer-performance monitor, and not authorization to reduce human oversight. The
purpose is to make oversight real where it is claimed, not to justify having less of it.

## Activation

Revisit when a consequential system in the portfolio routes decisions to humans at a rate
one reviewer cannot evaluate; when approval records exist that cannot be distinguished
from ceremonial approvals; or when Computable Accountability requires a channel-state
field to reconstruct why a recommendation was accepted.

RESERVED — DO NOT BUILD.
