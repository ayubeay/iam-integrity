# RESERVE — Recalculation Doctrine

Status: Foundational Execution Principle (Reserve)
Classification: Constitutional execution doctrine — governs existing
systems, introduces no new product
Captured: 2026-07-17

## Core philosophy

Execution systems must distinguish **temporary route failure** from
**destination failure**. Most autonomous systems retry the same execution
until limits are reached; a mature system asks: *should the route change
while preserving the objective?* The destination remains stable. The route
adapts. A GPS never says "you failed" — it says "recalculating." The
mission continues; only the path changes.

Traditional: Goal -> Try -> Fail -> Retry -> Retry -> Retry -> Timeout.
Doctrine: Goal -> Observe -> Obstacle -> Recalculate -> Alternative Route
-> Continue. The objective never changes unless explicitly instructed.

Cost grounding (LinkedIn loops signal): vague objectives create expensive
retry loops; clear intent, defined constraints, and better planning reduce
unnecessary iterations. The shift is from retrying to REASONING about
execution: is the objective clear; is the current path still appropriate;
should the plan adjust before spending more resources?

## How it governs existing modules

**OROS:** classify ROUTE_BLOCKED / RECALCULATING instead of FAILED.
**KONIGO Connect:** already the doctrine at the network layer — "primary
path degraded, searching alternate route, switching" rather than "ISP
failed." **HelixAtlas:** visualize original route -> obstacle -> alternate
route -> success; the route bends around blocked nodes, the original
remains visible as a historical branch — a living map of adaptation, not
red failure flashes. **VERITY:** score recalculations (count, reason,
policy compliance, integrity, outcome) — some recalculations improve
execution, some indicate instability; not all are equal. **IAM:** identity
remains continuous — an agent does not become a "new" agent by selecting a
different path; mission continuity preserved. **DRIFT:** a recalculation is
NOT automatically drift (original API down -> fallback API -> mission
completes is healthy adaptation); drift occurs only when the new route
changes the intended objective or violates policy.

## Reroute receipts

Every reroute produces an explanation: mission, original route, reason for
recalculation, alternative selected, policy approval, result, integrity —
execution stays observable rather than mysterious.

## Design principle

**Execution should optimize for mission completion, not attachment to a
particular route.** True for AI agents, network routing, payment routing,
logistics, mining orchestration, robotics, autonomous vehicles, business
workflows. The destination is the invariant; the route is adaptable. When
progress stalls, first ask whether the route should change rather than
repeating the same action.

## Relationships

Sibling doctrines: Adaptive Execution Layer (WHEN to adapt — evidence
gates) and Proof Before Promotion (how much evidence claims require).
The HANOI Planner reserve is the planning machinery this doctrine governs
at runtime; plan amendments there are recalculations here.

---

## Extension 2026-08-30 — Objective-Review Trigger

Status: RESERVED — DO NOT BUILD. Boundary refinement of this doctrine.

### Why this belongs here and not in its own file

An intake proposed a reserve for obstruction-aware execution reorientation. Almost all of
it is already here: *temporary route failure versus destination failure*, the
`Goal → Observe → Obstacle → Recalculate → Alternative Route → Continue` loop, reroute
receipts, OROS classifying `ROUTE_BLOCKED / RECALCULATING` rather than `FAILED`, and the
principle that execution should optimize for mission completion rather than attachment to
a route. Its failure taxonomy is `hanoi-planner.md`'s deficit classification; its
dispositions are distributed across this doctrine, `strategic-admissibility.md` and
`intelligence-resource-governance-layer.md`.

**One boundary survived.** This doctrine states that *the objective never changes unless
explicitly instructed* — a deliberate and load-bearing invariant against accidental
mission drift. What it did not carry is the case where obstruction is evidence about the
destination rather than the route. That is this doctrine's own boundary condition, so it
is named here rather than founding a competing parent.

### The complementary invariant

    ROUTE FAILURE       ≠  OBJECTIVE FAILURE      (this doctrine, unchanged)
    ROUTE PRESERVATION  ≠  OBJECTIVE IMMUNITY     (this extension)

Do not abandon an objective because one route failed. Equally, **a destination does not
become permanently unquestionable merely because it was once authorized.** An objective
may become nonviable, inadmissible, unsafe, or unsupported by the evidence that originally
justified it, and repeated obstruction is sometimes how that surfaces.

### Two cases, distinguished before recalculating

**1 · Route-local obstruction.** The objective remains valid; the path is blocked.
Recalculate, as above. This is the ordinary case and remains the default.

**2 · Objective-review trigger.** The obstruction appears structural, or is causally
relevant to an assumption the destination rests on. The correct action is **not** to
recalculate harder and **not** to abandon: it is to escalate the objective for
re-evaluation by whoever owns admissibility and authority.

    obstruction observed
    → route-local, or causally relevant to the objective's assumptions?
    → route-local:  recalculate
    → otherwise:    escalate for objective review

**This doctrine does not become the owner of objective rewriting.** It can detect that a
destination now warrants review; it holds no authority to change a mission. Review outcomes
are described conceptually — RETAIN · REAUTHORIZE · MODIFY · SUSPEND · ABANDON — and
deliberately not defined as a new state machine here. `strategic-admissibility.md` already
owns `ABANDON` as a resource-allocation decision, and vLOID owns admissibility.

### The counterfactual path test

The doctrine above says not to be attached to a particular route. This operationalizes it:

    If this execution path disappeared completely while the objective remained
    fixed, what other admissible routes would remain?

    viable alternatives exist   →  the evidence is primarily about the path
    no viable alternatives      →  objective review may be warranted

**The absence of a currently known route is not proof that the objective is impossible.**
Not knowing a path and there being no path are different findings, and only the first is
established by having looked. Treating them as one converts a search failure into a
mission verdict.

### Priority does not confer retry authority

    HIGH OBJECTIVE PRIORITY  ≠  UNBOUNDED RETRY AUTHORITY

A high-priority objective justifies more careful recalculation, not unlimited repetition.
The bounds are owned elsewhere and are cross-linked rather than restated:
`hanoi-planner.md` owns `LOOP_BUDGET` and the deficit classes, and
`intelligence-resource-governance-layer.md` owns continuation admission — *repetition
without state change* is that layer's signal, and this doctrine's *first ask whether the
route should change rather than repeating the same action* is the same instruction stated
as doctrine.

### Evidence boundary

The distinction above was exposed by a reflective reading of an oracle hexagram. That
reading is **the prompt that surfaced the question, not evidence that the principle is
correct.**

    SYMBOLIC INTERPRETATION  ≠  EMPIRICAL VALIDATION

The engineering claims stand or fall on their own terms, and nothing in this doctrine
depends on the originating reflection being anything more than a prompt.

RESERVED — DO NOT BUILD.
