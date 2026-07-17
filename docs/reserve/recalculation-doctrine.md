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
