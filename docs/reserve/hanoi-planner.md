# RESERVE — Constraint-Aware Recursive Execution Planner ("HANOI Planner")

Status: Reserve Only — Do Not Build Yet
Classification: Future vLOID / OROS Planning Architecture
Captured: 2026-07-16 (origin signal: Tower of Hanoi puzzle — recursive
decomposition under constraints as the essence of orchestration)

## Vision

A complex objective should be decomposed into the smallest VALID sequence
of actions while preserving all constraints at every step. The planner does
not merely generate task lists; it determines which actions are currently
admissible, which dependencies must move first, which intermediate states
are safe, which actions create irreversible conflicts, and the shortest
valid route to completion. Doctrine: **progress is not the number of
actions performed; progress is movement through valid states toward the
objective.**

## Core problem

Agent systems decompose goals yet still fail: wrong order, hidden
dependencies, constraint violations in intermediate steps, undoing prior
work, locally convenient actions that block the final objective, claiming
completion without proving the required state. Knowing the destination is
not having a safe path. The HANOI Planner is about the path.

## Planning model

Initial State -> Objective -> Constraints -> Dependencies -> Permitted
State Transitions -> Candidate Execution Paths -> Admissible Route ->
Verified Completion. Distinguish: desirable action vs technically possible
vs admissible vs THE CORRECT NEXT action.

**Recursive decomposition** (the Hanoi pattern): what must be true before
the main action; which obstacles move first and where they can safely
exist; the critical irreversible/high-value action; what must be restored
afterward. Move blocking dependencies -> execute critical transition ->
restore/advance dependencies -> verify final state. Unlike ordinary task
breakdown, every subtask stays bound to the GLOBAL constraint system.

## Constraint classes

Structural (B not before A; deployment not before tests; settlement not
before identity). Permission (authority, human approval, restricted data
classes, geography). Resource (budget, compute, time windows, rate limits,
staff). Safety (no destructive action without recovery state; no mutation
before snapshot; no deployment under unresolved critical risk). Doctrine
(consistency with LITMUS; failure cannot be hidden; evidence before
completion; the shortest route cannot override constitutional rules).

## State transition graph

Execution as movement through a state graph; each transition evaluated for
admissibility, reversibility, cost, risk, dependency impact, downstream
optionality, alignment with the objective. Reject actions useful locally
that create dead ends later.

**Minimum VALID path, not shortest path.** A path may be longer because it
includes security checks, approvals, backups, evidence generation,
compliance gates, rollback preparation, human review. Optimization is
subordinate to admissibility.

**Temporary State Doctrine.** Some objectives require temporarily moving
away from the final configuration (remove dependency -> perform blocked
operation -> reintroduce in correct order). Every temporary state carries:
reason, permitted duration, expected exit condition, restoration path,
receipt — preventing temporary exceptions from becoming permanent drift.

**Critical Transition Identification.** The "largest disc": central
dependency, irreversible migration, production deployment, settlement,
ownership transfer, data deletion, contract execution, infrastructure
cutover. Before it, all blocking conditions cleared; after it, the workflow
converges rather than reopening the problem.

## Receipts

**Planning receipt** (before execution): plan ID, objective, initial/target
state, constraints, dependencies, selected route, REJECTED routes, critical
transition, expected cost/duration, rollback conditions, required
approvals, completion criteria, planner + policy versions, signature —
explaining WHY that route was selected.

**Execution receipts** (per step): step ID, previous state, action,
actor/agent, authorization, constraint check, resulting state, evidence,
cost, latency, failures, next permitted actions, signature — a replayable
history, not a hidden chain of decisions.

**Failure = a state transition,** not an exception outside the model:
retry from current state, return to last safe state, alternate valid route,
human intervention, pause on missing dependency, or terminate because no
admissible route remains. Never improvise around a failed constraint
without a new plan or amendment receipt.

**Plan amendment.** Environment changes never silently mutate the plan:
original plan, changed condition, affected steps, new constraints,
recomputed route, reason, authorization — drift stays visible.

## Ecosystem placement

Inside vLOID execution admissibility: LITMUS (constitutional constraints),
VERITY (trust of proposed actors/tools/evidence), IAM (identity, authority,
scope), OROS (observe/adjudicate/execute/settle/learn coordination),
DRIFT_EXEC (deviation from approved sequence), DRIFT_SYS (environmental
invalidation), Intent Verifier (route still serves the original objective).
HelixAtlas visualizes initial/target states, blocked transitions, valid and
rejected routes, active step, critical transition, rollback branches,
receipts — the plan moving through the graph like a visible machine (the
wood-marble-machine mental model: one stable system, interchangeable paths,
visible motion, traceable state, replayable outcomes). HelixShield inspects
every proposed transition for security exposure, permission escalation,
unsafe irreversibility, supply-chain compromise, adversarial route
manipulation — efficient-but-insecure routes rejected.

## Use cases (future)

Software deployment (snapshot -> test -> scan -> approval -> deploy ->
verify -> receipt); infrastructure migration; money movement (identity ->
intent -> reserve -> rail -> settle -> receipt); incident response
(contain -> preserve evidence -> root cause -> remediate -> restore ->
verify); agent workflows (discover -> verify -> negotiate -> execute ->
validate -> settle); business operations (bottleneck -> dependencies ->
approval -> decision -> reconciliation).

## Quality metrics (research)

Valid completion rate, constraint violation rate, unnecessary steps,
rollback frequency, dead-end frequency, planning latency, execution cost,
route stability, amendment frequency, receipt completeness. Judge planners
by whether success was reached safely, efficiently, transparently — not
merely eventually.

## Non-goals

Not a task manager, to-do generator, chain-of-thought recorder,
unconstrained autonomous planner, OROS replacement, or consumer puzzle
product.

## Activation

Revisit when OROS requires multi-step autonomous planning across real
production workflows rather than isolated execution decisions.

---

## Extension 2026-08-29 — Execution Bounds, Deficit Classification & Verification Debt

Status: RESERVED — DO NOT BUILD. Architectural refinement of this reserve.

### Why this belongs here and not in its own file

Four submissions proposed a new runtime layer for bounded autonomous execution. Most of
what they described is already owned: `execution-economics.md` owns cost per verified
successful execution, `intelligence-resource-governance-layer.md` owns resource admission
and waste classification, `context-integrity.md` owns governed context state, and this
reserve owns permitted state transitions, constraint-preserving paths and temporary
states. What none of them carried was the **explicit form** of three things this planner
already implies. A bound belongs with the planner that must respect it, so they are named
here rather than in a new abstraction.

### Execution bounds are Resource-class constraints

The Resource constraint class above already covers "budget, compute, time windows, rate
limits, staff." Where autonomous execution is planned, that class is enumerated:

    STEP_BUDGET          how many transitions this plan may consume
    LOOP_BUDGET          how many times a cycle may repeat before replanning
    TIME_BUDGET          wall-clock ceiling for the plan or a segment
    TOOL_BUDGET          how many distinct capabilities may be exercised
    COST_BUDGET          defers to execution-economics for the unit
    VERIFICATION_BUDGET  how much verification the plan can afford to defer

These are **constraints, not targets.** A plan that consumes its full budget has not
performed well; it has consumed its full budget. Optimization remains subordinate to
admissibility, and the minimum VALID path may legitimately cost more than the shortest.

### TERMINATION_REASON

This reserve already enumerates the outcomes of a failed transition — retry from current
state, return to last safe state, alternate valid route, human intervention, pause on
missing dependency, or terminate because no admissible route remains. That enumeration is
now named, so a receipt can carry it:

    COMPLETED · RETRIED · REVERTED_TO_SAFE_STATE · REROUTED
    · HUMAN_INTERVENTION · PAUSED_ON_DEPENDENCY
    · TERMINATED_NO_ADMISSIBLE_ROUTE · TERMINATED_ON_BOUND

`TERMINATED_ON_BOUND` is distinct from failure. **A plan stopped by its own budget did not
fail; it was stopped.** Recording it as failure would corrupt both the planner's quality
metrics and the economics that consume them. The receipt records **which bound fired**.

`ESCALATION_THRESHOLD` names the point at which the planning receipt's existing *required
approvals* become mandatory rather than advisory.

### Deficit classification

When no admissible route remains, the useful question is not *did this fail* but **which
constraint could not be satisfied.** The five constraint classes above are already the
taxonomy; a deficit is a named blocked class:

    STRUCTURAL_DEFICIT   a prerequisite state cannot be reached
    PERMISSION_DEFICIT   required authority does not exist or was not granted
    RESOURCE_DEFICIT     a bound would be exceeded
    SAFETY_DEFICIT       no route satisfies the safety constraints
    DOCTRINE_DEFICIT     every remaining route violates a constitutional rule
    EVIDENCE_DEFICIT     admissibility cannot be established either way

The last is not one of the original five and is the reason this section exists: **a route
may be blocked because we do not know whether it is admissible**, which is distinct from
knowing it is not. That deficit terminates at UNKNOWN rather than DENY, and is resolved by
obtaining evidence rather than by finding another route.

This is deliberately not a waste taxonomy. `intelligence-resource-governance-layer.md`
classifies what was **spent wastefully**; this classifies what was **missing**.

### Verification debt

The Temporary State Doctrine above already requires every temporary state to carry a
reason, a permitted duration, an expected exit condition, a restoration path and a
receipt, *preventing temporary exceptions from becoming permanent drift.* **Deferred
verification is a temporary state of exactly that shape**, and is recorded as one:

    what was not verified · why deferral was admissible · what is assumed in the interim
    · what the verification would establish · permitted duration
    · exit condition (verification performed, or the assumption invalidated)
    · what must not proceed while the debt is outstanding

Debt is not a failure; unacknowledged debt is. A plan may legitimately defer verification
to reach a state where verification becomes possible. What it may not do is let the
deferral expire silently, which is the drift the Temporary State Doctrine already forbids.

**Verification debt outstanding is itself a constraint on the next transition.**

RESERVED — DO NOT BUILD.
