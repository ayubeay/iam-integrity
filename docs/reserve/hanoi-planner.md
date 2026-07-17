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
