# RESERVE - Execution Economics & Outcome-Aware Routing

**Status:** Reserved. Two related capabilities, deliberately separate.
**Urgency:** MEDIUM for Execution Economics. LOW for the router.

## Why they are separate
Execution Economics is useful with no routing at all - for audit, billing, budgeting,
benchmarking, procurement, and explaining why an agent cost what it did. The router is a
consumer of that intelligence, not the reason it exists.

Building the router first would embed the measurement inside an algorithm nobody can
inspect. Measure first.

---

## Reserve 1 - Execution Economics

Published model pricing does not reveal the cost of accomplishing a task. Two models at
similar token prices consume different token counts, make different numbers of tool calls,
retry differently, fail differently, and succeed at different rates.

    the meaningful unit is    COST PER VERIFIED SUCCESSFUL EXECUTION
    not                       COST PER TOKEN

### Cost envelope
Inference, reasoning and token consumption, cache usage, tool and API costs, compute,
measurable network cost, verification cost, retry cost, recovery and failover cost.

Derived: cost per attempt, per success, per VERIFIED success, retry-adjusted,
failure-adjusted, latency-adjusted, and where appropriate risk-adjusted.

### Outcome verification
A cheap execution that fails is not cheap. Success must not be inferred from a model
returning an answer. Connect outcomes to the existing receipt infrastructure so the system
distinguishes ATTEMPTED, COMPLETED, VERIFIED_SUCCESS, PARTIAL_SUCCESS, FAILED, RECOVERED,
DENIED.

### Task-specific, not universal
No provider has one economic score. Learn per workload class:

    task_class: contract_extraction
      MODEL_A  success 98.1%  median $0.031  retry-adjusted $0.033  p95 4.8s
      MODEL_B  success 91.7%  median $0.014  retry-adjusted $0.027
      MODEL_C  success 99.0%  median $0.089

The cheapest token price is not the cheapest successful execution.

---

## Reserve 2 - Outcome-Aware Execution Router

Not ROUTE_TO_CHEAPEST_MODEL. ROUTE_TO_BEST_ADMISSIBLE_EXECUTION_PATH.

A candidate path may span model, compute, tools, network path, verification and execution -
different models, providers, specialised versus general, local versus remote, tool
alternatives, regions, compute environments, connectivity paths, verification and retry
strategies. Modular; not every execution needs every dimension.

### Subordinate to governance
    intent -> IAM -> VERITY -> LITMUS -> admissible candidates
    -> execution economics -> router -> OROS -> verification -> receipt -> telemetry

**Economic optimisation must never override admissibility.** Placement is conceptual, not a
fixed topology.

### Objectives, policy-defined
MINIMIZE_COST, MINIMIZE_LATENCY, MAXIMIZE_SUCCESS_PROBABILITY, MAXIMIZE_QUALITY, BALANCED,
HIGH_ASSURANCE, or combinations under budget and risk constraints. A financial execution, an
infrastructure failover and a low-risk summarisation should not share one objective.

### DRIFT
Economics are not static. Model performance, provider pricing, tokeniser changes, latency
degradation, task-specific regressions, tool pricing, availability - a previously optimal
route can stop being optimal. Historical receipts feed continuously updated routing rather
than hard-coded rankings.

### Learning loop
    observe -> route -> execute -> verify -> receipt -> measure -> learn -> re-route

Every routed execution should answer: what path, why, what cost, did it succeed, was success
verified, were retries needed, would another route have done better?

---

## Doctrine
Admissibility before optimisation. Verified outcomes before superficial completion. Total
execution economics before token price. Task-specific evidence before universal rankings.
Dynamic routing before permanent provider assumptions. Receipts before efficiency claims.

## Grading
The measurement doctrine is usable now and prevents a specific mistake: choosing a provider
on token price. The router presumes several live providers and comparative volume, neither
of which exists yet.

---

## Extension 2026-08-29 — Bounded Execution and the Cost of Stopping

Status: Reserved. Architectural refinement of this reserve.

### Why this belongs here and not in its own file

`hanoi-planner.md` now enumerates execution bounds as Resource-class constraints and names
`TERMINATED_ON_BOUND` as a planning outcome. The bound belongs to the planner. Its
**economic consequence** belongs here, because this reserve already owns the unit and the
outcome vocabulary that consequence has to be expressed in. Adding a second cost accounting
elsewhere would defeat the reason this reserve exists.

### A stopped execution is not a failed one

The outcome vocabulary above distinguishes ATTEMPTED, COMPLETED, VERIFIED_SUCCESS,
PARTIAL_SUCCESS, FAILED, RECOVERED and DENIED. A budget-terminated execution is none of
these. It consumed cost, produced no verified success, and was ended by policy rather than
by error:

    TERMINATED_ON_BOUND    spend incurred, no verified success, stopped by its own limit

Two distinctions this preserves:

    TECHNICAL_FAILURE  ≠  GOVERNED_TERMINATION
    STOPPED            ≠  FAILED

Recording it as FAILED would attribute a defect to the execution path that belongs to the
bound. Recording it as DENIED would suggest admissibility refused it, which it did not.
**Both distortions corrupt task-class economics**, which is the evidence routing decisions
are meant to rest on. The receipt carries **which bound fired**, since a step ceiling, a
time ceiling and a cost ceiling are different findings about the same execution.

### Derived measures

The derived set above extends with: cost per bounded termination · the share of task-class
spend ending in bounded termination · and, where a bound was later raised, whether the
additional spend produced verified success. That last comparison is the only evidence that
tells a bound from a genuine limit — **a bound that never produces a completion when raised
was measuring the wrong thing.**

### The asymmetry worth preserving

A cheap execution that fails is not cheap; this reserve already says so. The corollary:
**an execution stopped early is not economical merely because it stopped.** Spend without
verified success is spend, and a system tuned to terminate frequently can show falling cost
per attempt while its cost per verified success rises.

RESERVED — measurement doctrine only. No router implementation authorized.
