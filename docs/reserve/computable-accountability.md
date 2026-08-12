# RESERVE - Computable Accountability

**Status:** Reserved doctrine. DO NOT IMPLEMENT.
**Type:** Cross-cutting architectural property, not a service or product.

## Purpose
Prevent AI systems, metrics, rankings and autonomous agents from creating an accountability
vacuum in consequential decisions.

    A measurement may inform a decision without being allowed to masquerade as the
    reason for the decision.

AI may observe, score, rank, recommend, forecast, detect or flag. Those outputs must not
automatically become organisational responsibility.

## Do not create score -> action
    observation -> model or metric signal -> VERITY -> contextual exception evaluation
    -> LITMUS -> OROS / vLOID -> authorized human or delegated agent -> execution
    -> accountability receipt

The receipt preserves the causal and authorization chain, not merely that execution
occurred.

## The invariant
Delegating execution to an autonomous system does not eliminate organisational
responsibility. "The AI decided" must never become an architectural escape hatch.

If an agent executes, the system should answer: who authorized it, under what policy, what
scope was delegated, what evidence was considered, which model version participated, what
governance checks ran, why execution was admissible, what happened, and how it can be
challenged.

    SIGNAL != AUTHORITY
    CONFIDENCE != PERMISSION
    AUTOMATION != ABSENCE OF RESPONSIBILITY

## Contextual exceptions
Raw telemetry must not be read as intent, performance, misconduct or eligibility.

    activity_score = LOW  does NOT imply  worker_performance = LOW

Legitimate contexts - authorised leave, accommodations, role differences, incomplete
telemetry, measurement failure, changed responsibilities - depend on domain. Do not encode
employment or legal assumptions into the core governance layer; domain adapters supply
them.

## Where it matters most
Employment, worker eligibility and matching, scheduling, compensation, reputation, access,
financial execution, identity, safety, resource allocation, contractual rights. Not a
permanent list - the architecture should support risk classification.

For ShiftTrust youth-employment functions the requirements should be stricter: minors,
guardian permissions, hour restrictions, task safety, supervision, school calendars.

## Layers that must not collapse
    telemetry     is evidence
    a metric      is an interpretation
    a model output is a recommendation
    governance    determines admissibility
    authority     determines who may decide
    execution     produces the consequence
    receipts      preserve the chain

## Non-goals
No new microservice. Do not duplicate VERITY, LITMUS, IAM, OROS or DRIFT. Do not refactor
stable paths. Do not assume human approval is required for every execution, nor that
autonomous execution eliminates accountable authority. Receipt schemas stay proportional to
risk.

## When implementation is warranted
Inventory existing schemas, reuse primitives, identify the minimum missing accountability
fields, add risk-proportional extensions, preserve compatibility, add authority provenance,
add contextual-exception hooks at domain boundaries, and write tests demonstrating that a
model score alone cannot silently become an unauthorized consequential action.
