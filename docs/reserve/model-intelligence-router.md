# RESERVE - Model Intelligence Router / Execution Model Market

**Status:** Reserved architecture. Not a build.
**Urgency:** MEDIUM-LOW as a router. MEDIUM for the measurement doctrine, which prevents a
future mistake.

## Purpose
Prevent dependence on one model or provider. Claude, GPT, Kimi, DeepSeek, open-weight and
local models, and models that do not exist yet, are replaceable execution resources beneath
the governance layer.

The router asks: which available intelligence is appropriate for THIS governed execution?
Highest benchmark score is not the same as best execution choice.

## Selection dimensions
Task capability, observed task-specific reliability, latency, input and output cost, total
execution cost, context-window requirements, context compatibility, tool compatibility,
structured-output reliability, privacy, jurisdiction and data residency, policy
restrictions, availability, historical failure rate, retry requirements, security posture.

## Task-class performance, not a leaderboard
Maintain empirical history per task class - coding and debugging, document extraction,
research, planning, classification, security analysis, long-context reasoning, tool
execution, structured transformation. Different models may win different classes.

## The measurement that matters
    COST PER SUCCESSFUL GOVERNED EXECUTION

Token price alone is insufficient. Incorporate retries, tool calls, latency, failed
executions, verification cost and fallback cost.

A cheap model needing three attempts can cost more than an expensive one that succeeds. An
expensive frontier model should not handle routine classification a local model does
reliably. And a provider that is cheapest today may not be - DeepSeek warned of a
significant API price increase shortly after the pricing comparisons circulated, which is
precisely why hard-coding architecture around today's cheapest provider is itself a risk.

## Selection receipt
run_id, task_class, candidate_models, capability requirements, policy and privacy and
jurisdiction constraints, historical task success, estimated cost and latency, context
requirements, chosen model, selection reason, fallback order, actual cost and latency,
result.

## Fallback
    PRIMARY -> SECONDARY -> LOCAL/OPEN -> DEFER

Subject to vLOID admissibility. Never silently route sensitive execution to a provider
violating privacy, jurisdiction, identity or security policy merely because another failed.

## Interaction with Context Integrity
    Context Integrity  what intelligence did the execution receive?
    Model Router       which intelligence should receive it?
    vLOID              is the execution admissible?
    OROS               how is the approved execution coordinated?
    receipts           what happened, what was considered, and why?

Never conclude model A beat model B without controlling for supplied context, versions,
tool availability, policies, instructions and environment. Model comparison needs
reproducible execution envelopes - which is why this reserve and Context Integrity belong
together.

## Grading
No pressure behind the router today - one model does most of the work and the routing
decision costs nothing yet. The measurement doctrine is the valuable half and is worth
holding to now, because the mistake it prevents is choosing a provider on token price.
