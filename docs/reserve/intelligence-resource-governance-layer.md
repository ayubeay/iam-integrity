# RESERVE — Intelligence Resource Governance Layer (IRGL)

Status: RESERVED — future architecture. NOT an active build.
Captured: 2026-08-27.

## Scope boundary — read this first

Inference **placement** is already canonically owned. IRGL deliberately does not restate
or replace it:

- `model-intelligence-router.md` — model/provider selection, provider independence,
  measurement doctrine.
- `sovereign-intelligence-routing.md` — sovereignty and jurisdiction constraints on where
  inference may execute.
- `execution-economics.md` — cost accounting and outcome-aware routing.
- `context-integrity.md` — context as governed execution state and context provenance.

**No `adaptive-inference-placement` exists, and none should be created** — that
responsibility is owned three ways above. IRGL exists only for the control surfaces none
of them own.

## What IRGL owns

1. **Deterministic-solvability gate (NO_MODEL).** The first admissibility decision is
   whether an LLM is needed at all. A deterministic solution is preferred when code, SQL,
   regex, an API, a rules engine or a verified cached result can reliably solve the
   objective.

       NO_MODEL · LOCAL_MODEL · REMOTE_MODEL · FRONTIER_MODEL
       DETERMINISTIC_TOOL · CACHED_RESULT · ESCALATE · DEFER · DENY

2. **Memory admission.** Separate retrieval from admission. Retrieval may surface many
   potentially relevant memories; admission decides which subset actually enters the
   execution context. Prefer memories useful for a future execution, bounded by entity,
   condition and time. Avoid dumping project history into context.

3. **Progressive capability / tool admission.** Expose only the capabilities the current
   objective requires instead of permanently loading the full tool universe:

       objective → capability family → tool discovery → schema retrieval → execution

   **Capabilities should be addressable, not permanently resident in context.** Applies to
   MCP tools, APIs, plugins, database schemas, workflow actions and long-term memories.

4. **Total-objective-cost accounting.** Optimize for **expected total cost per
   successfully completed objective** — not token price, model price, or cost per
   inference in isolation.

5. **Waste classification.**

       UNNECESSARY_INFERENCE · CONTEXT_BLOAT · MEMORY_OVERFETCH · TOOL_OVEREXPOSURE
       DUPLICATE_REASONING · RETRY_WASTE · CACHE_MISS_REGRESSION
       MODEL_OVERKILL · MODEL_UNDERPOWERED · FAILED_EXECUTION_SPEND

## Sequence

    objective → deterministic-solvability check → context admission → memory admission
    → capability/tool admission → model capability requirement
    → privacy/sovereignty constraints → execution-location candidates
    → predicted total objective cost → execution → verification
    → escalation/fallback → receipt

Context admission defers to `context-integrity.md`; placement and sovereignty defer to the
router and SIR; cost accounting defers to `execution-economics.md` and extends it only
with per-objective totals.

## Measurement

objective_id · task_class · model_calls · input/output tokens · cache reads/writes ·
retrieved vs **admitted** context · retrieved vs **admitted** memories ·
tool_schemas_loaded · tool_calls · retries · escalations · execution location · latency ·
verification result · human intervention · final quality · total_objective_cost.

## Two corrective principles

**A cheaper model is not automatically more efficient.** If it requires more turns,
retries, human intervention, tool calls or eventual escalation, the stronger model may
have the lower effective cost per successful objective.

**A larger context window is not permission to load more information.** Context is an
execution resource and should be admitted only when useful.

## Policy vocabulary

    ALLOW · ALLOW_WITH_REDACTION · ALLOW_LOCAL_ONLY · ALLOW_UP_TO_COST_LIMIT
    ALLOW_ESCALATION_AFTER_FAILURE · REQUIRE_VERIFICATION · DEFER
    DENY_EXTERNAL_EXECUTION

A routine classification may use a small model; a deterministic validator may bypass
models entirely; a production incident may justify frontier reasoning; restricted data may
be local-only; an agent may get one cheap attempt before escalation; a task may stop when
expected remaining intelligence cost exceeds expected objective value.

## Intelligence Consumption Receipt

    objective → information admitted → memory admitted → tools admitted
    → model/location selected → calls performed → cache behaviour → verification
    → escalations → result → total cost → governing policy

Computable accountability for machine intelligence consumption.

## Relationship to existing canonical reserves

The four scope-boundary reserves above, plus vLOID (admissibility), KONIGO Connect
(connectivity/continuity state), `../RESERVE-VKOS.md` (progressive knowledge and
capability exposure), `computable-accountability.md`, and
`agent-metacognition-calibration-layer.md` (epistemic yield is the reasoning-side sibling
of resource governance).

## Doctrine

*Use the minimum sufficient intelligence envelope that reliably completes the objective,
escalate only when evidence justifies it, and preserve a receipt explaining every resource
consumed.*

RESERVED. DO NOT BUILD.
