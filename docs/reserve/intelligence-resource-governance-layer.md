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

---

## Extension 2026-08-29 — Continuation Admission & Self-Constructed Capabilities

Status: RESERVED — DO NOT BUILD. Architectural refinement of this reserve.

### Why this belongs here and not in its own file

Two mechanisms arrived attached to a proposed runtime layer. Both are admission decisions,
which this reserve owns, and neither is a placement decision — the scope boundary above
still holds, and **no `adaptive-inference-placement` is created or implied here.**

### Continuation admission

The policy vocabulary above already contains the stopping rule: *a task may stop when
expected remaining intelligence cost exceeds expected objective value.* What was not named
is that this is a **different decision from the per-call gate.**

    per-call admission    should this inference happen at all?           (NO_MODEL gate)
    continuation admission should this objective continue iterating?     (this section)

They differ in what they observe. The per-call gate looks at one request. Continuation
admission looks at the trajectory: progress made per iteration, whether recent iterations
changed the objective's state, escalations already spent, and expected remaining value.

    CONTINUE · CONTINUE_WITH_ESCALATION · REPLAN · DEFER · STOP_INSUFFICIENT_VALUE
    · STOP_NO_PROGRESS

**Repetition without state change is the signal.** An objective iterating without altering
its own state is consuming budget to stay still, and `DUPLICATE_REASONING` in the waste
classification above is what that looks like after the fact. Continuation admission is the
gate that prevents it prospectively.

The **bound** on iteration is not owned here — `hanoi-planner.md` owns `LOOP_BUDGET` as a
Resource-class constraint, and `execution-economics.md` owns the economic consequence of
stopping. This section owns only the admission decision at each continuation point.

### Self-constructed capabilities

Progressive capability admission above governs capabilities the system **exposes**:
MCP tools, APIs, plugins, database schemas, workflow actions, long-term memories. It
assumes capabilities pre-exist and are admitted into context.

An agent that **constructs** a capability mid-objective — a generated script, a composed
query, a derived helper, a chained tool wrapper — has created something that never passed
that gate.

**The governing invariant is not owned here.**

    AUTHORIZED_TO_CREATE  ≠  AUTHORIZED_TO_EXECUTE_CREATED_CAPABILITY

That is `ownership-proofs-vs-execution-rights.md`'s `possession ≠ permission`, applied to
an artifact the agent authored rather than one it acquired — and that reserve already
holds that ownership may persist while execution rights are withheld, conditional or
expired. `helixshield-execution-governance.md` states the autonomous-execution form:
capability, ownership and execution authority are separate layers, and a system may
possess the capability to act without the right to execute. `adaptive-execution-layer.md`
governs the construction itself — **"never allow silent behavioral mutation"** and
**"adaptation is a governed execution event, not an autonomous privilege"** — since a
capability built at runtime is exactly such a mutation.

**What this reserve owns is narrower: the admission consequence.**

    a capability the system exposed    → admitted
    a capability the agent constructed → admitted by the same gate, or not at all

Admission applies identically: the constructed capability declares what it does, what it
touches, what authority it requires, and what it is admissible to influence. It is
**bounded by the objective that created it** and does not survive into the next one. A
constructed capability that outlives its objective has become permanently resident, which
is the state this reserve's third principle exists to prevent.

Where the construction is consequential — it mutates state, spends, or reaches outside the
system — admissibility is vLOID's. Where it changes strategy, adaptation governance is the
Adaptive Execution Layer's. This section governs only whether the constructed capability
may enter the execution context.

RESERVED. DO NOT BUILD.
