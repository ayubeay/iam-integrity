# RESERVE — Agent Metacognition & Calibration Layer (AMCL)

Status: RESERVED — cross-stack architectural primitive. NOT an active build.
Captured: 2026-08-27.

## Definition boundary

"Metacognition" here does **not** imply consciousness, sentience, subjective experience
or human-like introspection. It means computable self-monitoring of: what an agent knows,
how it knows it, what it inferred, what it assumed, what it has not measured, what remains
unknown, what evidence conflicts, whether reasoning is making progress, whether the
environment has changed, whether execution actually occurred, whether expected outcomes
were verified, and whether to continue, investigate, change strategy, defer, stop or
escalate.

**AGENT CONFIDENCE ≠ AGENT KNOWLEDGE OF WHY CONFIDENCE IS WARRANTED.**
AMCL preserves the latter.

## Core loop

    observation → evidence classification → reasoning → epistemic self-assessment
    → reasoning-progress assessment → admissibility assessment
    → prediction / expected postconditions → execution → postcondition observation
    → predicted ↔ observed comparison → calibration update
    → continue / change strategy / verify / defer / escalate / stop

## Epistemic state model

Do not collapse agent knowledge into a single confidence number.

    OBSERVED · INFERRED · RETRIEVED · ASSUMED · DERIVED · UNAVAILABLE
    NOT_MEASURED · CONTRADICTORY · STALE · UNKNOWN

Invariants:

    NOT_MEASURED         ≠ MEASURED_SAFE
    NOT_OBSERVED         ≠ OBSERVED_NEGATIVE
    EXECUTION_REQUESTED  ≠ EXECUTION_COMPLETED
    EXECUTION_COMPLETED  ≠ OUTCOME_VERIFIED
    HIGH_CONFIDENCE      ≠ HIGH_EVIDENCE_QUALITY

Uncertainty must propagate rather than silently becoming success.

## Prediction receipts

Consequential actions may produce a machine-readable prediction receipt *before*
execution: action intent, known and assumed preconditions, expected state change,
expected postconditions and invariants, known unknowns, verification method and deadline,
evidence dependencies, epistemic state, execution risk.

After execution, PREDICTED STATE ↔ OBSERVED STATE. The difference is longitudinal
calibration evidence.

## Calibration

Historical competence does not establish current admissibility. An action reliable at 94%
in normal conditions may verify at 61% under condition X. Calibration concerns what the
agent expected versus what actually happened.

## Epistemic yield

The degree to which additional reasoning, retrieval, tool use, model calls or computation
materially reduces decision-relevant uncertainty.

    Epistemic Yield ≈ useful uncertainty reduction / resources consumed

The formula is illustrative; the principle matters more. An agent should be able to
recognize: *"six retrieval calls, and uncertainty on the decision-critical variable has
not materially decreased."* Responses: CHANGE_STRATEGY · SEEK_NEW_SOURCE · RUN_MEASUREMENT
· REQUEST_MISSING_INFORMATION · DEFER · ESCALATE · STOP.

## Cognitive-loop detection

Detect computational thrashing — repeated search/retrieve/reason cycles without
uncertainty reduction; retry loops with trivial prompt changes against the same error;
plan/replan cycles without execution or new evidence.

The objective is **useful computation, not minimum computation.** A long reasoning
process is justified if it materially increases evidence quality or reduces consequential
uncertainty.

## Regime awareness

**PREVIOUS COMPETENCE ≠ CURRENT ADMISSIBILITY.** AMCL consumes DRIFT/regime signals
indicating that assumptions supporting previous behaviour may no longer hold — API
behaviour, schema, jurisdiction, counterparty, network, market regime, credential state,
latency, tool behaviour, data distribution, physical environment, policy, dependencies.

## Governance principle

An agent must never silently transform absence of evidence into evidence of success.
Preferred: *"I don't currently have sufficient evidence to establish X"* over
*"X is probably fine."*

## Relationship to existing canonical reserves

- VERITY — "can this evidence/source be trusted?"
- Information Admissibility Governor — "is the available evidence sufficient for this?"
- DRIFT / `regime-evidence-engine.md` — "has the environment changed?"
- OROS — "how should admissible execution be coordinated?"
- vLOID — "is this execution admissible?"
- `computable-accountability.md` — reconstructing the full chain.
- `inference-evidence-ledger.md` — challengeable evidence graph AMCL can reuse.
- `context-integrity.md` — context provenance feeding epistemic state.
- AMCL is a reflective control layer *across* these, not a replacement for any.

## Reserved research questions

Machine-readable epistemic-state schemas · calibration across heterogeneous agents ·
uncertainty propagation in multi-agent systems · **epistemic debt** · **assumption
dependency graphs** · prediction-receipt formats · contradiction detection ·
strategy-change triggers · epistemic-yield metrics · thrashing detection · agent-specific
vs system-wide calibration · human escalation thresholds · adversarial manipulation of
self-confidence · **false certainty from correlated sources** · calibration under regime
change · knowledge-state visualization.

## Non-goals

Not machine consciousness. Not artificial sentience. Not an emotion simulator.
**Not generic chain-of-thought storage.** Not a single confidence score. Not another LLM
wrapper. Not an excuse to expose private internal reasoning. Not authority to override
governance. Not a replacement for VERITY, DRIFT or admissibility controls.

Store only auditable artifacts — supplied and retrieved evidence, observations, explicitly
generated forecasts, assumptions, alternatives, confidence, policies, tool requests,
interventions, executions, outcomes, human corrections.

## Strategic principle

Agents increasingly answer *"what should I do?"* AMCL reserves the capability to answer
*"what do I actually know, how do I know it, what am I assuming, what am I missing, has
the environment changed, did my previous action actually work, how often have predictions
like this been correct, is further reasoning reducing uncertainty, and should I still be
the system making this decision?"*

RESERVED. DO NOT ACTIVATE. NO STANDALONE REPOSITORY.
