# RESERVE — Default-State Admissibility / Inaction Semantics

Status: RESERVED — cross-cutting governance primitive within vLOID. NOT an active build.
Captured: 2026-08-27.
Origin: a system presented users with a pre-selected factual state. 53 of 55 responses
retained it — not necessarily because users verified it, but because accepting the
default required zero effort. The control worked exactly as designed, yet the resulting
records could falsely imply deliberate human judgment.

## Core doctrine

**ABSENCE OF INTERVENTION MUST NOT AUTOMATICALLY BE INTERPRETED AS AFFIRMATIVE INTENT.**

More precisely: *defaults may optimize execution, but defaults must not counterfeit
intent.* The operational outcome of an inherited default may be identical to an explicitly
selected value; their evidentiary meaning is not.

    BAD:     status = VERIFIED
             verified_by = user

    BETTER:  status = VERIFIED
             selection_source = DEFAULT_INHERITED
             explicit_confirmation = false
             interaction_observed = false

## Decision provenance

    configured_default → displayed_state → interaction / no interaction
    → submitted_state → inferred_intent → admissibility determination
    → authorized action → execution receipt

Extends provenance beyond *"what value was stored?"* toward *"how did this value become
authoritative?"*

## Inaction as a first-class state

    EXPLICITLY_SELECTED · EXPLICITLY_CONFIRMED · DEFAULT_INHERITED · SYSTEM_INFERRED
    ROUTER_SELECTED · POLICY_INHERITED · NO_INTERACTION · UNKNOWN

    NO_INTERACTION    ≠ AFFIRMATIVE_CONFIRMATION
    DEFAULT_INHERITED ≠ EXPLICIT_INTENT

unless a governing policy explicitly establishes that equivalence and the equivalence
itself is admissible.

## Semantic promotion failure

    UNTOUCHED → DEFAULT → FACT

For consequential assertions the Information Admissibility Governor should prevent this.
Preferred: UNTOUCHED → UNKNOWN, or UNTOUCHED → DEFAULT_INHERITED with reduced evidentiary
confidence.

## Governance friction budget

This must not be read as "remove all defaults" — that creates another failure mode. Every
additional confirmation creates execution friction, and governance imposing excessive
friction is eventually bypassed, ignored, or mechanically approved.

    consequence × uncertainty × reversibility × provenance strength
    × authority required → intentionality requirement

    LOW       default may execute automatically
    MEDIUM    default may execute, provenance identifies it as inherited
    HIGH      explicit affirmative selection/confirmation required
    CRITICAL  explicit confirmation + authorization + stronger evidence

Scoring and thresholds are deliberately unspecified.

## Agent application

Agent systems contain many invisible defaults: model selection, reasoning configuration,
context carryover, memory retrieval, tool permissions, execution region, escalation
policy, retry behaviour, confidence thresholds, routing, fallback providers, data
retention, authorization inheritance.

    model = MODEL_X
    selection_source = ROUTER_DEFAULT
    human_selection = NONE

rather than a record that could later imply a human deliberately selected MODEL_X.

## Responsibility laundering

A final button press must not retroactively convert every upstream inherited
configuration into deliberate human intent.

## Two separate questions

*What happens if nobody does anything?*
*What is the system allowed to infer from the fact that nobody did anything?*

## Relationship to existing canonical reserves

VERITY (evidentiary strength of asserted states) · Information Admissibility Governor
(whether inherited states are admissible inputs) · OROS (explicitly selected vs
policy-derived vs router-derived vs inherited) · IAM (identity/authority of deliberate and
inherited decisions) · DRIFT (when historical defaults cease to be appropriate) ·
`computable-accountability.md` · `context-integrity.md` ·
`model-intelligence-router.md` and `intelligence-resource-governance-layer.md`
(router-selected defaults) · `evidence-governed-runtime.md`.

## Constraint

Do not activate a new implementation track. Do not create a standalone Default-State
product. Do not refactor production systems merely to accommodate this. At activation:
inspect existing architecture first, determine whether equivalent semantics already exist,
extend rather than duplicate, define the smallest useful state vocabulary, add
receipt/provenance semantics, test adversarial default/inaction cases, preserve backward
compatibility, produce implementation evidence.

Adversarial test definition: `docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md`.

RESERVED. NO ACTIVE BUILD. NO NEW REPOSITORY. NO NEW PRODUCT.
