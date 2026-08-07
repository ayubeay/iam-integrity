# RESERVE - Context Integrity / Context Provenance Layer

**Status:** Reserved architecture. Not a build.
**Urgency:** HIGH relative to this batch. The failure mode is already observable in daily
work.

## Purpose
Treat the context supplied to an agent as governed execution state. Agent degradation must
not be attributed to the model by default. The architecture should distinguish:

    MODEL CHANGE  vs  CONTEXT CHANGE  vs  TOOL/EXECUTION CHANGE

The central question: what exactly did the model see, why did it see it, and how did that
differ from a previous execution?

Not another RAG or memory product.

## Why this is not speculative
The failure modes below are observable in this project's own sessions. A patch anchor that
no longer matches because the file changed is CONTEXT CHANGE misread as capability. A
recalled fact contradicted by a grep is CLASH. Carrying a stale assumption forward across
several turns is POISONING.

## Provenance per context item
source, source identity, version, creation and update timestamps, retrieval timestamp,
retrieval reason, authority, trust level, freshness, expiration policy, supersession,
conflicts, the execution that introduced it, and any compression or transformation history.

## Degradation taxonomy
    POISONING    incorrect or stale information keeps influencing later executions
    DISTRACTION  volume or low-value content obscures execution-critical information
    CONFUSION    irrelevant material affects reasoning or tool selection
    CLASH        two sources give incompatible claims, state or instructions

## Lifecycle
    ACTIVE -> COMPRESSIBLE -> SUPERSEDED -> EXPIRED

Expiry must not destroy the provenance needed to reproduce past executions.

## Authority, not recency
Conflicts do not resolve by timestamp. An explicit hierarchy, configurable and
domain-aware:

    production state          >  obsolete design document
    verified execution receipt >  unverified narrative note
    current explicit policy    >  superseded historical instruction

## Context receipt
run_id, context_item_ids, sources, versions, retrieval_reasons, authority, trust and
freshness, conflicts_detected, items excluded / compressed / superseded / expired, and a
context fingerprint.

The receipt should let someone reconstruct what was actually supplied - not what existed
somewhere in the system.

## Context Diff - the first-class diagnostic
Given a successful run and a degraded one, identify what changed: context added and
removed, source and version changes, authority changes, new conflicts, expired information
retained, relevant information omitted, compression changes, instruction changes, tool
results.

Turns "the agent got worse" into "behaviour change correlates with context_delta_7".

This is the same method the SURVIVOR scoring work used - a harness comparing variants
against a fixed population - applied to agent execution rather than token scoring.

## Mapping
VERITY source and claim trust. IAM identity and permissions on context. DRIFT changes
across executions. vLOID whether context state makes execution admissible. OROS
coordination and receipts. KONIGO Connect continuity when sources are unavailable.
HelixAtlas eventual visualisation of how information entered and influenced an execution.

Do not force these if the architecture suggests cleaner ownership.

## Smallest useful first step
A **context fingerprint on existing receipts**. Hash what was supplied, record it, change
nothing else. That single field makes Context Diff possible later and costs almost nothing
now. Everything else can wait.

## Do not build
Another vector database. Another generic memory service. Another telemetry system if the
existing one suffices.
