# RESERVE — Evidence Lifecycle State & Provenance Envelope (ELSPE)

Status: RESERVED — research / future architecture. NOT an active build.
Captured: 2026-08-27.
Origin: examination of a multi-platform social-listening API for agents. Its normalized
+ optional raw-response design showed that normalization and provenance preservation can
coexist. Discussion with its developer then exposed a deeper problem: **source
observability is asymmetric** — deleted content remains observable on some platforms and
not others.

The reusable discovery is independent of that product: multi-source normalization is one
problem; governing the *temporal evidentiary status* of normalized information is a
different and deeper one.

## Core principle

Evidence has a lifecycle. A system must distinguish:

    WHAT WAS OBSERVED
    WHAT IS CURRENTLY OBSERVABLE
    WHAT IS CURRENTLY KNOWN ABOUT THE SOURCE
    WHETHER THE EVIDENCE IS ADMISSIBLE FOR A PARTICULAR DECISION

"This system observed X at T1" does not imply "the source still contains X now."
And "the system cannot retrieve X now" does not imply "X was deleted."

## Pipeline

    source → observation → raw observation artifact → normalized representation
    → provenance envelope → lifecycle state → admissibility assessment
    → decision / reasoning → execution → receipt

## Provenance envelope

evidence_id · source/platform · source object ID · source URL · provider · retrieval
method · observed_at · published_at · author/context · thread relationship · raw artifact
reference · normalization version · transformation history · content hash.

**published_at ≠ observed_at.** That distinction must survive normalization.

## Lifecycle states (provisional taxonomy)

    OBSERVED · REOBSERVED · UNCHANGED · MODIFIED · DELETED · UNAVAILABLE
    UNVERIFIABLE · CONFLICTED · SUPERSEDED · RETRACTED

## Critical semantic rule

**DELETED and UNAVAILABLE must not be conflated.**

Positive deletion evidence justifies DELETED. Failure to retrieve justifies only
UNAVAILABLE or UNVERIFIABLE unless further evidence establishes deletion.

    ABSENCE FROM A PROVIDER ≠ PROOF OF DELETION

Non-observation may result from deletion, indexing changes, API limits, permission or
privacy changes, suspension, outage, rate limiting, geographic restriction,
authentication state, retrieval failure, upstream data loss, platform changes, URL
mutation, moderation, or retention policy.

## Observability capability

Per platform / provider / endpoint / observation: can_observe_creation ·
can_reobserve_content · can_detect_edit · can_detect_deletion ·
can_retrieve_deleted_content · can_retrieve_revision_history · can_verify_author ·
can_preserve_thread_context.

The same retrieval outcome must produce different confidence depending on what the
provider is capable of reporting.

## State is not admissibility

Lifecycle state describes what is known. Admissibility determines whether that evidence
may be relied on for a particular claim.

A deleted post may remain admissible as evidence that *"this statement was publicly
observed at T1"* while being inadmissible for *"the author currently maintains this
position."* Admissibility is claim-sensitive and time-sensitive. The operative question:
**what exact claim is this evidence being used to support?**

## Temporal claim semantics

HISTORICAL OBSERVATION · CURRENT SOURCE · AUTHORSHIP · CURRENT-POSITION ·
DELETION · MODIFICATION claims each require different evidence.

## Historical evidence principle

A later source change must not silently rewrite the system's historical observation
record. If X was observed at T1 and acted upon at T2, and the source shows Y at T3, both
observations are preserved. This does not assert X was correct; it preserves the factual
history of the system's information state.

## Relationship to existing canonical reserves

- VERITY — source reliability, provenance quality, corroboration, integrity.
- Information Admissibility Governor — what the system is justified in doing.
- DRIFT / `regime-evidence-engine.md` — distinguishing *world changed* from *source
  changed* from *provider visibility changed* from *retrieval failed*.
- `context-integrity.md` — context as governed execution state.
- `evidence-commitment-and-anchoring.md` — anchoring what was observed.
- `computable-accountability.md` — receipts preserving evidence state at decision time.
- `provider-qualification-and-routing.md` — provider observability as a qualification
  dimension.

## Doctrine

*Preserve what was observed without pretending it remains true, current, available or
verifiable forever.*

*Absence of present observability must not be silently promoted into knowledge of what
happened to the source.*

## Activation

Reserve only. Future work should determine whether ELSPE belongs inside VERITY, inside
the Information Admissibility Governor, as shared evidence infrastructure beneath both,
or as a cross-stack protocol. Do not prematurely force module ownership.

RESERVED. DO NOT BUILD.
