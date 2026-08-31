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

---

## Extension 2026-08-28 — Human-Mediated Execution / Decision Influence Accountability

Status: RESERVED — DO NOT BUILD. Architectural refinement of this reserve, not a separate
mechanism.
Scope: VERITY · Information Admissibility Governor · this reserve · vLOID · OROS · DRIFT ·
execution receipts.

### Why this belongs here and not in its own file

This document already establishes that AI may observe, score, rank, recommend, forecast,
detect or flag, that *a model output is a recommendation*, and that the chain runs through
an authorized human or delegated agent to execution. What it did not yet carry is
**influence provenance** — the evidence needed to reconstruct what the human actually saw
and how strongly it was presented. That is a deepening of this reserve's own invariant, so
it is recorded here rather than as a competing parent.

### Origin

Execution governance can correctly block an unauthorized *machine* action while missing
the pathway:

    information → model interpretation → recommendation → human belief
    → human-authorized action → physical / economic consequence

The final click, signature, transfer, approval, dismissal, purchase, deployment, treatment
decision or operational command may technically belong to a human while the
machine-generated recommendation materially shaped it.

The architecture must therefore distinguish **execution authority** from **decision
influence**.

This claims nothing about machine intent, manipulation, or moral agency. The engineering
fact is simpler: *machine-generated information can materially alter human decisions, and
human execution converts that influence into real-world consequences.*

### Two consequential paths, separately governable

    DIRECT:         agent interpretation → proposed action → execution admissibility
                    → machine execution → consequence

    HUMAN-MEDIATED: agent interpretation → consequential recommendation
                    → human receives / interprets → human authorization
                    → human execution → consequence

**The second path must not become an accountability escape hatch merely because a human
performed the final action.**

### Governing principle

**Human execution does not erase machine influence.**

Where a machine-generated recommendation materially contributes to a consequential
decision, preserve enough evidence to reconstruct: what information the system observed ·
where it originated · what evidence was admitted or rejected · what inference was made ·
what uncertainty existed · what recommendation was presented · how strongly it was
represented · which human received it · what authority that human independently possessed ·
what action occurred · what consequence followed · whether later evidence contradicted the
recommendation.

The purpose is not automatic blame assignment. It is **computable causal accountability
without responsibility laundering.**

### Consequential Recommendation

Not every model response requires governance. A **Consequential Recommendation** is a
machine-generated statement reasonably capable of causing material action in domains such
as money or capital, contracts, procurement, employment, healthcare, infrastructure,
cybersecurity, robotics, physical-world operations, legal or regulatory action, high-value
commerce, and safety-critical decisions.

*"Supplier X appears unreliable"* may be ordinary analytical output. *"Terminate Supplier X
immediately and move the $2M contract to Supplier Y"* is materially closer to execution and
should carry stronger evidentiary and uncertainty requirements.

**The architecture is consequence-aware rather than treating all generated text
identically.**

### Recommendation receipt — conceptual schema only

    recommendation_receipt:
      recommendation_id · timestamp · requesting_identity · receiving_identity
      decision_domain · source_set · source_independence · provenance_state
      admitted_evidence · rejected_evidence · inference · uncertainty · confidence
      known_unknowns · alternative_explanations · recommended_action
      expected_consequence · consequence_severity · authority_required
      human_decision · resulting_execution_id · later_outcome · later_recalculation

No implementation is authorized by this extension.

### Doctrine — fluency is not authority

A recommendation must never acquire evidentiary weight because it is confidently worded,
highly fluent, personalized, repeated, generated by a more capable model, presented through
an authoritative interface, or accompanied by synthetic citations.

    presentation quality ≠ evidence quality

### Doctrine — interface is not truth

    information side:  source     → provenance → admissibility → inference
    execution side:    capability → authority  → admissibility → execution

An information interface does not establish truth any more than an execution interface
establishes authority. Search, RAG, API, MCP, CLI, database retrieval, model memory and
human input are transport or access mechanisms; none independently establishes whether the
underlying evidence deserves trust.

### Source independence

Ten sources are not necessarily ten independent pieces of evidence. Distinguish *ten
reports* from *one original claim copied by nine downstream sources*. Where material
decisions depend on apparent consensus, provenance should attempt to identify shared
upstream origins. **False source multiplicity must not inflate confidence.** The falsifiable
form of this claim is specified in
`docs/research/experiments/EXP-GENEALOGY-001.md`.

### Recommendation escalation

As consequence severity rises, the evidence burden rises:

    LOW       → ordinary explanation
    MODERATE  → provenance + uncertainty
    HIGH      → corroboration + alternatives + explicit human review
    CRITICAL  → independent validation + admissibility gate + authority verification
                + auditable receipt

Exact thresholds remain future research.

### Worked example — finance

An AI recommends *"Transfer $50,000 to this account. Finance has approved it."* Even where
vLOID correctly prevents the AI from initiating the transfer, a human employee may possess
legitimate banking authority and execute it manually. The system should make
reconstructable: why the AI believed approval existed · what source asserted it · whether
that source was authenticated · whether the recommendation exceeded available evidence ·
what uncertainty was communicated · who acted · what authorization they held · what
happened.

The human action remains a human action. The machine recommendation remains part of the
causal record. **Neither erases the other.**

### Relationship to persuasion

This does not regulate ordinary persuasion, conversation or opinion. The concern begins
where recommendation + context + authority + consequence create a meaningful execution
pathway. **Do not create a universal persuasion governor.** The architecture stays
proportional to consequence.

### Anti-patterns

Do not classify every model response as a governed recommendation · assume AI intent where
none is evidenced · automatically blame the model when a human acts · automatically blame
the human when the system supplied misleading information · allow "human in the loop" to
become responsibility laundering · treat retrieved text as verified evidence · treat
citation count as source independence · suppress legitimate human autonomy · create
surveillance merely to reconstruct every ordinary decision · use this extension as
justification for behavioural control.

### Layer separation

    HFAI (docs/reserve/human-fairness-dignity-accountability-institute.md)
        → ethical / institutional doctrine and rights framework
    VERITY + Information Admissibility  → evidence integrity
    Computable Accountability (this)    → causal record
    vLOID                               → machine execution admissibility
    OROS / adapters                     → execution
    Receipts                            → reconstructable evidence

**An ethical institution must not quietly become a universal technical control plane.**
HFAI consumes this evidentiary substrate; it does not become the runtime governor.

### Research questions

At what consequence level does a recommendation become receipt-worthy? How are advice and
operational instruction distinguished? How is uncertainty represented without creating
alert fatigue? How is source independence estimated? How is recommendation causality
represented when multiple humans and systems contribute? When does repeated recommendation
become materially different from a single one? How does later contradictory evidence update
the historical record without rewriting it? How is human autonomy preserved while
accountability is maintained? How should high-risk recommendations expose alternative
explanations? Can outcome calibration determine which recommendation classes deserve more
or less trust over time?

### Activation

Remain reserve-only until one real workflow contains all four of: a machine-generated
consequential recommendation · a human with genuine execution authority · a material
physical, financial, legal, medical, employment, security or infrastructure consequence ·
and existing receipts that cannot reconstruct the recommendation-to-action chain.

### One-line doctrine

> When machine-generated information materially influences human-authorized execution,
> preserve the evidence, uncertainty, recommendation, authority and consequence chain
> without confusing influence with authority, or human involvement with absolution.

## Extension 2026-08-31 — Downstream Reconsideration on Upstream Invalidation

Status: RESERVED — DO NOT BUILD. Architectural refinement of this reserve, not a separate
product, module or implementation track.
Scope: VERITY · Information Admissibility Governor · vLOID · execution receipts · the
domain owners named below.

### Why this belongs here and not in its own file

This reserve already requires a receipt to preserve *whether later evidence contradicted
the recommendation*. That field records the fact. What it does not carry is the obligation
that follows from it: when a basis is later invalidated, something must establish what
already depended on it. The record lives here, so the obligation belongs here.

`invariant-precomputation.md` governs the adjacent case and is not extended by this
section. A reusable artifact carries dependency information and invalidation conditions,
and its warrant is checked **at the moment of reuse** — *where warrant cannot be
established, the correct state is recompute, not reuse.* That is invalidation checked on
the pull. A conclusion that has already authorized an action has no further reuse event at
which to be checked, and that gap is what this section names.

### The three boundaries

    UPSTREAM INVALIDATION      ≠  AUTOMATIC DOWNSTREAM INVALIDATION
    KNOWN MATERIAL DEPENDENCY  ≠  PERMISSION TO IGNORE LATER INVALIDATION
    RECONSIDERATION REQUIRED   ≠  REVERSAL REQUIRED

A downstream conclusion must not remain silently authoritative because the invalidation of
its basis was never propagated. It must equally not be reversed merely because an ancestor
moved: a descendant may have acquired independent support, the dependency may not have
been material, the consequence may be irreversible, or later evidence may sustain it on
other grounds.

### The obligation

    an upstream evidentiary, policy, authorization or execution basis is invalidated
    → material downstream dependencies are identified from preserved lineage
    → a reconsideration obligation is recorded for affected descendants
    → the consequence is routed to whoever owns that domain
    → what changed, why, and what followed is preserved in the receipt

Reconsideration does not rewrite historical state. The earlier conclusion, authorization
or execution remains recorded as what occurred under the evidence and authority available
at that time; later invalidation and any resulting reconsideration are preserved as
subsequent evidence and governance events.

**Reconsideration is horizontal; consequence is domain-owned.** This section establishes
only that affected descendants are identified and the obligation recorded. It does not
decide what happens to them. `iam-external-identity-risk-signals.md` decides for identity
risk, `evidence-backed-guardrail-lifecycle.md` for controls,
`independent-validation-capability-promotion.md` for promoted capabilities, and
`recalculation-doctrine.md` for objectives. No disposition vocabulary is defined here and
those domain sets are not merged into one.

### This is not a dependency engine

The obligation applies where a material dependency is **known, or reconstructable from
lineage this reserve already preserves.** It does not require discovering arbitrary causal
dependencies across a system, and it authorizes no traversal machinery, no graph service,
no propagation scoring and no dependency product.

    NO RECORDED DOWNSTREAM DEPENDENCY  ≠  NO DOWNSTREAM DEPENDENCY

The absence of a dependency in preserved lineage establishes only that the preserved
lineage records no such dependency. Whether none existed depends on the coverage of the
instrumentation that produced that lineage, which is a separate and currently unresolved
question.
**No dependency found is not a clearance.**

### Research questions

What makes a dependency material enough to oblige reconsideration? How is the coverage of
preserved lineage itself established, so that an absence in it can be interpreted at all?
Should a reconsideration obligation expire, and on what evidence? How does this
interact with irreversible consequences, where reconsideration cannot restore a prior
state? What distinguishes a descendant with genuinely independent support from one whose
support is correlated with the invalidated basis — a question effective evidence
multiplicity already governs in `independent-validation-capability-promotion.md` and which
this section does not restate?

RESERVED — DO NOT BUILD.
