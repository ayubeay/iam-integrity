# RESERVE — Opportunity Intelligence & Evaluation Engine

Status: Reserved for future research and implementation. No execution
authority granted here.
Canonical home: iam-integrity/docs/reserve/opportunity-intelligence-evaluation-engine.md
Captured: 2026-07-25 (consolidated packet — promoted from the six-canonical
list flagged in staging/reserves-2026-07-24.md)

## Purpose

Provide an evidence-driven engine that helps autonomous systems and humans
decide whether an opportunity is worth pursuing **before** resources are
committed. It transforms scattered information into structured intelligence
and produces a governed recommendation with supporting evidence and a
decision receipt. Applies across suppliers, startups, acquisitions, APIs, AI
models, manufacturers, datasets, partnerships, products, investments,
hiring, vendors, and infrastructure. The objective is disciplined
decision-making, not prediction.

## Non-goals

Not a prediction or forecasting engine, not a recommendation feed, and not a
scraper that outputs a single opaque score. It does not replace human
judgment or discover opportunities in the first place — discovery is
upstream (Commerce Sniper for commercial imbalance; Operational Workflow
Discovery for broken workflows). Its job begins once a candidate exists and
ends at a transparent, auditable recommendation.

## Relationship to existing stack

This engine answers "should we pursue it?" where Commerce Sniper and the
Operational Workflow Discovery Engine answer "what exists / what is broken?"
Placement:

    Commerce Sniper / OWDE (discover) ->
    Opportunity Intelligence & Evaluation Engine (evaluate) ->
    vLOID (admissibility of recommended actions) ->
    HELIX / OROS (execute approved follow-ups) ->
    HelixAtlas (compare opportunities, track outcomes)

VERITY verifies claims, counterparties, and evidence confidence; IAM binds
evaluator identity and reviewer accountability; the decision receipt feeds
the Universal Execution Timeline and FEE's margin/outcome modeling.

## Activation condition

Do not build until VERITY claim-scoring and the receipt substrate are
operational and there is a real recurring decision stream (supplier
selection, model selection, or investment screening) to govern. Reserve is
not build.

## Workflow

    Opportunity -> Intelligence Collection -> Evidence Normalization ->
    Claim Verification -> Technical Evaluation -> Commercial Evaluation ->
    Competitive Landscape -> Risk Assessment -> Capital Requirements ->
    Execution Complexity -> Margin & Outcome Modeling -> Governance Review ->
    Recommendation -> Decision Receipt -> Outcome Tracking

## Intelligence sources (future)

Technical documentation, public websites, APIs, source repositories,
patents, supplier data, market data, pricing history, customer feedback,
community discussion, execution history, internal receipts, and historical
decisions — normalized before they influence a recommendation.

## Future research

Evidence-confidence scoring, opportunity similarity search, a comparable-
opportunity database, supplier-reliability scoring, investment-readiness
assessment, due-diligence automation, execution-cost forecasting, scenario
simulation, post-decision learning, and recommendation explanations.

## Doctrine

Every significant decision should be supported by structured evidence,
transparent reasoning, governed execution, and measurable outcomes. The
result is not merely a score — it is a transparent reasoning process that
can be audited and replayed.

## Cross references

Commerce Sniper (discovery of commercial imbalance) · Operational Workflow
Discovery Engine (discovery of broken workflows) · Capital Admissibility
Framework and Domain-Aware Capital Intelligence (capital-side evaluation) ·
Proof Before Promotion (evidence ladder the evaluation applies) · Flow
Economics Engine (margin/outcome primitives) · Meta-Architecture:
Observation to Strategic Moat (this engine is the "Opportunity Evaluation"
node).
