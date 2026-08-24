# RESERVE — Sovereign Intelligence Routing (SIR)

Status: RESERVED / INACTIVE. Evidence-driven activation; no hardware procurement implied.
Parent: new reserve that plugs into the vLOID ecosystem (not another governor).
Captured: 2026-08. Reconciliation note: the stack already uses Claude / GPT-4 / custom toolchains and SURVIVOR already governs whether execution is permissible, but no prior reserve formalizes model/inference-provider routing itself (local↔hosted failover + evidence-based selection). So this is genuinely new.

## Purpose

Prevent agents/applications from becoming structurally dependent on any single model, vendor, inference provider, or deployment environment. SIR provides a governed intelligence-routing abstraction that can select among local/self-hosted, hosted frontier, and specialized models without redesigning the consuming application around a vendor. Objective: replaceability, continuity, privacy optionality, and evidence-based model selection. NOT a directive to replace any current model, and NOT authorization to buy GPUs.

## Doctrine

Routing dimensions: experimentally-verified task capability, task requirements, data-sensitivity/privacy policy, context-window needs, latency, cost, availability, reliability, execution geography, governance constraints, historical execution evidence. A local model must not be preferred merely for being local; a frontier model must not be preferred merely for being generally more capable. Selection is workload-specific and evidence-bearing. Distinguish observed behavior from experimentally verified capability; marketing/benchmark claims and successful API connectivity are not verified capability.

Governance: SIR is subordinate to vLOID admissibility, SURVIVOR governance, identity/permission controls, API Connect capability/evidence declarations, and execution receipts — it is not a bypass. Continuity (intelligence-provider continuity, complementary to KONIGO's connectivity continuity): primary model unavailable/inadmissible/unsuitable → evaluate approved alternatives → select admissible model → execute → routing+execution receipt, so an agent survives provider failure/policy incompatibility without silently degrading governance. Local inference is one execution target and potential privacy boundary / continuity backstop / cost path / offline-edge path / specialized-workload env — with no assumption it outperforms hosted frontier models. Hardware is relevant only when measured workloads justify it.

## Relationship (cross-reference, do not duplicate)

vLOID / SURVIVOR (admissibility & governance authority) · IAM (identity/permissions) · API Connect (model/inference providers eventually represented as capability-bearing connectors — the likely evidence surface) · KONIGO Connect (connectivity continuity; do NOT turn KONIGO into an AI-model router) · Computable Accountability (routing receipt lineage).

## Non-goals

Do not build SIR now; do not alter production routing; no premature model abstraction; do not duplicate SURVIVOR/vLOID governance; do not turn KONIGO into a model router; no infrastructure purchase; no capability-equivalence claims without evidence.

## Activation

Revisit when multiple model providers are required in production; provider outages materially affect execution; sensitive workloads require local inference; local inference becomes economically advantageous at measured volume; a workload performs better on a specialized/local model; model-provider policy creates an execution constraint; agents require automatic intelligence-provider failover; or API Connect begins representing inference providers as capability-bearing connectors. Principle: depend on capabilities and admissibility, not vendor identity — models are execution resources behind the architecture, not the architecture. Until then: RESERVE ONLY.
