# RESERVE — Independent Validation & Capability Promotion (IVCP)

Status: RESERVED / NOT ACTIVE BUILD. Governance/evidence primitive — not a "multiple AIs checking each other" product, not authorization to add LLM providers for diversity.
Parent: VERITY (expected major consumer/owner of the evidence semantics); reconcile with Computable Accountability and Backpack (Research OS).
Captured: 2026-08-22. Origin: experiment where agents from different vendors independently used/evaluated capabilities, challenged one another, and required multiple independent executions before promotion.

## What is genuinely new

No existing reserve owns evidence-independence reasoning or an evidence-graded capability-promotion gate. The deeper problem is not "have Claude check ChatGPT" but: **how do we know apparently-independent validators represent independent evidence, and what must a capability accumulate before a governed system trusts/promotes it?**

## Core thesis

**Agent count ≠ evidence independence.** Multiple agents can agree while sharing a failure (model lineage, prompts, context, retrieved docs, poisoned sources, tools, benchmarks, environment, orchestrator, evaluation criteria, upstream observations). Reason about evidence *ancestry and correlation*, not endorsement count. Canonical principle: **independent verification means independent failure opportunity, not different model branding.**

## Subprimitives (reserve only)

**IFD — Independent Failure Discovery:** validators independently attempt to falsify a claim/capability; where practical they do not see one another's conclusions first (reduces anchoring/conformity/copied error). Disagreement is useful evidence; consensus is not the goal.

**EIG — Evidence Independence Graph:** represent lineage/dependency of evidence (producer, model vendor/family, prompt lineage, context hash, source ids, retrieval path, toolchain, environment, method, parents, shared dependencies, result, contradictions). Relationship types incl. DERIVED_FROM / CORROBORATES / CONTRADICTS / SHARES_SOURCE_WITH / SHARES_MODEL_LINEAGE_WITH / REPRODUCES / FAILS_TO_REPRODUCE / INDEPENDENT_OF. Make correlated evidence visible.

**Epistemic Independence Assessment:** do not prematurely lock to one number; research vector/graph/tier/probabilistic forms across dimensions (model, source, context, method, tool, environment, temporal, oracle diversity). Two different-vendor models with independent prompts/sources/environments may corroborate far more than ten same-model/same-prompt/same-source agents.

**CPG — Capability Promotion Gate:** a capability earns authority through evidence (CANDIDATE → BUILDER TEST → INDEPENDENT VALIDATION → FAILURE DISCOVERY → CORRECTION → COUNTEREXAMPLE REPLAY → CROSS-METHOD/MODEL VALIDATION → SHADOW → LIMITED → TRUSTED → RUNTIME MONITORING → REVALIDATION). States: EXPERIMENTAL/VALIDATING/SHADOW/LIMITED/TRUSTED/DEGRADED/REVALIDATION_REQUIRED/SUSPENDED/RETIRED. Promotion is never based on code existing / tests passing / one benchmark / creator confidence / agent count / vendor diversity alone. **Independence scales with consequence** (validation burden scales with reversibility, uncertainty, exposure, consequence; do not hard-code monetary thresholds). **Negative evidence must survive** — failed attempts, who found them, environment, failed assumption, correction, replay result, unresolved disagreements are capability provenance, not garbage to erase.

## Relationship (cross-reference, do not duplicate)

VERITY (should distinguish evidence *quantity* from evidence *independence*; confidence must not rise linearly as agents repeat one lineage) · Computable Accountability (claim→evidence→lineage→evaluator→method→disagreement→correction→authorization→promotion→execution→outcome; prevents "multiple AIs approved it" laundering) · Backpack/Research OS (IVCP as a validation methodology for experiments) · WIRE (robot-skill certification: sim vs real evidence) · EBGL (complementary, not merged — EBGL = control lifecycle; IVCP = epistemic quality of validating evidence).

## Non-goals

Do not count agents as independent evidence; do not assume vendor diversity guarantees independence; no copied-reasoning-as-corroboration; do not hide failed evaluations after repair; do not optimize for consensus or treat disagreement as failure; no single-benchmark universal proof; the builder must not be sole validator; do not inflate confidence via duplicated evidence; do not require expensive multi-agent validation for trivial actions or let validation become an unbounded token-cost multiplier.

## Deeper principle

Everything may be connected, but connections can destroy independence: a mature system must know which observations are actually independent, which claims share ancestry, and which connections should *discount* rather than increase confidence.

## Activation

Revisit when capabilities are promoted from agent-generated evaluations; multiple models act as validators; WIRE begins skill certification; autonomous financial execution needs heterogeneous validation; VERITY must reason about correlated evidence; repeated cases show evaluators agreeing via a shared faulty source; capability/robot-skill marketplaces need trustworthy certification; or EBGL needs independent evidence for high-consequence guardrail health. Until then: RESERVE ONLY.
