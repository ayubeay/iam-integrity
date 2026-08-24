# RESERVE — Evidence-Backed Guardrail Lifecycle (EBGL)

Status: RESERVED / NOT ACTIVE BUILD. Governance/evidence primitive.
Parent: vLOID / SURVIVOR-Shield governance (narrowest home; reconcile with LITMUS doctrine layer).
Captured: 2026-08-22. Origin: autonomous-agent experiment where an agent created guardrails from failures, tracked whether they ever triggered, and hit the ambiguity that a zero-trigger guardrail may be healthy, unnecessary, unreachable, superseded, mis-wired, or broken.

## What is genuinely new

Three extracted mechanisms with no existing owner: (1) proof-of-life for dormant controls; (2) the activation-evidence vs proof-of-life distinction; (3) capability-leakage analysis as a disclosure-governance input distinct from intent.

## Core thesis

**Absence of observed failure ≠ evidence of control health.** Zero natural triggers are ambiguous (never occurred / deterred / intercepted upstream / obsolete / unreachable threshold / mis-wired detector / broken telemetry / superseded / broken). Controls require independent evidence they can still recognize and govern the conditions they claim to protect against.

## Subprimitives (reserve only)

**GPOL — Guardrail Proof-of-Life:** periodically challenge selected dormant/critical controls with isolated, bounded, non-destructive, explicitly-identified synthetic cases whose expected disposition is predetermined; compare EXPECTED vs ACTUAL; record the test environment in the receipt so simulation success is never represented as production proof. GPOL is not permission to inject dangerous actions into production.

**Activation evidence vs proof-of-life (keep orthogonal, do not conflate):** "Do we need this control?" (activation) is a different question from "Would it work if we needed it?" (proof-of-life). Concrete example from 2026-08 API Connect work: zero observed provider failovers gives no evidence that intelligent provider scoring is *needed*, yet the existing failover mechanism still needs evidence it would *function* if the primary failed. This distinction generalizes across every autonomous system.

**Governed guardrail lifecycle:** never let a system retire its own constraints from trigger statistics (a working control suppresses behavior → zero triggers → naive retirement → behavior returns). Use OBSERVATION → FAILURE/NEAR-MISS → ROOT-CAUSE → PROPOSED GUARDRAIL → COUNTEREXAMPLE/FAILURE TEST → SHADOW/CONTROLLED VALIDATION → GOVERNED ACTIVATION → MONITOR → GPOL → EFFECTIVENESS EVIDENCE → KEEP/REFINE/MERGE/RETIRE → RECEIPT. States: PROPOSED/TESTING/SHADOW/ACTIVE/DEGRADED/SUPERSEDED/RETIREMENT_CANDIDATE/RETIRED; no transition inferred solely from a trigger count. A guardrail becomes an evidence-bearing governed object, not an anonymous conditional.

**CLA — Capability Leakage Analysis:** NOT a second Intent Verifier. Intent verification asks "what is requested?"; CLA asks the downstream question "if we satisfy this request, what new operational capability does the recipient gain?" — the capability delta (thresholds, liquidation conditions, timing, routing logic, fallback topology, evasion conditions) can be hostile even when the request looks benign. CLA reconciles with the Information Admissibility Governor (intent is one input, capability delta another; neither alone determines admissibility).

Confluence/authority principle: external message ≠ authorized command (provenance → identity → authority → intent → info admissibility → capability/consequence → execution admissibility → action → receipt).

## Relationship (cross-reference, do not duplicate)

SURVIVOR/Shield Router (runtime controls that may produce GPOL evidence) · vLOID (admissibility boundary) · VERITY (confidence in probes/sources/results) · LITMUS (which controls may exist / what cannot be autonomously changed) · DRIFT (triggers mandatory revalidation on regime change) · Information Admissibility Governor (likely canonical owner of CLA's decision interface) · Intent Verifier (remains distinct) · Computable Accountability (observation→…→activation→execution→evaluation receipt) · API Connect (activation-vs-proof-of-life example; do NOT reopen it) · KONIGO (failover proof-of-life) · WIRE (robot-safety synthetic tests require strict sim/shadow).

## Relationship to IVCP (do not merge)

EBGL governs whether a control remains alive/reachable/necessary/effective. IVCP governs whether the *evidence* validating controls/capabilities is genuinely independent. They interact but are separate reserves.

## Non-goals

Not "an AI that writes more rules for itself"; zero triggers ≠ safe and ≠ useless; no silent self-deletion of safety constraints; synthetic probes must not become uncontrolled real attacks; simulation proof ≠ production proof; do not create another Intent Verifier; do not accumulate guardrails without overlap/contradiction analysis; do not reopen parked systems merely because this could apply.

## Activation

Revisit when a production autonomous system accumulates dynamic guardrails; critical controls show long zero-trigger periods with no other health signal; a safety-critical path needs runtime evidence beyond unit/integration tests; policy overlap/obsolescence creates governance debt; benign-looking disclosures are shown to create circumvention capability; or physical/financial autonomous execution could be harmed by stale controls. Until then: RESERVE ONLY.
