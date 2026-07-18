# RESERVE — Universal Execution Interface (UEI)

Status: RESERVED ONLY — DO NOT BUILD
Repository: iam-integrity
Captured: 2026-07-18

## Thesis

Current AI products optimize for conversation; future AI infrastructure
optimizes for execution. The UEI is a reusable interaction layer that
transforms user intent into governed execution across multiple products
while preserving identity, verification, policy compliance, receipts, and
accountability. Rather than building separate AI assistants per product,
**every product becomes an execution surface powered by the same execution
infrastructure.**

Current model: User -> Chat -> LLM -> Answer (the interaction ends with
text). Execution model: User -> Intent -> IAM -> VERITY -> vLOID -> OROS
-> Domain Engine -> HELIX -> Execution -> Receipt. **Conversation is only
the interface. Execution is the product.**

## Universal flow

Identity -> Intent Detection -> Context Collection -> Verification ->
Governance -> Execution Planning -> Execution -> Receipt -> Learning.

## Layered architecture

**L1 Identity (IAM)** — who is requesting execution.
**L2 Verification (VERITY)** — trust, permissions, history, reputation,
signatures, admissibility.
**L3 Governance (vLOID)** — constitutionally admissible? Outcomes: Allow /
Guard / Escalate / Defer / Reject.
**L4 Execution Planning (OROS)** — observe, understand, plan, adjudicate,
execute, learn.
**L5 Execution Rail (HELIX)** — carries out execution across domains. Not
limited to blockchain: the universal execution rail for every product
(swaps, collaborations, procurement routing, API generation, workflows).
**L6 Receipts** — always produced; denied execution gets receipts too.
Failure remains a first-class outcome.

## Domain engines (modular, below the interface)

SoundKeep, Commerce Sniper, Momentum Sniper, API Connect, Industry
Intelligence, Earthwise, FRONISS, KONIGO — every module plugs into the
same execution lifecycle. Examples of the shift from browsing to
execution: SoundKeep detects producer/genre/project and proposes verified
collaborators with one-tap introductions (no search); Commerce Sniper
turns supplier arrival into ranked procurement opportunities with trust
scores; API Connect turns developer arrival into generated SDKs and
executed test requests; Momentum Sniper needs no chat at all — wallet ->
intent -> risk -> eligibility -> JANUS -> HELIX -> receipt.

## One reusable UI pattern

"What are you trying to accomplish?" -> Understanding -> Recommended
Executions -> Approval -> Execute -> Receipt. The UI barely changes across
products; the execution engine never does.

## Relationship to existing architecture

UEI replaces nothing — it is the presentation and orchestration layer
above IAM/VERITY/vLOID/OROS/HELIX. Doctrine: **build one execution
infrastructure; expose many execution surfaces.** The strategic
distinction: many companies build AI chat interfaces; few build AI
execution infrastructure. Matured, this is an Execution Operating System,
not a collection of AI apps.

## Boundary doctrine (captured with this reserve)

Observation and execution sit on opposite sides of the decision boundary
and must never silently merge. The concrete case that prompted this note:
**Helius** (external Solana infrastructure), **HeliusSwapProvider**
(api-connect's swap-TELEMETRY adapter, Phase A d3981b7), and **HELIX**
(the governed execution rail) are three different things despite the
visual similarity of "HeliusSwap" and "Helix swap." The data layer answers
"what swap activity is occurring?"; it must never answer "should we trade
and how?" Keeping them separate guarantees: observation cannot silently
become execution; a provider failure cannot trigger or alter a trade;
consumers get metrics without wallet authority; HELIX can reject an
attractive signal on policy/exposure/slippage/identity grounds; the
metrics service stays reusable by systems that never trade. Naming note:
consider renaming to HeliusSwapMetricsProvider at next code touch to make
"observations, not execution" unmistakable. The intended full chain:
swap_metrics -> POI/Momentum interpretation -> candidate execution intent
-> vLOID + LITMUS + VERITY -> HELIX dry-run (quote, route, slippage,
balance) -> Shield Router authorization -> submit or deny -> signed
execution receipt. Phase B's fail-open shims must NOT execute through
HELIX; HELIX enters only when an accepted decision object is explicitly
connected to the execution path.

## Non-goals

Not a general chatbot, not another LLM wrapper, not a support widget, not
a standalone conversational-AI company. The purpose is governed execution.

## Activation condition

Only after: vLOID execution governance is mature; HELIX execution rails
are production-ready; at least two domain products share enough common
execution patterns to justify a unified interface.

## Strategic insight

Most AI products stop at generating responses. The UEI begins where
conversational AI ends: its purpose is not to answer questions but to
transform verified intent into governed execution with observable
receipts across every product in the ecosystem.
