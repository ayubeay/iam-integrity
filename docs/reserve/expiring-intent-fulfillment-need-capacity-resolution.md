# RESERVE — Expiring Intent Fulfillment (EIF) + Need-Capacity Resolution (NCR)

Status: RESERVED / NOT ACTIVE BUILD. Two related but DISTINCT coordination/execution primitives — colocated here for the boundary, but not merged and neither is a new standalone product.
Parent: OROS coordination + vLOID admissibility + Intent Verifier (extend/compose, do not duplicate).
Captured: 2026-08-23.

## What is genuinely new

No existing reserve owns either the downstream last-mile intent→outcome router (EIF) or the upstream verified-need→qualified-capacity resolver (NCR). Both compose existing components rather than replacing them.

## EIF — Expiring Intent Fulfillment / Intent-to-Outcome Router

Problem: valid intent frequently exists but is lost in the friction between expressed willingness and completed execution (availability changes, prices move, urgency decays, handoffs fail, humans disengage). **EIF begins after intent exists** — it must NOT replace the Intent Verifier. Intent Verifier: "is this the action this human/agent intends to authorize?" EIF: "given valid intent, what remaining dependencies must be resolved to produce the authorized outcome before the opportunity expires?" Relationship: Intent Verifier → EIF → governed execution → outcome receipt. Loop: NEED → MATCH → INTENT → VERIFY INTENT → ADMISSIBILITY → DEPENDENCY RESOLUTION → EXECUTION → CONFIRMATION → OUTCOME → RECEIPT (dependencies: scheduling, identity, eligibility, permissions, inventory, transport, comms, payment, authorization, routing, temporary financing). Vertical-agnostic (blood donation is a forcing-function example, not the definition). Privacy: resolve with minimum-necessary execution objects (a blood-demand object exposes resource type/compatibility/urgency/region/window, not patient identity). Metric doctrine: measure the actual state transition (intent capture, fulfillment, time-to-fulfillment, verified outcome) — not conversations/messages/bookings. Keep the acronym EIF ("Expiring" captures that intent has a half-life); Intent-to-Outcome Router is its descriptive role. TGFP may resolve one specific EIF dependency (a small, temporary, measurable capital bottleneck) but is NOT merged into EIF.

## NCR — Need-Capacity Resolution

Problem: verified needs coexist with capable resources yet fulfillment fails because demand and capacity fragment across organizations/systems/locations/permissions/time. Purpose: convert verified unmet demand into admissible, qualified, available capacity while minimizing discovery-to-fulfillment friction and preserving consent/authority/evidence. Loop: NEED → VERIFY → DISCOVER CAPACITY → QUALIFY → MATCH → AUTHORIZE → ROUTE → EXECUTE → VERIFY FULFILLMENT → RECEIPT. Substrate-neutral (capacity may be human, org, machine, vehicle, robot, compute, capital, inventory, agent). Blood donation again a forcing function, not the product; final clinical eligibility stays with the authorized institution — NCR must not become unauthorized clinical decision-making.

## EIF ↔ NCR boundary (preserve explicitly)

NCR solves "who/what can satisfy this verified need?" (the matching gap, upstream). EIF solves "now that an admissible party expressed valid intent, how do we reach a completed outcome before it expires?" (the execution gap, downstream). They compose — verified need → NCR discovers/qualifies capacity → match → Intent Verifier → EIF resolves last-mile dependencies → vLOID-governed execution → outcome → receipt — but neither requires the other in every workflow.

## Relationship (cross-reference, do not duplicate)

VERITY (need/source/counterparty trust) · IAM (identity/permissions) · Intent Verifier (authentic intent — distinct from both) · vLOID (admissibility) · OROS (orchestration — do not duplicate) · DRIFT (changing demand/capacity/latency) · KONIGO Connect (continuity — distinct from responsibility continuity) · ShiftTrust/WIRE (specialized human/contributor capacity consumers-providers) · Commerce Sniper (commercial need/capacity) · TGFP (capital capacity, stricter doctrine) · Computable Accountability (observation→trust/admissibility→intent/authorization→execution→outcome).

## Non-goals

No universal monolith; preserve specialized systems and authority boundaries; neither replaces the Intent Verifier; do not duplicate OROS orchestration or KONIGO connectivity; optimize for verified state transitions, not message delivery.

## Activation

EIF: activate only after evidence shows recurring measurable loss between qualified/verified intent and completed execution that existing orchestration cannot close. NCR: activate only on evidence of recurring cases where a verified need and qualified capacity simultaneously exist but systematically fail to discover/match. At activation, define the smallest intent/responsibility and need/capacity state machines + receipt semantics first. Until then: RESERVE ONLY.
