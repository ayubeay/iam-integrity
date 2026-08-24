# RESERVE — Intent Continuity / Intent Preservation Primitive (ICP)

Status: RESERVED / HYPOTHESIS TO TEST — NOT ACTIVE BUILD.
Parent: compose from existing OROS + IAM + vLOID + VERITY + Intent Verifier (test whether it needs to be a primitive at all before creating one).
Captured: 2026-08-23. Origin: workflow reconnaissance (website-form → SMS-follow-up service for local businesses). The abstraction is not real estate/SMS/Twilio/chatbots; it is that valuable external intent can enter a system successfully yet still be lost because responsibility, acknowledgment, routing, response, escalation, or resolution fails afterward.

## What is genuinely new (hypothesis)

A possible reusable governed primitive for **preserving valuable human/machine intent across operational handoffs until it reaches an explicit outcome**. No existing reserve owns this. It is explicitly a hypothesis to test, not a justified module — and it must not become a second Intent Verifier.

## Core hypothesis & invariants

Failure loop: INTENT ARRIVES → system technically receives it → no reliable ownership/routing → delay/spam/missed-notification/human-inaction → intent decays → customer leaves → opportunity lost. The system did not fail to *receive* the signal; it failed to *preserve the intent through execution*.

Governing invariant: **MESSAGE_SENT ≠ INTENT_HANDLED.** Likewise FORM_RECEIVED ≠ LEAD_HANDLED, EMAIL_DELIVERED ≠ REQUEST_RESOLVED, NOTIFICATION_SENT ≠ RESPONSIBILITY_ACCEPTED, AGENT_RESPONSE_GENERATED ≠ RESPONSE_DELIVERED ≠ COUNTERPARTY_ENGAGED. Preserve REQUESTED ≠ ATTEMPTED ≠ DELIVERED ≠ ACKNOWLEDGED ≠ RESOLVED. Transport acknowledgments (e.g. SMS_PROVIDER_ACCEPTED) are evidence, not terminal outcomes. Intent can be perishable — model intent value/relevance over time (not purely monetary; some intents are operational/legal/safety-relevant). Responsibility continuity: an unresolved intent must never become ownerless; receipts should make responsibility transitions explicit. Least-autonomy: ICP does not imply autonomous customer interaction by default (may be limited to ACKNOWLEDGE/ROUTE/REMIND/ESCALATE/REQUEST_HUMAN_RESPONSE); vLOID determines what automated action is admissible.

Conceptual states (illustrative, do not implement): INTENT_OBSERVED/ADMISSIBLE/REJECTED/ACCEPTED → ACKNOWLEDGED → OWNER_ASSIGNED → ROUTED → RESPONSE_PENDING/ATTEMPTED/DELIVERED → ENGAGED → ESCALATED/REROUTED → RESOLVED/CONVERTED/DECLINED/INVALID/EXPIRED/UNRESOLVED.

## Composition test (do this before creating a module)

Test whether the behavior composes from existing architecture: Information Admissibility Governor (is incoming intent/evidence admissible?) · VERITY (authenticity/spam/fraud/source reliability) · IAM (responsible identities/permissions) · OROS (acknowledgment/assignment/routing/follow-up/escalation) · vLOID (which automated actions are admissible) · KONIGO Connect (infra/comms continuity — distinct from responsibility continuity; do NOT merge) · DRIFT (abnormal latency/conversion/routing failure) · Shield Router/SURVIVOR (boundaries) · Computable Accountability (intent→evidence→decision→responsibility→execution→outcome) · Intent Verifier (distinct — do not duplicate). If OROS+IAM+vLOID+VERITY already express the requirement cleanly, ICP remains a doctrine/pattern rather than a module.

## Non-goals

Do not build now; no "AI realtor follow-up" product; no new CRM/chatbot; do not assume SMS; message delivery ≠ resolution; AI-generated responses must not create commitments without authorization; do not auto-escalate everything; not every intent deserves indefinite preservation; do not duplicate OROS/KONIGO/Intent Verifier; **do not create a new module merely because the concept has a name.**

## Activation

Revisit when a real workflow repeatedly loses valuable intent after successful capture; OROS needs explicit responsibility-continuity semantics; multiple projects independently implement lead/request ownership+escalation+resolution tracking; SoundKeep/Earthwise/Commerce Sniper/WIRE/ShiftTrust hit the same intent-decay problem; real users validate the pain; or testing shows existing OROS/IAM/vLOID cannot express it cleanly. Promote to architecture only if evidence shows existing components cannot express it and a genuine operational need exists. Until then: RESERVE ONLY.
