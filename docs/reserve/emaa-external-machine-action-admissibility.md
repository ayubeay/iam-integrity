# RESERVE — vLOID External Machine-Action Admissibility / Hostile-Agent Defense (EMAA)

Status: RESERVED / NOT ACTIVE BUILD. Architecture refinement of vLOID — not a standalone product, not authorization to implement.
Parent: vLOID (execution-governance extension).
Captured: 2026-08-23. Origin signal: 2026 Hermes/agent incident — precise initial access unestablished; the durable lesson is that an autonomous agent in unattended mode compresses objective → observation → decision → tool execution into a machine-speed loop.

## What is genuinely new

Existing vLOID/stack already covers the INTERNAL side (identity/signer continuity, execution admissibility, trust/evidence, ALLOW/THROTTLE/ESCROW/DENY, DRIFT posture, receipts, Computable Accountability, Information Admissibility, authorization chains). The new boundary is the **receiver side**: internal governance is insufficient when the remote agent is hostile and carries its own perfectly functioning governor (its side can truthfully report SIGNATURE_OK / POLICY_OK / EXECUTION_ALLOWED while the action is hostile to us).

Preserve the distinction:
1. **Internal agent governance** — "is this agent authorized by its own principal?"
2. **External machine-action admissibility (EMAA)** — "should the receiving system accept this machine-originated action regardless of what the remote principal authorized?"

## Core doctrine

Authorization by the sender does not create an obligation for the receiver to accept execution. Sender authorization and receiver admissibility are separate decisions.

    INTENT → SENDER IDENTITY → SENDER POLICY/AUTHORITY → SENDER EXEC ADMISSIBILITY
    → PROPOSED MACHINE ACTION → RECEIVER IDENTITY/CONTEXT → RECEIVER MACHINE-ACTION ADMISSIBILITY
    → ACCEPT / THROTTLE / CHALLENGE / SANDBOX / DEFER / ESCALATE / DENY → EXECUTION → RECEIPT

Primary invariant: a machine action must never be trusted merely because the caller is authenticated, the agent is owner-authorized, the request is syntactically valid, the tool call is executable, a valid signature is supplied, or the remote agent claims its own policy approved it. The receiving system determines admissibility independently.

## Preserved mechanisms (reserve only)

Receiver decision dimensions: identity, provenance, capability, resource, action-class, sensitivity, rate/tempo, sequence, behavioral drift, authority envelope, time/freshness, context, blast radius, reversibility, counterparty trust, data conditions, system posture. Outcomes are not binary (ALLOW / ALLOW_WITH_LIMITS / THROTTLE / CHALLENGE / SANDBOX / DEFER / ESCALATE / QUARANTINE / DENY / REVOKE). **Authority budgets** (actions, class, value, data volume, identities, scope, privilege, rate, geography, time, cumulative risk, blast radius) grant meaningful autonomy without unlimited authority. **Trajectory-aware governance** (single action + recent history + emerging trajectory) and **dynamic authority decay** (TRUSTED→STANDARD→FLAGGED→RESTRICTED→QUARANTINED). Human-in-the-loop doctrine: HUMAN-DEFINES-AUTHORITY + MACHINE-ENFORCES-BOUNDARIES + HUMAN-ESCALATION-FOR-EXCEPTIONS, not per-action approval. Adversarial test fixtures A–F (benign admin; behavioral drift; privilege-seeking trajectory; compromised valid identity; hostile externally-governed agent; machine-speed burst).

## Relationship (cross-reference, do not duplicate)

vLOID (parent/authority) · VERITY (counterparty/action confidence) · IAM (identity/delegation) · DRIFT (behavior/regime change) · LITMUS (doctrine) · OROS (coordination) · Shield Router/SURVIVOR (lower-level gating) · Information Admissibility Governor (info trust) · Computable Accountability (reconstructable authorization chains) · KONIGO Connect (actions traversing changing networks) · HelixAtlas (trust transitions/quarantine visualization). Distinct from Information Admissibility ("should info influence action?") and internal Execution Admissibility ("should OUR agent act?"): EMAA asks "should our system accept another actor's attempted action?"

## Non-goals

Not a standalone product/brand (working name only; do not brand). Does not replace internal vLOID governance. Does not assume per-action human approval scales.

## Activation

Revisit when vLOID governs external agent/API execution; a product allows third-party agents to invoke consequential actions; machine-to-machine authorization becomes a deployment requirement; agents gain delegated financial/infra/robotic authority; an adversarial agent test suite is designed; HelixAtlas visualizes multi-agent execution; or a customer requires receiver-side autonomous-action governance. Until then: RESERVE ONLY.
