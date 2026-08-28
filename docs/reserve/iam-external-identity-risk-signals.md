# RESERVE — IAM External Identity-Risk Signal Ingestion

Status: RESERVED — architectural refinement of IAM. NOT an active build.
Parent: IAM → VERITY → vLOID execution admissibility.
Captured: 2026-08-27. Not a standalone identity-protection product.

## Purpose

Extend IAM beyond static authentication state so externally observed evidence of
identity compromise, credential exposure, account takeover, device compromise or fraud
can dynamically influence what an identity is permitted to execute.

    Authentication establishes who is presenting an identity.
    Identity-risk intelligence helps determine how much execution authority
    that identity should currently possess.

**External exposure must not automatically equal compromise.** Every signal is evaluated
for provenance, freshness, severity, corroboration, affected credential and confidence
before authorization posture changes.

## Core loop

    external risk signal → normalize → provenance verification → evidence assessment
    → VERITY confidence → identity-risk state → authorization recalculation
    → ALLOW / STEP-UP / THROTTLE / ESCROW / DENY → remediation → re-verification
    → authority restoration → receipt

## Signal shape

subject identity · source · evidence type · observed_at · affected asset/credential ·
severity · confidence · freshness/TTL · corroboration · scope · recommended posture.

## Graduated response

    NORMAL → WATCH → ELEVATED → RESTRICTED → COMPROMISED → RECOVERY

Avoid binary SAFE/COMPROMISED. A historical email breach may produce WATCH with no
interruption; fresh credential exposure may require STEP-UP before sensitive execution;
confirmed session compromise may invalidate credentials and DENY high-risk actions; an
uncertain but serious signal may THROTTLE or ESCROW pending verification.

## Blast-radius awareness

Risk propagates along the identity's actual authority graph. If an affected identity
controls API keys → agents → wallets → production systems → financial permissions, the
governor determines which dependent capabilities require restriction rather than
indiscriminately shutting everything down.

## Temporal admissibility

Identity trust is time-dependent. A permission admissible at T0 may become inadmissible
at T1 after new evidence arrives. Receipts preserve identity state at execution, known
evidence at execution, the authorization decision, later evidence, and whether
retrospective investigation is required — distinguishing execution finality from
later-discovered identity admissibility failure.

## Recovery is first-class

    COMPROMISED → REMEDIATING → REVERIFYING → RESTORED

Restoration may require credential rotation, session revocation, device re-attestation,
step-up verification, administrator review, source confirmation or cooling periods.
Restoration itself produces a receipt.

## Strategic boundary

Do not turn vLOID into a dark-web-monitoring or identity-protection company. Specialized
providers perform exposure discovery. The differentiated layer is: *given changing
evidence about an identity, determine what that identity remains admissible to execute —
and prove why.* Identity-security providers become signal suppliers, not competitors.

## Relationship to existing canonical reserves

- IAM owns identity state and authority; VERITY evaluates signal/evidence trust; vLOID
  determines execution admissibility; OROS applies consequences; receipts preserve the
  chain (`computable-accountability.md`).
- `emaa-external-machine-action-admissibility.md` — hostile external machine actions.
- `api-trust-exposure-model.md` — trust governance for external surfaces.
- `evidence-lifecycle-state-provenance-envelope.md` — freshness and provenance of signals.

## Activation

Revisit when IAM governs consequential production execution, when external
identity/security feeds become available, or when a use case requires authorization to
respond dynamically to changing identity risk.

RESERVED. DO NOT BUILD.
