# vLOID — Operator Card (Workspace Identity Credential)

**Status:** RESERVED — visual concept approved (canonical render on file)
**Reserved:** July 18, 2026
**Distinct from:** docs/vloid-card-reserve.md (the payment/card-issuing layer). These must never be conflated: the Operator Card is IDENTITY; the payment card is SPEND.

## What it is (from the approved render)
A physical/digital credential for vLOID workspace operators — employees, developers, and potentially governed agents. Black metal, gold frame, "vLOID EXECUTION INFRASTRUCTURE" wordmark. Elements:
- Four module emblems: VERITY (blue), IAM (green, keyhole), OROS (orange), DRIFT (purple) — reading as capability/clearance badges for the subsystems an operator can touch
- Operator ID (format VX-#####)
- Public key (truncated hex on card face)
- QR code for verification

## Why it fits the stack (intersections)
- IAM: the Identity-Aware Engine gains a bearer artifact — operator identity as a first-class credential, not a login row
- VYRE: Ed25519 signing already exists in the stack; the card's public key is the natural anchor — QR scan > resolve operator > verify signatures/receipts issued under that key
- OROS: governance decisions can bind to Operator ID — who authorized what, receipted
- Reputation bands: operator standing (restricted/watch/standard/trusted) could surface through the credential

## Reserved capabilities (to ratify against the pending research text)
1. Operator registry: ID > public key > module clearances > standing
2. QR verification endpoint: scan > operator profile + validity + clearances (public-safe subset)
3. Signed-action provenance: actions in vLOID systems attributable to an operator key — the receipts doctrine applied to PEOPLE, completing the set (decisions, distribution, ownership, and now operators)
4. Physical issuance as culture: the card as belonging/credential for the workspace, à la early-crypto metal cards — brand weight included
5. DRIFT: appears on the card as a fourth module emblem — NOT yet documented anywhere in the doctrine. Must be defined before this reserve activates.

## Open items
- ADDENDUM PENDING: the accompanying research/conversation text failed to transfer (3 attempts, empty attachments). Append verbatim when it arrives; capabilities above are provisional until checked against it.
- Decide: operators = humans only, or governed agents too? (Agent operator cards would be very vLOID.)
- Canonical render: store the approved card image in the repo (docs/assets/) per the canonical-assets design rule.

## Activation gate
Requires IAM operator registry + at least one system consuming operator signatures. Do not build ahead of a consuming use case.
