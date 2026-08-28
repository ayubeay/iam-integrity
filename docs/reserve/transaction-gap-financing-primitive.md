# RESERVE — Transaction-Gap Financing Primitive (TGFP)

Status: RESERVED — research reserve. NOT an active build. **No capital deployment.**
Captured: 2026-08-27 (canonicalized in this batch because
`execution-jurisdiction-gap.md` ships depending on it as a sibling Execution Gap
Primitive; previously existed only in working history).

## Core idea

Some viable transactions fail not because they are unsound but because a **temporary,
measurable capital bottleneck** sits between a confirmed obligation and its fulfilment.

    viable transaction → temporary measurable CAPITAL bottleneck
    → finance only the execution gap

Not general lending. Not working-capital provision. Not a fund. The primitive finances
the *specific measured gap* in a *specific verified transaction*, and terminates when
that transaction settles.

## Required boundaries

- **Verify the transaction.** A claimed order, invoice, contract or obligation is a claim
  until evidence establishes it.
- **Verify the repayment source.** Financing is admissible only when the settlement path
  that repays it is itself identified and verifiable.
- **Finance only the measurable execution gap** — not the counterparty, not the business,
  not a runway.
- **No capital deployment from this reserve.** Research and architecture only.

## Loop

    opportunity → transaction verification → counterparty verification
    → obligation evidence → gap measurement → repayment-source verification
    → risk and reversibility assessment → admissibility → authorization
    → capital release → execution → settlement observation → repayment → receipt

## Relationship to the Execution Gap Primitives family

TGFP addresses the **capital** gap. `execution-jurisdiction-gap.md` addresses the
**institutional admissibility** gap. Both begin by diagnosing *which* primitive is
missing before asking whether the gap can legitimately be closed. Neither assumes the
gap should be closed.

## Relationship to existing canonical reserves

`capital-admissibility-framework.md` — whether capital/project execution is admissible.
`governed-capital-eligibility.md` — eligibility governance.
`domain-aware-capital-intelligence.md` — domain-specific capital evaluation.
`execution-liquidity-intelligence.md` — liquidity conditions.
`flow-economics-engine.md` — economic attribution.
VERITY, vLOID, OROS, Universal Money Router, receipts as elsewhere.

TGFP is distinct from all of these: it concerns the *gap-shaped* financing decision for a
single verified transaction, not the admissibility framework around capital generally.

## Activation

Revisit when repeated verified instances of gap-blocked viable transactions appear, when
verification of both transaction and repayment source is achievable from evidence rather
than assertion, and when a compliant capital structure exists. Legal, regulatory, tax and
licensing review precede any activation.

RESERVED. NO CAPITAL DEPLOYMENT. DO NOT BUILD.
