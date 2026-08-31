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

---

## Extension 2026-08-29 — Boundary case: inventory demand is not a transaction

Status: RESERVED — DO NOT BUILD. NO CAPITAL DEPLOYMENT. Boundary refinement of this
reserve. **This extension narrows application; it authorizes nothing new.**

### Why this belongs here and not in its own file

This reserve already states what it is not — *"not general lending, not working-capital
provision, not a fund"* — and requires a **confirmed obligation** with a **verified
repayment source**. What it did not carry is a worked negative case showing how a
plausible-looking financing request fails that test. A boundary is easier to hold when
the canon records something that sits just outside it.

### The classification rule

A measurable cash-conversion cycle does not by itself establish a transaction-gap case.
**The transaction must exist independently of the financing request.**

    HISTORICAL DEMAND            ≠ CONFIRMED OBLIGATION
    APPROVED SALES CHANNEL       ≠ TRANSACTION
    EXPECTED SELL-THROUGH        ≠ IDENTIFIED REPAYMENT SOURCE
    INVENTORY REPLENISHMENT NEED ≠ TRANSACTION GAP
    PRODUCT-FAMILY SIMILARITY    ≠ DEMAND TRANSFER

The last one matters where a request bundles a proven line with adjacent untested ones:
demonstrated demand for one product is evidence about that product, not about a
neighbouring product that resembles it.

### The negative case

A founder sought production capital comprising a replenishment run of an existing product
plus two adjacent product runs. The available evidence was **founder-reported historical
sell-through, an intended replenishment, a reported multi-month cash-conversion cycle, and
approved vendor status with a corporate gifting channel.** There was **no disclosed
purchase order, no committed order, and no specific confirmed customer obligation**
underlying the request.

That is a working-capital and inventory-cycle problem. **It is not a transaction-gap case
under this reserve's own definition**, and financing it would require exactly the general
working-capital provision this reserve excludes. All reported figures remain **case signal,
unverified**; no financing assessment is implied.

### The useful question

Rather than asking whether the request can be financed, ask what would change its class:

    What additional evidence or event would convert this from general inventory
    financing into a specific transaction-gap case?

Illustrative only, and not a template for any actual party:

    verified purchase order → exact production obligation → measurable shortfall
    → identified buyer settlement → finance only the shortfall → fulfilment
    → settlement → financing terminates

The difference is not size or plausibility. It is whether an obligation exists that would
still exist if the financing request were withdrawn.

### Scope, held

This case may indicate an adjacent research problem in inventory-cycle financing. **Batch 3
does not create that architecture, and this reserve does not absorb it.** One encountered
case is not evidence of a missing primitive. Use-of-funds decomposition for ordinary
capital requests likewise remains unplaced — it is not authorized here, and relocating it
to another capital owner requires its own collision analysis.

The capital separation is unchanged:

    Capital Admissibility  should additional resources enter this execution at all?
    DACI                   who is competent to assess it?
    GCE                    should this admissible execution receive support now?
    TGFP                   is a verified transaction blocked only by a temporary
                           measurable gap with identifiable repayment?

**Classification precedes absorption.** A problem that resembles this primitive does not
belong to it.

RESERVED. NO CAPITAL DEPLOYMENT. DO NOT BUILD.
