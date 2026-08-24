# RESERVE — Commerce Sniper

Status: Reserved under the wider execution architecture — NOT an immediate
build
Captured: 2026-07-18

## What it is

A future intelligence and execution system for buying and selling goods
and services. Not another marketplace — the layer that detects what is
underpriced, what is in demand, where it can be sourced, where it can be
sold, whether the counterparty is trustworthy, whether the margin survives
logistics/fees/delays/risk, and whether the transaction should execute at
all. Core idea: **find commercial imbalance, verify it, execute it
safely.**

Relation to Momentum Sniper: momentum = price movement -> entry -> exit in
tradable assets; commerce = demand gap -> source -> verify -> buy -> route
-> sell -> settle in real goods and services. Not speculative flipping —
the edge is information asymmetry, regional price differences, supply
shortages, excess inventory, timing, logistics, buyer intent, service
capacity, verified execution.

## Initial categories (narrow first, learn one deeply)

Used electronics, surplus inventory, building materials, agricultural
products, freight capacity, equipment rental, software development
contracts, local professional services, business liquidation inventory,
wholesale-to-retail.

## Core opportunity object

Structured commercial thesis: item/service, source market, target market,
purchase cost, expected selling price, verified demand, platform/payment
fees, shipping/fulfillment, taxes/duties, time to sale, counterparty risk,
return/refund risk, net expected profit, confidence, recommended action.
Recommendations are graded, never just "buy": OBSERVE / CONTACT /
NEGOTIATE / RESERVE / BUY / LIST / ROUTE / HOLD / EXIT / REJECT.

## Engine (six parts)

**1. Opportunity Scanner** — marketplaces, suppliers, service platforms,
local listings, liquidation channels, demand signals: price spreads,
shortages, repeated buyer requests, negotiable slow sellers, urgent
demand, geographic mismatch, unused service capacity.
**2. Demand Verifier** — real buyer or attractive-looking gap? Completed
sales, buyer requests, search volume, listing velocity, repeat purchases,
waitlists, contract demand, regional shortages, historical time-to-sale.
Prevents buying something merely because it appears cheap.
**3. Counterparty & Listing Trust (VERITY)** — seller/buyer identity,
listing history, payment behavior, disputes, suspicious pricing,
duplicated images, manipulated reviews, delivery reliability, business
legitimacy. A cheap product from an unverifiable seller is not an
opportunity.
**4. Margin & Route Engine** — expected sale value − purchase − shipping −
storage − taxes − platform fees − payment fees − refunds − expected loss −
time cost = risk-adjusted profit; compares routes (local->local,
wholesale->online, online->regional fulfillment, service lead->verified
provider, bundling).
**5. Negotiation Agent** — strict boundaries (max purchase, min sale,
delivery windows, proof requirements, deposit limits, cancellation rules,
escalation points); early versions PREPARE messages, human approves before
sending or committing funds.
**6. Execution & Settlement (OROS)** — contact, proof, negotiate, reserve,
inspect, fulfill, receive payment, release funds, receipt.

## Services are first-class

Service spreads exist too (e.g. $8,000 website scope verified, routed to a
qualified team at $4,500, delivery tracked). NOT an exploitative
middleman: arrangements transparent, providers paid fairly, delivery
responsibility explicit. Categories: freight, cleaning, repairs, software,
design, installation, maintenance, agricultural processing, translation,
business sourcing.

## Architecture fit

Scanner -> demand+margin verification -> VERITY trust -> vLOID
admissibility (ALLOW/REVIEW/ESCROW/DENY) -> OROS coordination -> logistics
+ Universal Money Router -> signed commercial receipt -> HelixAtlas
opportunity map. IAM (identity, authorization, transaction limits), LITMUS
(commercial doctrine, prohibited behavior), DRIFT (market changes,
collapsing demand, fee changes, logistics disruption), Shield Router
(unauthorized payments, unsafe counterparties), KONIGO (communications and
transaction continuity).

## Commercial receipts

Every completed OR rejected opportunity: opportunity_id, source,
item_or_service, parties, prices, estimated vs realized margin,
verification evidence, risk score, shipping route, payment route,
approvals, execution status, failure_reason, timestamp. Failures recorded
(identity verification failed, demand disappeared, shipping cost rose,
buyer withdrew, inspection failed, margin below threshold, payment
disputed, delivery window exceeded).

## Doctrine (hard rules)

Verified demand before capital deployment. Net margin, not headline
spread. No autonomous purchase in early stages. No counterfeit, stolen,
prohibited, or unsafe goods. No deceptive resale or hidden defects. No
counterparty without adequate verification. Escrow above defined risk
thresholds. Every recommendation explains its evidence. Every execution
leaves a receipt. Capital preservation before transaction volume.

## Phase Zero (observation first, per house discipline)

Paper-commerce engine — buy nothing: detect opportunity -> record
expected costs and selling price -> monitor whether the item actually
sells -> compare predicted vs realized -> measure false opportunities.
Metrics: predicted vs realized margin, sell-through rate, time to sale,
demand accuracy, seller reliability, cancellation rate, logistics
variance, capital exposure. Purchasing enabled only after the engine
repeatedly proves its opportunities are real (Proof Before Promotion).

## Strategic positioning

Not "a marketplace for buying and selling" — **a verified commercial
opportunity and execution engine.** Marketplaces show listings; Commerce
Sniper determines which opportunity is real, which route is profitable,
which parties can be trusted, and whether execution is admissible.

---

## Inventory-to-Market Execution Primitive (IMEP) — refinement 2026-08-23

Status: RESERVED / NOT ACTIVE BUILD. A future execution primitive WITHIN
Commerce Sniper (not a separate product). Origin: a vertically specialized
coin-inventory listing/distribution tool; the abstraction is the general loop
from owned inventory → appropriate markets → reconciled inventory + settlement,
not coin collecting or marketplace automation.

**Gap it fills:** the operational layer between "we own/control an asset" and
"the asset has been represented in appropriate markets, sold, settled, and
reconciled." Commerce Sniper already reserves discovery/intelligence/routing/
verification; IMEP adds owned-inventory sell-side execution and reconciliation.

**Loop:** ASSET → EVIDENCE/PROVENANCE → INVENTORY STATE → VALUATION/MARKET
INTELLIGENCE → MARKET DISCOVERY → CHANNEL ELIGIBILITY → MARKET REPRESENTATION →
CHANNEL SELECTION → EXECUTION ADMISSIBILITY → DISTRIBUTION → INQUIRIES/OFFERS →
COUNTERPARTY ASSESSMENT → NEGOTIATION/TRANSACTION → SETTLEMENT → INVENTORY +
CHANNEL RECONCILIATION → OUTCOME MEASUREMENT → RECEIPT.

**Governing principle:** the canonical asset stays distinct from its market
representations. A listing is not the asset. One canonical asset may spawn many
channel-specific MARKET_REPRESENTATION objects differing in title/description/
price/currency/media/fees/terms/audience/duration — changing a representation
must never silently alter the underlying factual asset record.

**Multi-channel reconciliation:** solve the double-sale/stale-listing problem —
when one channel's transaction becomes authoritative, canonical inventory state
governs the rest (AVAILABLE → RESERVED/TRANSACTION_PENDING → SOLD →
RECONCILE_ALL_REPRESENTATIONS; remaining representations WITHDRAW_PENDING/
WITHDRAWN/EXPIRED/FAILED_TO_WITHDRAW/MANUAL_ACTION_REQUIRED). Requesting
withdrawal ≠ withdrawal occurred; reconcile channel state from evidence.

**Channel capability model:** do not hard-code around individual marketplaces;
channel adapters declare capabilities (LIST/UPDATE/WITHDRAW/READ_STATUS/
INGEST_INQUIRY/INGEST_OFFER/NEGOTIATE/TRANSACT/READ_SETTLEMENT/RECONCILE).
Absent capability → explicit human/manual boundary, not simulated automation.
Governed execution separates DISCOVERED/RECOMMENDED/PREPARED/AUTHORIZED/
DISTRIBUTED/TRANSACTION_PENDING/EXECUTED/SETTLED/RECONCILED — discovery ≠
authorization. Market intelligence optimizes policy-defined expected net
outcome, not headline asking price. Feedback loop compares predicted vs
realized (price, time-to-sale, fees, liquidity, counterparty risk).

**Relationship (cross-reference):** VERITY (listing/marketplace/counterparty
trust) · vLOID (commercial execution admissibility) · OROS (multi-step
execution) · IAM (who may list/modify/negotiate/approve/override) · Shield
Router/SURVIVOR (transaction boundaries) · Universal Money Router (settlement) ·
KONIGO Connect (continuity) · DRIFT (price/liquidity/demand regime change) ·
HelixAtlas (assets/markets/representations/reconciliation) · Computable
Accountability (evidence→recommendation→authorization→execution→settlement→
reconciliation) · Confluence Governance Principle (marketplace/settlement
confluence) · EIF/NCR (intent/need-capacity composition).

**Non-goals:** do not build now; do not clone the observed coin tool; no
marketplace scraping/automation in violation of terms; no autonomous listing/
selling merely because a channel supports it; asking price ≠ market value;
listing ≠ ownership/provenance; generated listing text must not overwrite
canonical asset evidence; offer accepted ≠ transaction complete (settlement +
reconciliation are separate states); no marketplace-specific logic in the core
where a capability adapter belongs.

**Activation:** when Commerce Sniper enters active development and needs
owned-inventory execution; a real business needs the same inventory across
multiple channels; Earthwise or another business needs governed multi-market
distribution; a resale/sourcing workflow needs automated channel selection +
reconciliation; multiple projects independently build inventory→listing→txn
logic; or a real user validates the workflow. At activation, begin with the
smallest canonical-asset + market-representation + reconciliation state machine
— not a generalized marketplace platform. RESERVE ONLY.
