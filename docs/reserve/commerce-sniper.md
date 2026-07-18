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
