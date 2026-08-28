# RESERVE — Loyalty Value Routing & Stranded Value Intelligence

Status: RESERVED — research only. No custody, no point purchasing, no point resale, no
automated transfer execution, no program-rule circumvention, no capital deployment.
Captured: 2026-08-28.
Provenance: recovered direction. The concept was believed to have been discussed
previously; the Batch 2 collision scan across the canon returned four hits, all
incidental (a SoundKeep lyrics file, a staging ledger, tokenized securities, an economic
ratchet). No materially equivalent reserve existed, so this is canonicalized as a
**recovered/rediscovered** direction rather than a new invention.

## Purpose

An intelligence and execution-governance layer for identifying, valuing and **lawfully**
optimizing fragmented loyalty and reward entitlements: hotel points, airline miles,
credit-card rewards, transferable reward currencies, free-night certificates, upgrade
instruments, companion benefits, status-linked entitlements, travel credits, expiring
promotional entitlements and other issuer-controlled loyalty value.

**The system must not treat loyalty points as ordinary money, and must never assume that
technical transferability implies contractual permission to sell, barter, broker or
transfer.**

## Core problem

Consumers and organizations accumulate economically useful entitlements across many
incompatible ecosystems. Nominal balances do not reveal actual utility: 100,000 points
has no single intrinsic economic value. Realizable value depends on issuing program,
redemption rules, destination, dates, inventory, transfer partners, transfer ratios,
bonuses, expiration, fees, taxes, account standing and user objective.

## Core loop

    entitlement discovery → rights and restrictions → transformation routes
    → availability → effective value → user objective
    → apply admissibility constraints → recommend permissible route
    → user authorization → execute where permitted → preserve receipt
    → measure realized value

## Stranded Loyalty Value (SLV)

**Stranded Loyalty Value** is legitimately acquired loyalty or reward utility unlikely to
be efficiently realized because of fragmentation, expiration, poor redemption knowledge,
incompatible programs, inaccessible inventory, small residual balances, forgotten
certificates, transfer restrictions, or failure to identify a superior permissible route.

Research objective: maximize realized user utility from stranded value subject to program
rules, user intent, availability, time, fees, applicable tax and legal constraints,
transfer restrictions, account permissions and uncertainty.

**Do not equate SLV with cash sitting in an account. SLV is conditional economic
utility.**

## Loyalty value graph

Model each entitlement as a constrained graph rather than a currency balance. Candidate
edge and node attributes — examples, not a frozen schema:

    CAN_CONVERT_TO_PARTNER · CAN_BOOK_FOR_THIRD_PARTY · CAN_SELL_FOR_CASH
    TRANSFER_LIMIT · EXPIRATION_RULE · ACCOUNT_AGE_REQUIREMENT · REVERSIBILITY
    PROGRAM_TERMS_VERSION · RULE_EVIDENCE_SOURCE · RULE_LAST_VERIFIED · RULE_CONFIDENCE

## Program admissibility registry

A machine-readable registry representing program-specific permissions must distinguish:

    PERMITTED · PROHIBITED · CONDITIONALLY_PERMITTED · UNKNOWN · STALE_EVIDENCE

**`UNKNOWN` must never silently become `PERMITTED`.**

## Critical doctrine

**A technically executable value route is not necessarily an admissible value route.**

A user can technically transfer points to another account. That does not establish that
the user may sell those points for cash. Admissibility derives from issuer rules and
applicable constraints, not from technical possibility.

This makes the problem directly relevant to the broader vLOID execution-admissibility
doctrine **without requiring loyalty systems to become part of vLOID itself.**

## User intent

Optimization is objective-dependent. The optimizer must not assume that the
mathematically highest nominal redemption is the outcome the user wants. Effective value
depends on what the person is actually trying to achieve.

    ENTITLEMENT → RIGHTS → RESTRICTIONS → TRANSFORMATION ROUTES → AVAILABILITY
    → EFFECTIVE VALUE → USER OBJECTIVE → ADMISSIBLE EXECUTION

Maintained accurately across many programs, this becomes reusable infrastructure rather
than a travel-points recommendation application.

Do not fabricate certainty when award inventory or program rules can change between
observation and execution. Evidence freshness is a first-class constraint.

## Scope boundary

Let loyalty and travel be the bounded research domain. The deeper primitive —
**conditional-value routing under changing rules** — may eventually extend to store
credits, vouchers, promotional entitlements, certificates, benefits and rebates. **Do not
extract a general Conditional Value Router yet.** If experiments later show the same
mechanism repeating across other entitlement classes, extract it then rather than
prematurely inventing one.

## Research questions

How much loyalty value is stranded annually through expiration, fragmentation and
inefficient redemption? Which program rules can be represented reliably as
machine-readable constraints, and how fast do they drift? Can award availability be
observed reliably enough for routing? How should effective value be calculated without
misleading users with simplistic cents-per-point metrics? How should irreversible
transfers be governed? What evidence freshness is required before execution? Which
actions require explicit human authorization? Can optimization occur without custody of
user credentials or points — operating through read-only or account-authorized
connections and user-directed execution? Where do travel-agent, financial-service,
consumer-protection and tax boundaries arise? Is the strongest initial wedge expiration
prevention, trip optimization, fragmented-balance aggregation, or API infrastructure?

## Activation

Do not activate merely because the concept is compelling. Activation requires evidence of
meaningful stranded-value pain, a lawful and permitted data-access path, reliable rule
ingestion, measurable optimization advantage, users willing to pay for the intelligence,
manageable program and legal constraints, and clear differentiation from existing
award-search products.

RESERVED — DO NOT BUILD.
