# RESERVE — Demand-Sovereign Market Infrastructure (DSMI)

Status: RESERVED — architecture doctrine / future primitive. NOT an active standalone
product, not authorization to implement.
Captured: 2026-08-28.

## Core insight

**Software ownership, demand ownership and market operation need not belong to the same
party.**

Rather than requiring every marketplace in the ecosystem to acquire both sides of its
market, allow organizations, creators, communities, enterprises or regional operators to
bring demand and supply networks they already possess, while the infrastructure provides
the governed execution layer beneath them.

    demand/community owner → market configuration
    → qualified supply + qualified demand → trust / identity / admissibility
    → matching / routing → transaction / execution → settlement
    → receipt + reputation state → market intelligence

The external operator retains its audience, community relationship, brand and
distribution. The infrastructure earns its position by making safe, reliable market
operation substantially easier than rebuilding the institution independently.

## Moat principle

Do not treat source code or marketplace UI as the moat. Defensible state compounds
through verified identities · participant reputation · transaction history · fraud and
adversarial intelligence · qualification and certification state · matching and execution
history · dispute-resolution history · settlement reliability · market-specific operating
rules · provenance · execution receipts · cross-market intelligence.

**Cloning the interface must not equal cloning the institution.**

## Executable market capacity

DSMI **consumes** the thinnest-leg primitive owned by
`executable-capacity-thinnest-leg.md`. Do not restate it here.

At any execution moment, usable market capacity is bounded by the scarcest required leg —
qualified demand, qualified supply, matching capacity, trust/admissibility capacity,
execution capacity, settlement capacity — subject to location, timing, qualification,
price, policy and risk. Aggregate registration counts do not establish local liquidity.

## Applications across existing reserves

`helix-builders.md` — communities bring builders and customers; infrastructure supplies
identity, trust, matching, governed execution and receipts.
WIRE — enterprises, robotics companies or regional operators originate missions;
contributor qualification, validation, provenance and receipts come from the platform.
SoundKeep / Sound-as-a-Service — artists, labels or creative networks operate specialized
collaboration markets while retaining audiences.
ShiftTrust — local organizations originate labour demand; qualification, availability,
dispatch, safety and receipts are governed.
`commerce-sniper.md` — counterparties retain customer relationships; infrastructure
supplies intelligence and governed execution.

## Stack mapping

IAM (participant identity) · VERITY (counterparty and market trust) · DRIFT (changing
liquidity, reliability, conditions) · Information Admissibility Governor (which market
signals may support decisions) · OROS (coordination) · vLOID (admissibility) · Shield
Router / SURVIVOR (safety verification) · Universal Money Router (settlement routing) ·
HelixAtlas (market topology, liquidity constraints, execution routes) · receipts.

## Business-model doctrine

Do not assume a universal revenue split. Pricing should reflect actual infrastructure
burden and value: transaction fees · platform fees · usage-based pricing · enterprise
licensing · settlement fees where legally appropriate · premium trust and verification ·
market-operation services · revenue sharing where strategically appropriate.

Governing test: *value plus operational and risk burden removed for the operator exceeds
the cost and strategic benefit of replacing the infrastructure.* The platform becomes hard
to replace through accumulated operational capability and trusted state, **not artificial
lock-in.**

## Mutual capability

Demand-sovereign operators are not acquisition channels. The relationship follows the
mutual-capability-compounding doctrine: operators become better able to serve their
communities while the infrastructure becomes more capable through legitimate market
learning — **without taking ownership of the operator's community merely because the
infrastructure powers its transactions.**

## Research questions

Which existing product provides the strongest first use case? What market primitive is
genuinely reusable across verticals? How should operator-owned reputation interact with
global reputation? How should liquidity be measured at location, time and qualification
granularity? How should thin-leg shortages trigger pricing, recruiting, routing or
deferral? What data may legitimately compound across markets, and what must remain
isolated between operators? At what scale does the infrastructure become an institution
rather than marketplace software?

## Activation

Revisit when an existing ecosystem product has a real two-sided market, outside
communities that already possess distribution, recurring rather than hypothetical
execution, and evidence that reusable identity/trust/execution/settlement infrastructure
materially reduces the cost or risk of operating that market.

**Do not build a generic marketplace-as-a-service product merely because the architecture
is possible.** The first deployed vertical teaches what the reusable primitive actually
needs to be.

RESERVED — DO NOT BUILD.

---

## Extension 2026-08-29 — Demand-Induced Supply Formation

Status: RESERVED — DO NOT BUILD. Architectural refinement of this reserve.

### Why this belongs here and not in its own file

This reserve's own research questions ask *how should thin-leg shortages trigger pricing,
**recruiting**, routing or deferral?* — and recruiting is supply formation. What follows
answers that question rather than founding a parent for it. The thinnest-leg primitive
remains owned by `executable-capacity-thinnest-leg.md` and is consumed, not restated.

### Demand can instruct supply creation

A market is usually modelled as matching existing supply to existing demand. Where the
scarce leg is supply, sufficiently evidenced demand becomes an instruction to create it:

    observed unmet demand → qualification of that demand
    → marginal supply value → recruitment or capacity formation
    → qualification of new supply → matching → execution → receipt

**Marginal Supply Value** is what one additional qualified unit of the scarce leg is worth
at this location, time and qualification level — not the average value of supply in the
market. A market with abundant unqualified supply and one missing qualified provider has a
high marginal supply value and a low average one, and only the first should drive
recruitment.

### The qualification ladder

Demand evidence is not one state, and treating it as one is how synthetic demand becomes
real capacity commitment:

    INTEREST     ≠  COMMITMENT  ≠  TRANSACTION

Interest is an observation. Commitment carries a cost of abandonment. A transaction has
settled. Recruiting supply against interest, or pricing against it, converts an
observation into an obligation someone else has to bear.

### Synthetic Demand Attack

Where demand signals induce supply formation, fabricating demand becomes a way to induce
cost in the market or its participants — recruiting providers who then find nothing to
serve, moving pricing, exhausting qualification capacity, or degrading the reputation of
a market that appears unreliable.

Defences belong to existing owners rather than here: VERITY for counterparty and signal
trust, the Information Admissibility Governor for which signals may support a decision,
and IVCP's effective-multiplicity extension for the specific case where **many demand
signals share one origin.** Twenty enquiries traceable to one campaign are not twenty
demand observations.

### Doctrine

**Do not create supply against demand you have not qualified.** The infrastructure's
value to an operator is that it can tell the difference; a market that recruits against
noise imposes the cost of that noise on its participants, who did not observe it and
cannot audit it.

RESERVED — DO NOT BUILD.
