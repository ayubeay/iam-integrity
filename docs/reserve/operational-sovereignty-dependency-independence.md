# RESERVE — Operational Sovereignty & Dependency Independence

Status: RESERVED — DO NOT BUILD. Cross-stack doctrine. Not a multi-cloud mandate, not a
redundancy programme, not a procurement policy.
Captured: 2026-08-29.

## Scope boundary — read first

This reserve consumes four existing owners and **must not restate any of them**. If it
ever drifts into general resilience prose, it has stopped being distinct and should be
folded into the reserves below rather than maintained separately.

- `provider-qualification-and-routing.md` — provider evaluation, workload-aware routing,
  jurisdiction as a routing dimension, and the rule that *"a fallback provider does not
  inherit another provider's trust or evidentiary status. Continuity of supply must not
  become continuity of assumed credibility."*
- `sovereign-intelligence-routing.md` — intelligence routing sovereignty.
- `verity-provenance-resilience.md` — *"one removable or forgeable signal must not become
  a single point of truth."*
- `independent-validation-capability-promotion.md` — the Evidence Independence Graph, and
  in particular `SHARES_SOURCE_WITH`, which is the evidence-side analogue of the ancestry
  question asked here.

**What is genuinely new:** PQR is provider-scoped by its own purpose and evaluates
providers within a capability. No reserve asks whether dependencies *across different
classes* share a control ancestor — a question no provider-routing model can see, because
the shared ancestor is not a provider.

## Core invariant

    REDUNDANCY  ≠  SOVEREIGNTY

Redundancy counts alternatives. Sovereignty asks whether the alternatives can fail
independently. Three providers on one substrate are one provider with three invoices.

## Dependency classes

Ancestry must be reasoned about across classes, not within one:

    compute / hosting · network transit · DNS · certificate authority
    identity provider · package registry · model vendor · model lineage
    data source · oracle · payment rail · settlement venue
    jurisdiction · legal entity · key custody · observability

A single upstream — one cloud region, one registrar, one CA, one foundation model
lineage, one clearing venue, one jurisdiction — can appear in several of these
simultaneously without appearing twice in any one of them.

## Common-control ancestry

For any capability the system depends on, the question is not *how many providers serve
it* but *how many independent failure opportunities exist*:

    declared dependency → resolved actual dependency → upstream ancestry
    → shared-ancestor detection across classes → effective independence
    → mission impact if the ancestor fails

Ancestry states, indicative: `INDEPENDENT · SHARED_SUBSTRATE · SHARED_CONTROL ·
SHARED_JURISDICTION · SHARED_LINEAGE · UNRESOLVED`.

**`UNRESOLVED` is a first-class state and must not default to `INDEPENDENT`.** An
unexamined dependency is not an independent one; treating it as independent is the
failure this reserve exists to prevent.

## Mission Independence Envelope

Sovereignty is not absolute and should not be pursued absolutely. The envelope states,
for a given mission: what must keep operating, for how long, under which named failures,
at what degraded level, and what may legitimately stop.

    mission → essential capabilities → tolerable degradation → survivable failures
    → required independent failure opportunities → envelope
    → CONTINUE / DEGRADE / DEFER / SUSPEND when the envelope is breached

Declaring that everything is essential produces no envelope and no decisions. The
envelope's value is in what it deliberately declines to protect.

## Doctrine

**Independence is a property of failure, not of contracts.** Two vendors, two invoices and
two support channels create no independence if one power grid, one registrar, one model
lineage or one legal regime can remove both.

**Sovereignty scales with consequence.** Independence is bought with cost and complexity,
so the burden should rise with what fails when the ancestor does — the same shape as
IVCP's *"independence scales with consequence."*

## Non-goals

Not a mandate to duplicate every dependency · not a multi-cloud requirement · not an
argument for self-hosting · not a claim that local infrastructure is inherently more
sovereign · not a procurement or vendor-selection policy · not a reason to reject a
superior single provider where the envelope tolerates its failure.

## Relationship to existing canonical reserves

`provider-qualification-and-routing.md` (provider-scoped evaluation and routing; supplies
the resolved dependency facts this reserve reasons over) · `sovereign-intelligence-routing.md`
· `verity-provenance-resilience.md` · `independent-validation-capability-promotion.md`
(`SHARES_SOURCE_WITH`) · `execution-jurisdiction-gap.md` (institutional admissibility as a
dependency class) · KONIGO Connect (continuity and failover) · DRIFT (ancestry changing
without notice) · `computable-accountability.md` (receipts recording which ancestry state
was believed at execution).

## Research questions

How is actual ancestry discovered rather than declared, given that providers do not
publish their own upstreams? How often does resolved ancestry change without notice, and
what invalidates a prior independence assessment? Can effective independence be expressed
without collapsing to a single score — the same restraint IVCP applies to evidence
independence? How should an envelope breach interact with obligations already accepted?

## Activation

Revisit when a single upstream failure degrades two or more capabilities believed
independent; when a mission acquires a stated continuity requirement; when jurisdiction
becomes an operational rather than theoretical constraint; or when PQR's routing needs an
independence input it cannot compute from provider observations alone.

RESERVED — DO NOT BUILD.
