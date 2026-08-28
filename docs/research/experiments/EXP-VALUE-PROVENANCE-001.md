# EXP-VALUE-PROVENANCE-001 — Computational Value Provenance & Coordinated Admissibility

Status: `PROPOSED`
Registered: 2026-08-28
Canonical relationship: VERITY · IAM · `docs/reserve/instrument-admissibility-envelope.md` ·
`docs/reserve/intelligence-resource-governance-layer.md` ·
`docs/reserve/computable-accountability.md`

**Do not create a standalone module unless this experiment demonstrates an actual
representational gap.**

## Origin signal

A published AI-gateway fraud report describing coordinated fraudulent signups, stolen-card
usage, AI-credit acquisition and downstream resale or consumption. The specific reported
monetary amount is operator-reported and is **not** required for the architectural finding —
the finding does not depend on the number being accurate.

## Research question

Can the existing stack preserve and reason over the complete transformation

    funding source → entitlement / credits → computational execution
    → generated economic value → downstream disposition

**without losing provenance when value changes form?**

The deeper hypothesis: **computational capability can become an intermediate economic
asset.** Fraud therefore need not terminate at a payment event — value can be transformed
through computation before being monetized again.

Preserve the distinction:

    PAYMENT ADMITTED ≠ ACCOUNT TRUSTED ≠ RESOURCE CONSUMPTION ADMISSIBLE
                     ≠ DERIVED VALUE CLEAN

## Claim A — Computational Value Provenance

Determine whether the architecture can answer, for a consequential compute entitlement:
what funded it · which identity received it · which account, device or agent exercised it ·
what resources were consumed · what workload resulted · whether the entitlement was
transferred or resold · whether suspicious downstream economic activity can still be linked
to its upstream provenance.

Desired receipt lineage:

    funding evidence → entitlement issuance → identity / authority
    → compute allocation → execution receipts → derived artifact / service / value
    → disposition

**The goal is not universal transaction surveillance.** Preserve only the evidence necessary
for governed execution, fraud investigation and accountability, with privacy controls and
bounded retention.

## Claim B — Coordinated Admissibility

Test the invariant: **local admissibility does not imply collective admissibility.**

Synthetic setup: 1,000 accounts each individually appearing admissible — plausible signup,
distinct-looking identity, ordinary individual usage, low payment value, no per-account
threshold breach. Collectively they exhibit synchronized account creation, correlated
infrastructure, shared funding ancestry, similar entitlement acquisition, coordinated
workload patterns, common downstream destinations, and rotating identities under one
controlling entity.

Can VERITY/IAM represent group-level or graph-level evidence **without falsely converting
correlation into proof of common control?**

Statuses must preserve uncertainty:

    INDIVIDUALLY_ADMISSIBLE · COLLECTIVELY_UNASSESSED · COORDINATED_PATTERN_OBSERVED
    · COMMON_CONTROL_INFERRED · COMMON_CONTROL_VERIFIED

**Do not silently promote the second or third state into the fifth.**

## Claim C — Observation-Boundary Integrity

Model the same attack from several partial observers: a payment provider sees payment
attempts · an identity/email provider sees account creation · an AI gateway sees signup
velocity, credit issuance and compute use · a model provider sees inference traffic · a
downstream marketplace may see resale.

Test whether the system can combine trustworthy partial observations while preserving source
provenance → what each observer actually saw → overlap → independence → privacy boundary →
confidence → unresolved gaps.

**No source may be represented as knowing more than its observation boundary permits.**

## Acceptance / rejection criteria

Existing architecture is **sufficient** if the full lineage and coordinated-risk reasoning
can be represented through current IAM, VERITY, IAE, IRGL and receipt primitives without
semantic distortion.

A new primitive is justified **only** if one of these remains structurally unrepresentable:
value provenance across transformations · coordinated/group admissibility distinct from
individual admissibility · cross-observer evidence lineage · bounded propagation of upstream
compromise into downstream derived value.

If representable, **extend existing schemas later rather than creating another module.**

## Metrics (simulation)

Individual false-positive rate · collective false-positive rate · coordinated-ring detection
rate · time from first signal to coordinated-pattern detection · percentage of value lineage
reconstructable · unsupported common-control inference rate · provenance breaks · stale
evidence use · privacy exposure required per detection · successful isolation without
unnecessary account-wide shutdown.

## Doctrine preserved regardless of outcome

**Fraud may transform value rather than merely transfer it.**

**A system that sees only the payment can miss the attack; a system that sees only the
account can miss the campaign.**

**Correlation is evidence for investigation, not automatic proof of conspiracy or common
control.**

## Evidence boundary

Any conclusion holds only for the synthetic population, observer topology and entitlement
model constructed for the run. It does not generalize to a real gateway's fraud rate, and
the origin report is not treated as measured evidence.

## Provenance

    source artifact:       operator-published AI-gateway fraud report (origin signal only)
    registered:            2026-08-28
    implementation commit: none — no implementation authorized
    evidence boundary:     synthetic simulation only
    conclusion date:       pending
