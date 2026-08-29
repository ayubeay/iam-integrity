# RESERVE — Trust-Graph Provenance & Credibility Laundering

Status: RESERVED — DO NOT BUILD. VERITY evidence primitive. Not a social-graph product,
not a reputation score, not a network-analysis tool.
Captured: 2026-08-29.
Parent: VERITY. Consumes IVCP.

## Scope boundary — read first

Four reserves own adjacent territory and are **not restated here**:

- `independent-validation-capability-promotion.md` — evidence ancestry, the Evidence
  Independence Graph, and `Agent count ≠ evidence independence`. This reserve consumes
  that machinery; it does not redefine an edge class.
- `verity-provenance-resilience.md` — artifact provenance, evidence bundles, and
  *"absence of evidence is not evidence of the opposite proposition."*
- `verity-physical-provenance.md` — physical producer discovery.
- `api-trust-exposure-model.md` — trust governance across interface surfaces.

## Sibling relationship — stated in both directions

`iam-external-identity-risk-signals.md` and this reserve share a substrate and run in
opposite directions:

    iam-external-identity-risk-signals   propagates RISK        downward along the authority graph
    this reserve                         propagates CREDIBILITY  upward through endorsement

IAM-risk asks: *given evidence an identity is compromised, which dependent capabilities
must be restricted?* This asks: *given that a trusted party vouches for an unknown one,
how much trust has actually been established?* Neither owns the other. Recording the
relationship here is what prevents the two being read as duplicates later.

## Core invariant

    ENDORSEMENT COUNT  ≠  INDEPENDENT TRUST ROOTS

Eighty-eight mutual connections are not eighty-eight independent attestations. They may
be one community, one introduction event, one platform's suggestion algorithm, or one
actor's coordinated set.

## The three mechanisms

**Trust Transitivity Attack.** Trust is treated as transitive when it is not. A trusts B,
B trusts C, therefore A extends trust to C — but B may have vouched for a different
property, at a different time, at a lower stake, or under no stake at all. Transitivity
that was never asserted is manufactured by traversal.

**Credibility Laundering.** Weak or absent evidence acquires apparent strength by passing
through a trusted intermediary. The intermediary's credibility attaches to a claim it
never evaluated. The laundering is usually invisible at the receiving end, because what
arrives is *a trusted party's assertion* rather than *the unexamined thing the trusted
party forwarded.*

**Verification Scope Expansion.** A party is verified for one property and subsequently
presented, or read, as verified generally.

    verified for X  ≠  verified
    verified at T0  ≠  verified now
    verified by us  ≠  verified by anyone
    verified as an identity ≠ verified as competent, solvent, or honest

## Trust edge properties

An endorsement edge is not a boolean. Where it is used consequentially, preserve: who
vouched · for what specific property · on what evidence · at what time · at what stake ·
whether the voucher would bear a cost if wrong · whether the voucher had independent
knowledge or was itself relaying · and what the receiving system inferred from it.

Edge classes, indicative: `ATTESTED_DIRECTLY · RELAYED · ALGORITHMIC_SUGGESTION ·
CO_MEMBERSHIP · TRANSACTIONAL_HISTORY · UNKNOWN_BASIS`.

**`UNKNOWN_BASIS` must not default upward.** An edge whose basis was never established is
not a weak attestation; it is not an attestation.

## Hard constraint — no ratio

**The Trust Independence Ratio remains a research concept. Do not define a formula.**

Reducing an eighty-eight-connection graph to `3/88` before relationship semantics, common
control, timing and genuine independence are understood would recreate the compression
problem this reserve exists to name. IVCP already states the governing restraint — *"do
not prematurely lock to one number"* — and that instruction governs here without being
re-legislated.

## Loop

    endorsement observed → basis established or marked UNKNOWN_BASIS
    → property scope identified → stake and independence assessed
    → shared-origin detection across edges → effective trust roots
    → trust contribution graded by consequence
    → ACCEPT / DISCOUNT / REQUIRE_DIRECT_EVIDENCE / DEFER / DENY → receipt

## Anti-patterns

Counting connections as attestations · treating platform-suggested relationships as
social evidence · allowing an intermediary's reputation to substitute for the evidence it
forwarded · carrying a verification beyond the property, time or stake it covered ·
inferring competence from identity verification · computing a trust ratio before edge
semantics are understood · treating mutual connection as bidirectional endorsement.

## Relationship to existing canonical reserves

`independent-validation-capability-promotion.md` (evidence ancestry; consumed, not
restated) · `iam-external-identity-risk-signals.md` (sibling, opposite direction) ·
`extraordinary-claim-evidence-tree.md` (`original → repost → reaction → article` lineage
for claims; this reserve is the same problem for endorsements) ·
`computable-accountability.md` (*"false source multiplicity must not inflate confidence"*)
· `verity-provenance-resilience.md` · `api-trust-exposure-model.md` · VERITY · IAM · vLOID.

## Research questions

Can an endorsement's basis be established without asking the endorser? How is stake
detected where none is declared? Which edge classes carry information at all, and which
are noise that merely looks like evidence? How should a trust assessment expire? Can
shared-origin detection operate on a graph whose edges are mostly `UNKNOWN_BASIS` — and
if not, is the honest output `UNRESOLVED` rather than a low number?

## Non-goals

Not a social-graph analysis product · not a reputation score · not a people-ranking system
· not a network-mapping tool · not authorization to compile personal information across
sources · not a formula for trust.

## Activation

Revisit when VERITY must evaluate a counterparty whose primary evidence is who vouches
for them; when an endorsement chain contributes to a consequential admissibility decision;
or when a case demonstrates credibility attaching to a claim no endorser evaluated.

RESERVED — DO NOT BUILD.
