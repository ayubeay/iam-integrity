# RESERVE — Human–Machine Sovereignty Boundary (HMSB)

Classification: Doctrine.
Status: RESERVED — DO NOT BUILD.
Captured: 2026-08-28.
Adjacencies: `docs/reserve/ownership-proofs-vs-execution-rights.md` ·
`docs/reserve/computable-accountability.md` ·
`docs/reserve/counterfactual-execution-governor.md` · EMAA · IAM · vLOID.

## Scope boundary — read first

This reserve survived collision analysis on a narrow margin and its scope is deliberately
tight.

- **Ownership Proofs vs Execution Rights** owns `possession ≠ permission`.
- **EMAA** owns hostile *external* machine action.
- **Computable Accountability** owns the responsibility chain from recommendation through
  authorized human to execution.
- **HMSB owns only this:** *which powers must remain non-delegable to machines even when
  the machine is authenticated, capable, correctly configured and otherwise admissible.*

If this file ever drifts into general "humans remain sovereign" prose, it has stopped
being distinct and should be folded into the reserves above rather than maintained
separately.

## Purpose

Define the boundary between machine *capability* and legitimate *authority over humans*.
Increasing machine capability must never be interpreted as increasing entitlement to
exercise power over a person.

## Core invariant

    Capability does not create authority.
    Prediction does not create consent.
    Inference does not create permission.
    Automation does not erase accountability.

## Separations that must not collapse

    identity              ≠ authority
    ownership             ≠ execution right
    preference inference  ≠ consent
    recommendation        ≠ decision
    decision              ≠ authorization
    authorization         ≠ unlimited delegation
    machine execution     ≠ disappearance of human/institutional responsibility

**A system must not silently expand its authority merely because it learns more about a
person or becomes more technically capable.** Capability growth and authority growth are
separate events with separate evidence requirements.

## Required chain for consequential physical-world action

    authorized intent → evidence → authority → machine interpretation
    → proposed action → admissibility → execution → observed outcome
    → accountable parties → receipt

## Protected boundary classes

Powers over these are candidates for non-delegable status and require explicit,
scoped, revocable human authority rather than inferred permission:

- bodily autonomy
- identity and impersonation
- property and control rights
- freedom of movement
- consequential financial authority
- employment and eligibility decisions
- access to essential services
- surveillance and privacy
- physical safety
- irreversible or difficult-to-reverse environmental actions

The list is a starting set, not a closed enumeration. Adding a class requires evidence
that delegation in that domain produces harm no downstream receipt can remedy.

## Failure state — sovereignty laundering

**Sovereignty laundering:** a human or institution allows an automated system to exercise
consequential power, and subsequently treats the machine as the responsible actor.

This is the failure mode HMSB exists to name. Computable Accountability's receipt chain
should make it impossible to hide: if the accountability graph terminates at "the machine
acted," the graph is incomplete, not finished.

## Research questions

Which powers are genuinely non-delegable versus merely high-risk? How is
non-delegability expressed in an authorization schema without becoming an
unmaintainable exception list? How should scoped delegation expire, and what evidence
revokes it? How does a system distinguish an authority *it was granted* from an authority
*it inferred it had*? Can sovereignty laundering be detected from receipts alone, or does
detection require the counterfactual — what a human would have decided?

## Activation

Activate only when a consequential system in the portfolio is about to exercise power in
one of the protected boundary classes, and existing admissibility governance cannot
express the non-delegable constraint. Until then this is doctrine that constrains design
reviews, not a component to build.

RESERVED — DO NOT BUILD.
