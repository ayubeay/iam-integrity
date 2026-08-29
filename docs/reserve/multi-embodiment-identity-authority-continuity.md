# RESERVE — Multi-Embodiment Identity & Authority Continuity

Status: RESERVED — DO NOT BUILD. IAM/vLOID delegation primitive. Not an identity product,
not a session manager, not a device-management system.
Captured: 2026-08-29.

## Scope boundary — read first

- `human-machine-sovereignty-boundary.md` — **cross-link only, never host.** HMSB owns
  *which powers must remain non-delegable to machines*, and pre-commits in its own text
  to being folded if it drifts into general sovereignty prose. This reserve owns
  delegation **mechanics**, which is a different subject. Neither absorbs the other.
- `emaa-external-machine-action-admissibility.md` — authority budgets and dynamic
  authority decay for an **external** actor's proposed action at our boundary. This
  reserve governs **our own** identity's authority as it moves across embodiments. Same
  vocabulary, different subject; the budgets are consumed, not restated.
- `ownership-proofs-vs-execution-rights.md` — `possession ≠ permission`. This reserve
  extends that separation from assets to embodiments.
- IAM — identity state, roles, delegated permissions.

## Core invariant

    IDENTITY PERSISTS  ·  AUTHORITY DOES NOT TRAVEL WITH IT

Recognizing an identity in a new embodiment establishes *who is present*. It establishes
nothing about *what may be done there*. Authority is granted per embodiment, per scope,
per duration — never inherited by recognition.

## Embodiment

An embodiment is any distinct locus through which one identity can act: a device, a
session, an agent instance, a delegated sub-agent, a robot, a browser context, a
scheduled job, a background worker, a replica in another region.

**Embodiment divergence.** One identity acting through several embodiments accumulates
divergent state: different context, different observations, different tools, different
staleness, different compromise status. Two embodiments of one identity are not one actor
with two hands; they are two actors sharing a name.

    same identity + different embodiment
      → different observations
      → different context freshness
      → different tool exposure
      → different compromise surface
      → different admissible authority

## Execution lease

Authority reaching an embodiment is a **lease**: bounded, expiring, revocable, evidenced.

    granted scope · granted capabilities · granted duration
    · granted consequence ceiling · issuing authority · issuance evidence
    · exit condition · renewal requirement · revocation evidence · receipt

`execution lease` is genuinely absent from canon; the mechanism is not. It is the explicit
form of EMAA's authority budgets — which already range over "scope, privilege, rate,
geography, **time**, cumulative risk, blast radius" — applied inward to our own identity.

Lease states, indicative: `ISSUED · ACTIVE · DEGRADED · SUSPENDED · EXPIRED · REVOKED ·
SUPERSEDED`.

**Expiry is the default; renewal is an event requiring evidence.** A lease that renews
because nothing objected is a permanent grant wearing a duration field. This is the same
failure `hanoi-planner.md`'s Temporary State Doctrine names — *"preventing temporary
exceptions from becoming permanent drift."*

## Continuity without inheritance

Identity continuity across embodiments is desirable; authority continuity is not.

    identity continuity   preserve who this is, and the history that belongs to them
    authority continuity  do NOT preserve what they could do somewhere else

An embodiment that has been offline, that ran an older policy version, or whose context
freshness cannot be established, may hold a valid identity and no admissible authority.

## Revocation and the un-actable problem

Revoking a lease stops future action. It does not undo action already taken, and an
embodiment may be unreachable at revocation time. Preserve the distinction between
`REVOKED` and `REVOCATION_CONFIRMED`, and record which embodiments were reachable —
requesting revocation is not revocation, in the same sense that
`commerce-sniper.md` records that requesting withdrawal is not withdrawal.

## Anti-patterns

Inferring authority from identity recognition · renewing a lease by default · treating
a session as an authority grant · assuming embodiments share observations · assuming a
compromise in one embodiment is contained to it, or that it necessarily spreads ·
treating unreachable as revoked · granting a lease with no exit condition.

## Relationship to existing canonical reserves

`human-machine-sovereignty-boundary.md` (non-delegable powers — cross-link only) ·
`emaa-external-machine-action-admissibility.md` (authority budgets, authority decay) ·
`ownership-proofs-vs-execution-rights.md` (`possession ≠ permission`, extended here from
assets to embodiments) · `iam-external-identity-risk-signals.md` (risk propagation along
the authority graph; a degraded identity should degrade its leases) ·
`counterfactual-execution-governor.md` (embodied branch: graceful authority degradation
in the physical case) · `hanoi-planner.md` (Temporary State Doctrine) ·
`computable-accountability.md` (which lease authorized which action) · IAM · vLOID.

## Research questions

What evidence should a renewal require, and does it differ from issuance? How is
embodiment divergence detected before it produces a conflicting action? Should a lease
carry a consequence ceiling independent of capability scope? How does revocation
propagate to an embodiment that is offline, and what is the correct posture in the
interval? When one embodiment is compromised, what is the evidence-based basis for
restricting the others rather than all or none?

## Non-goals

Not an identity or SSO product · not a device-management system · not a session store ·
not a claim that more embodiments require more authority · not authorization to expand
delegation because delegation has become expressible.

## Activation

Revisit when one identity acts through more than one consequential embodiment; when a
delegated agent spawns sub-agents that inherit authority implicitly; when robotics or
scheduled execution act under a human identity; or when HMSB's open question — *"how
should scoped delegation expire, and what evidence revokes it?"* — requires a mechanism
rather than a research note.

RESERVED — DO NOT BUILD.
