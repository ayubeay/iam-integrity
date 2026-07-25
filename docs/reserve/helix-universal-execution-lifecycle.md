# RESERVE — HELIX Universal Execution Lifecycle

Status: Reserved. Canonical execution doctrine for the HELIX ecosystem.
Architecture/governance only — no execution authority granted here.
Canonical home: iam-integrity/docs/reserve/helix-universal-execution-lifecycle.md
Captured: 2026-07-25 (consolidated packet; folds in "Context Classification
Layer" as a pre-execution layer and "Canonical Execution Doctrine
Preservation" as the evolution-governance section, per the compact-canon
directive)

## Purpose

Establish one canonical execution lifecycle for HELIX. Every enterprise,
marketplace, agency, startup, and individual ultimately faces the same
execution lifecycle; industries differ, execution primitives do not. This
document defines that single lifecycle, the environment-classification layer
that precedes it, and the governance rule that keeps future evolution from
fragmenting it into competing models.

## Non-goals

This is not a new module or a competing pipeline. It does not replace the
HELIX Execution Ecosystem reserve, the Universal Execution Interface, or any
domain engine — it names the lifecycle they all instantiate. It grants no
authority to build and introduces no top-level execution stage beyond the
seven defined here.

## The canonical lifecycle

    DISCOVER -> MATCH -> ROUTE -> COORDINATE -> VERIFY -> LEARN -> OPTIMIZE

- **Discover** — identify everything capable of solving the requested
  problem (APIs, suppliers, contractors, GPUs, regions, manufacturers,
  warehouses). Question: what exists?
- **Match** — determine which candidate best satisfies the objective given
  skills, cost, compatibility, capacity, geography, compliance, risk.
  Question: who or what is the best fit?
- **Route** — choose where execution should occur, by policy rather than
  price alone (latency, reliability, price, availability, preference,
  regulation, capacity). Question: where should execution happen?
- **Coordinate** — sequence the participant graph; execution is rarely a
  single actor. Question: who is waiting on whom?
- **Verify** — produce evidence, not assumptions (was it completed, who
  approved, what changed, what failed, can it be audited later). Output: a
  receipt.
- **Learn** — turn execution history into organizational knowledge (why was
  this slow, which supplier keeps failing, which provider becomes expensive,
  which workflow bottlenecks).
- **Optimize** — make progressively better decisions from that history
  (Supplier A for urgent, B for cost, C for quality, D for defect rate).

## Context Classification layer (pre-execution)

Execution should never occur without first classifying the operating
environment. Classification is not prediction — it describes what is
currently true so governance can decide. It may evaluate infrastructure
(health, capacity, latency, availability), markets (regime, liquidity,
volatility, participation), enterprise (business state, incident severity,
resources, congestion), security (threat level, trust posture, identity
confidence, integrity), and commerce (supply, counterparty reliability,
demand, pricing). It does not decide what to do; it presents a unified view
to Execution Governance, which then chooses ALLOW / ROUTE / THROTTLE /
ESCALATE / DEFER / DENY. This layer consumes DRIFT signals rather than
duplicating them: DRIFT detects and measures change; Context Classification
interprets the current environment; Execution Governance decides; Receipts
explain why.

## Evolution governance (Canonical Execution Doctrine Preservation)

There shall be one canonical Universal Execution Lifecycle. Future
innovations should map into existing stages, and the burden of proof for a
new top-level stage is intentionally high. When evaluating a new capability,
ask which existing stage it improves and whether it refines a stage, adds
evidence, or improves observability/governance/quality — if so, it extends
the architecture rather than creating another lifecycle. Prefer evolution by
refining stages, decomposing responsibilities, and adding sub-capabilities.
For example, Admissibility remains one stage while its internals decompose
(identity -> trust -> policy -> risk -> compliance -> capacity -> context ->
eligibility); Verification remains one stage while gaining cryptographic,
dependency, execution, policy, receipt, and identity checks; Routing remains
one stage while gaining cost/latency/capacity/geo/reliability/multi-provider
refinements.

A new top-level stage is admissible ONLY if all hold: it cannot be
represented as a refinement of an existing stage; it introduces a
fundamentally new execution responsibility; it applies universally across
domains; it simplifies rather than complicates the canonical doctrine; and
it is expected to remain foundational. This governance rule does not replace
the lifecycle — it governs how the lifecycle evolves.

## HELIX family mapping

API Connect (discovery/connectivity) · HELIX Exchange (capability matching &
provider routing) · HELIX Hash (compute/hash routing) · VERITY (trust,
evidence, verification) · Execution Governance (coordination & policy) ·
Receipts (auditable history) · HelixAtlas (visualization) · DRIFT (learning
from change) · future Optimization Engine (continuous improvement). No
component tries to do everything; together they cover the full lifecycle.

## Activation condition

Standing architectural doctrine, applied whenever any HELIX capability is
proposed or evaluated. Not an implementation task.

## Cross references

HELIX Execution Ecosystem reserve · Universal Execution Interface · HELIX
Exchange Layer · HelixShield Execution Governance · Execution Assurance
Layer · Universal Execution Timeline · Recalculation Doctrine · Adaptive
Execution Layer · Meta-Architecture: Observation to Strategic Moat (HELIX is
the execution span of that larger sequence).
