# RESERVE — Executable Asset Semantics / Governed Stateful Asset Primitive

Status: RESERVED — research only. NOT an active build.
Captured: 2026-08-27.
Origin: analysis of a Token-2022 options architecture exposed a broader pattern — an
asset can carry machine-readable lifecycle semantics, authoritative metadata,
admissibility constraints, privileged transition authorities and terminal-state
behaviour. The general mechanism is preserved; the originating product, naming and
implementation are not.

Distinct from `ownership-proofs-vs-execution-rights.md`, which is doctrine about
possession versus permission. This reserve concerns **portable lifecycle semantics** and
zero-bilateral-integration reconstruction.

## Core idea

Conventional digital assets separate representation → application database → business
rules → lifecycle execution. This reserve explores whether certain governed assets should
instead expose enough authoritative machine-readable semantics that independent systems
can determine:

    what the asset is → what rights/obligations it represents
    → what actions are currently admissible → what transitions are possible
    → who may invoke them → how the asset terminates or resolves

Avoid describing such assets as autonomous or "self-executing" unless execution literally
occurs without an external actor. The better model is **capability-bearing / stateful
assets** whose lifecycle semantics travel with the asset while execution remains governed.

## Semantic completeness

A self-describing asset must encode enough to reconstruct its economic or operational
behaviour, not merely its identity.

Financial: underlying · strike · expiry · exercise style · payoff function · payout cap ·
settlement currency · collateral model · oracle source · settlement methodology ·
authority model.

Non-financial: scope · issuer · beneficiary · validity window · transition rules ·
revocation rules · delegated capabilities · terminal condition.

**Identity metadata is not semantic completeness.** If an external system can identify an
asset but cannot independently predict what it can do or what outcome it produces, the
asset is not fully self-describing.

## Authority doctrine

Distinguish *no human private key* from *no human governance capability*. A
program-controlled capability remains indirectly governed while the controlling program is
upgradeable. Every asset receipt should expose the full chain:

    asset → delegated authority → governing program
    → upgrade/governance authority → current immutability status

## Admissibility model

    requested transition → asset state → authoritative semantics → required evidence
    → authority validation → ALLOW / DENY / DEFER / ESCALATE → transition → receipt

## Composability distinction

Two separate measures, frequently conflated:

1. **Semantic composability** — can another system understand the asset and its rules?
2. **Operational interoperability** — can another system actually manipulate it using
   supported standards and required hooks?

A system can score high on the first and low on the second. Do not call an asset
universally composable merely because its metadata and policies are on-chain.

## Primitive test

Zero-bilateral-integration test: give an independent agent only the asset identifier plus
standard protocol interfaces, and determine whether it can correctly reconstruct identity,
rights, constraints, current lifecycle state, permitted actions, authority chain and
terminal behaviour — **without the issuer's private application database.**

## Potential applications

Not restricted to derivatives: governed credentials, licenses, machine permissions,
warranties, claims, escrow rights, usage rights, programmable commercial instruments,
equipment authorizations, agent spending/execution rights, tokenized contracts — any asset
whose lifecycle matters after issuance.

## Research questions

Which semantics belong on the asset versus in an external canonical state object? How are
schema upgrades handled without silently changing the meaning of an already-issued asset?
How does an integrator distinguish immutable semantics from mutable issuer metadata? How
should authority-chain changes propagate to asset trust? Can lifecycle semantics remain
portable across chains and non-blockchain systems? What is the minimum standardized schema
for genuine zero-bilateral integration?

## Relationship to existing canonical reserves

`ownership-proofs-vs-execution-rights.md` (doctrine) · `future-rights-exchange.md` ·
`tokenized-securities-authority.md` · `instrument-admissibility-envelope.md` (consumes
these semantics) · `vyre-vyrel-evolution.md` (signing, packaging and transport of
semantic/transition receipts) · VERITY · vLOID · OROS.

## Activation

Do not build because a comparable product exists. Activate only when an existing project
requires durable rights or obligations that must remain independently interpretable after
leaving the originating application.

## Core principle

**A governed asset is not truly portable merely because ownership is portable. Its
meaning, constraints, authority chain and lifecycle must be portable too.**

RESERVED. DO NOT BUILD.
