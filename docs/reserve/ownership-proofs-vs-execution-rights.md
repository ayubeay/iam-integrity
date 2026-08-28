# RESERVE — Ownership Proofs vs Execution Rights

Status: Reserved future architecture (doctrine). Foundational — not a
blockchain feature. No execution authority granted here.
Canonical home: iam-integrity/docs/reserve/ownership-proofs-vs-execution-rights.md
Captured: 2026-07-25 (signal: recurring across Web3 ownership and AI-agent
capability discussions — possessing an asset is not the same as being
allowed to act with it)

## Purpose

Make a strict distinction between proving ownership and authorizing
execution. Ownership answers "do you possess this asset?" Execution rights
answer "what actions are currently allowed because you possess it?" Related
but fundamentally different, and never interchangeable. As assets begin to
control AI agents, autonomous workflows, financial systems, music rights,
software licenses, enterprise permissions, robotics, and digital twins, the
implicit assumption "own asset -> can do everything" becomes dangerous.

## Non-goals

Not an NFT or token feature, not a DRM scheme, and not a permissions library.
It defines a doctrine — a separation of layers — rather than a product.
Ownership alone never bypasses governance.

## Doctrine

Separate identity into two independent layers: an Ownership Layer (possession)
and an Execution Rights Layer (whether a requested action is admissible under
current policy, context, governance, and environment). Execution rights may
change without ownership changing; ownership may persist while execution
rights are revoked, restricted, delegated, expired, or made conditional.

    Own NFT: view yes, download no, resell yes, commercialize no,
             AI-derivatives yes, mint child works no
    Enterprise API key: read yes, write no, delete no, deploy no,
             rotate keys yes
    AI agent: research yes, schedule yes, spend money no, execute trades no,
             hire contractors (requires approval), delete files no
    Music rights: personal listening yes, commercial sync no, AI remix yes,
             redistribute no, derivatives licensed-only

## Relationship to existing stack

Ownership becomes one admissibility signal, never a direct authorization.
HELIX pipeline: Identity -> Ownership Proof -> Execution Rights Evaluation ->
VERITY -> Policy -> Admissibility -> Execution -> Receipt. Execution rights
are dynamic objects influenced by ownership, licenses, subscriptions, payment
status, geography, time, org policy, regulation, risk/trust scores, AI
confidence, delegation, and governance decisions — evaluated, not assumed.
Every receipt distinguishes ownership-verified from execution-right-granted,
with policy and reason. HelixShield applies this directly to autonomous
agents; Execution Assurance uses it to gate governed completion; Future
Rights Exchange and Governed Capital Eligibility inherit the separation.

## Activation condition

Standing doctrine, applied wherever possession might be mistaken for
authority. Not an implementation task.

## Why it matters

As AI agents act autonomously, ownership becomes common but execution becomes
the scarce, governed resource. Infrastructure should evolve from "who owns
this?" to "who may do what, under which conditions, with which evidence?"
Ownership establishes identity; execution rights establish authority; and
authority — not ownership — should determine what autonomous systems may do.

## Cross references

Future Rights Exchange · Governed Capital Eligibility · HelixShield Execution
Governance · Execution Assurance Layer · HELIX Universal Execution Lifecycle
(admissibility stage) · IAM / VERITY / vLOID.

---

## Extension 2026-08-28 — Knowledge Execution Rights

Status: RESERVED — research only.
Origin: the Information-to-Execution Integrity research family. Knowledge Execution Rights
folds into this reserve rather than becoming a separate file, because this document already
owns the `possession ≠ permission` invariant and this is that invariant applied to
knowledge rather than to assets.

### The proposition

Conventional `READ` / `WRITE` authorization may be insufficient for agent knowledge. A
single datum may warrant separately governed permissions:

    OBSERVE      may the system perceive it at all
    STORE        may it be retained, and for how long
    RETRIEVE     may it be surfaced into a working context
    CORRELATE    may it be joined with other data
    INFER        may conclusions be drawn from it
    DISCLOSE     may it be communicated onward, and to whom
    EXECUTE_FROM may an action be authorized on its basis
    DELEGATE     may any of the above be passed to another agent

The claim under test: **an agent lawfully permitted to `OBSERVE` and `STORE` a datum is
not thereby permitted to `CORRELATE`, `INFER`, `DISCLOSE` or `EXECUTE_FROM` it.** Access
and epistemic authority are separate grants, exactly as possession and execution right are
separate in the parent doctrine.

### Why this is not merely access control

Access control governs a boundary crossing at one moment. Knowledge rights govern what may
be *done with what has already legitimately crossed*. The distinction becomes consequential
when the agent's own inference is the harmful act — for example when individually
admissible facts compose into a capability nobody granted. That composition case is
specified as `docs/research/tests/TEST-CONCENTRATION-001.md`; the restraint case as
`docs/research/tests/TEST-RESTRAINT-001.md`.

### Open questions

Which of the eight rights are genuinely separable in practice, and which collapse into one
another? Can they be expressed without an unmaintainable per-datum permission matrix? How
do rights propagate to a derived artifact — does an inference inherit the most restrictive
right among its inputs? What revokes a right already exercised, given that inference cannot
be un-drawn? How does `DELEGATE` interact with agent-to-agent handoff?

### Activation

Reserve only. Do not implement a knowledge-rights schema before the two tests above
establish whether the existing admissibility primitives can already express these
constraints without semantic distortion. If they can, extend those primitives rather than
building a new layer.
