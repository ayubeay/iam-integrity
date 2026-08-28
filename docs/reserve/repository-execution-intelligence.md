# RESERVE — Repository Execution Intelligence & Architectural Admissibility Graph (REI/AAG)

Status: RESERVED — research / architecture direction. NOT an active build.
Captured: 2026-08-27.
Potential homes: vLOID developer/execution governance · HELIX Builders · API Connect
repository intelligence · future agentic software-development infrastructure.

## Core thesis

Understanding a repository is not equivalent to understanding whether a proposed change
is architecturally safe, appropriate, authorized, or compatible with the systems
depending on it.

Current AI coding loop: repository → retrieve files → infer context → generate change →
tests → merge.

Governed loop:

    repository → architectural model → dependency graph → runtime/consumer evidence
    → architectural invariants → proposed modification → blast-radius analysis
    → admissibility judgment → execution → verification → receipt
    → architectural model update

The stronger capability is not "AI understands the codebase" but *"the system understands
what a proposed change means to the architecture and can produce evidence explaining why
it was or was not admissible."*

## Architectural Admissibility Graph

Files → symbols → modules → packages → services → data types → APIs → databases →
events → dependencies → runtime paths → tests → deployments → external consumers →
owners → permissions → architectural invariants → historical decisions → production
evidence → execution receipts.

The graph distinguishes static relationships from relationships supported by runtime or
operational evidence.

## Change admissibility loop

    proposed change → affected graph nodes → dependency blast radius → known consumers
    → architectural invariants → tests / runtime evidence → permissions / ownership
    → uncertainty → ALLOW / ALLOW_WITH_VERIFICATION / REQUIRE_REVIEW / DEFER /
      ESCALATE / DENY

Each decision preserves the evidence that produced it.

## Static truth vs execution truth

A graph can say *"nothing imports X."* That does not establish *"nothing depends on X."*

Distinguish STATIC_CODE_EVIDENCE · RUNTIME_EVIDENCE · API_CONSUMER_EVIDENCE ·
DEPLOYMENT_EVIDENCE · HISTORICAL_EVIDENCE · HUMAN_DECLARED_CONSTRAINTS.

**Never assume "unused in this repository" means "unused."**

## Continuous architectural knowledge

Repository understanding should not be rebuilt from zero every agent session.
change → graph delta → verification → new receipt → architectural state update.

Node/edge states: CONFIRMED · OBSERVED · INFERRED · STALE · CONFLICTING · UNKNOWN.
**An agent must not silently convert inference into architectural fact.**

## Architectural invariants

API contracts that must remain compatible; security boundaries; identity/permission
requirements; data ownership; service responsibilities; required receipt production;
execution-governance constraints; deployment boundaries; backward compatibility;
prohibited dependency directions.

## Agent work boundaries

    objective → relevant architecture slice → required dependencies
    → authorized resources → protected boundaries → admissible operations → execution

An architectural sandbox based on *meaning* rather than filesystem permissions.

## Capability-specific project maps

Contributors need not receive identical repository context. An API engineer, a security
engineer, a frontend contributor and an architect each need different slices. Retrieve
the minimum sufficient architectural context for the task — reducing institutional-memory
transfer cost while limiting unnecessary exposure.

## Doctrine

    Repository retrieval is not repository understanding.
    Repository understanding is not architectural understanding.
    Architectural understanding is not execution authorization.
    Execution authorization is not proof of successful execution.

Preserve the full chain: observation → architectural model → evidence → uncertainty →
proposed change → admissibility → authorization → execution → verification → receipt →
updated knowledge.

## Relationship to existing canonical reserves

- IAM, VERITY, vLOID, OROS, DRIFT as identity, trust, admissibility, coordination and
  change detection respectively.
- `computable-accountability.md` — receipts.
- `helix-builders.md` — bounded architectural slices for contributors.
- `invariant-precomputation.md` — invariant handling.
- API Connect — API truth reconciliation: declared surface vs actual implementation vs
  known consumers vs runtime observation vs vestigial components vs unknown external
  dependencies. Supports evidence-backed retirement rather than deletion by search.

## Adversarial questions before activation

How reliably can architecture be inferred across heterogeneous repositories? How are
dynamic dependencies represented? How do we detect consumers outside repository
visibility? How quickly does architectural knowledge go stale? How is contradictory
evidence represented? What confidence threshold permits autonomous modification? Which
changes always require human approval? **How do we prevent an AI-generated architectural
graph from becoming a confidently incorrect source of truth?**

## Activation

Reserve only. Revisit when repository-scale agent execution, HELIX Builders, API truth
reconciliation, or autonomous software maintenance creates a concrete need.

RESERVED. DO NOT BUILD.
