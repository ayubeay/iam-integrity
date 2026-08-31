# RESERVE — Protocol-Independent Capability Envelope (PICE)

Status: RESERVED — architecture/research only. NOT an active build, no new repository.
Captured: 2026-08-28.

## Scope boundary — read first

Three existing reserves own adjacent territory and are **not restated here**:

- `../RESERVE-EXECUTION-ADAPTER-FRAMEWORK.md` (EAF) — adapters, provider bindings,
  protocol and transport translation, runtime invocation contract.
- `universal-execution-interface.md` (UEI) — the vertical execution stack, L1 identity
  through L6 receipts.
- `intelligence-resource-governance-layer.md` — progressive capability admission and
  total-objective-cost accounting for *intelligence* resources.

PICE owns only the object between them: **the semantic capability that governance
consumes**, shared across every binding of the same operation. EAF's own flow places
governance before the execution contract without saying what governance evaluates. PICE
is that. A local function call has no adapter and still needs the envelope.

## Core thesis

Do not couple execution governance to one capability protocol. The same underlying
operation may arrive via MCP, REST, GraphQL, CLI, SDK, local function, plugin, stored
procedure, message interface, robot controller, or a protocol that does not exist yet.

    TRANSPORT / INVOCATION ≠ CAPABILITY ≠ AUTHORITY
    ≠ EXECUTION ADMISSIBILITY ≠ SUCCESSFUL EXECUTION

## Loop

    objective → capability discovery → interface discovery → capability normalization
    → identity/authority → input/data admissibility → consequence/risk → policy
    → execution admissibility → ALLOW / LIMIT / STEP-UP / DEFER / ESCALATE / DENY
    → protocol-specific invocation → verification → receipt

**The governance decision concerns what the action does, not how it is invoked.**

## Envelope fields (provisional)

capability_id · semantic_operation · provider · resource · action · transport_type ·
transport_binding · input_schema · output_schema · side_effects · required_identity ·
required_authority · permission_scope · data_classification · external_data_release ·
financial_exposure · physical_exposure · reversibility · idempotency · blast_radius ·
dependency_requirements · pre/postconditions · verification_method · rate and cost
constraints · jurisdiction constraints · policy requirements · receipt requirements ·
version · provenance · confidence.

## Worked example

`payments.refund` (MCP) · `POST /refunds` (REST) · `payments refund create` (CLI) ·
`client.refunds.create()` (SDK) normalize to one capability:

    CAPABILITY: TRANSFER_CONTROLLED_VALUE_BACK_TO_CUSTOMER
    RESOURCE:   PAYMENT
    SIDE_EFFECT: FINANCIAL
    REVERSIBILITY: LIMITED
    AUTHORITY: REFUND_PERMISSION
    VERIFICATION: PROVIDER_CONFIRMATION + LEDGER_RECONCILIATION

The transport differs. The governed economic effect does not.

## Core doctrine

**Tool availability is not execution authorization.**

    CAPABILITY_DISCOVERABLE ≠ CAPABILITY_EXPOSED ≠ CAPABILITY_AUTHORIZED
    ≠ ACTION_ADMISSIBLE ≠ ACTION_EXECUTED ≠ OUTCOME_VERIFIED

This distinction must survive any interface standard.

## Static access vs dynamic admissibility

Traditional access control answers *can identity X call operation Y?* PICE with vLOID
enables the stronger question: *may identity X perform operation Y on resource Z for
objective O with data D under state S at consequence level C under policy P, right now?*

A $14 refund to a verified customer may ALLOW; $40,000 may require additional authority;
a refund to an altered destination may DEFER; a refund after identity-risk escalation may
DENY. **The capability has not changed. Its current admissibility has.**

## Semantic side-effect classification

`READ_ONLY · LOCAL_MUTATION · REMOTE_MUTATION · FINANCIAL · IDENTITY · COMMUNICATION ·
PUBLICATION · SECURITY · INFRASTRUCTURE · PHYSICAL_WORLD · IRREVERSIBLE`

One `HTTP POST` can send email, delete a database, transfer money, deploy production,
unlock a door, or move a robot arm. **The transport cannot establish consequence. The
semantic capability must.**

## Multiple bindings, binding-specific risk

    capability
      ├── MCP binding
      ├── REST binding
      ├── CLI binding
      └── local binding

Interface comparison should be evidence-based — reliability, latency, context overhead,
schema quality, error rate, security boundary, permission granularity, observability,
cost, determinism, offline availability — never ideological. Governance decisions should
be equivalent across bindings *unless binding-specific risk genuinely differs*.

## Security principle

Do not place long-lived provider credentials into agent-visible context merely because an
agent must invoke a capability. Prefer bounded execution brokers:
`agent intent → approved capability → bounded authorization → execution adapter → provider`.
PICE is **not** a secret-management system; IAM and secret infrastructure own credentials.

## Protocol survivability

Do not assume today's dominant standard remains dominant. The architecture must allow
replacing a binding without rewriting the semantic governance model.
**Protocols change. Capability semantics should remain stable where the real-world action
is stable.**

## Adversarial tests reserved

Same action via MCP/REST/CLI → equivalent governance unless binding risk differs · tool
renamed → capability still recognizable · provider changes schema → old version becomes
stale rather than silently trusted · access without authority → DENY · authority without
sufficient evidence → DEFER · capability advertised read-only but observed to mutate →
trust degradation and escalation · secret leaked into agent-visible schema → boundary
failure.

## Non-goals

Not a replacement for MCP, APIs or CLIs. Not a new agent protocol. Not a universal API
schema. Not a secret manager. Not another execution governor. Not permission to expose
every tool to an agent. Not a reason to wrap every API in MCP, nor to remove MCP where it
provides useful standardization.

## Activation

Activate when API Connect or another system must govern multiple invocation mechanisms for
one semantic capability; vLOID begins authorizing heterogeneous tool ecosystems; agent
deployments require protocol-independent permissioning; MCP/API/CLI duplication becomes a
measurable governance problem; or physical and digital capabilities need one semantic
control surface. At activation: inspect API Connect and vLOID semantics first; take one
real operation represented through at least two interfaces; define the smallest envelope
that governs it; test equivalent admissibility across bindings; deliberately test
binding-specific risk differences.

## Core doctrine

**Govern the action, not the adapter.**

A protocol can tell an agent *how* to invoke a capability. It cannot establish *whether*
the agent should be allowed to exercise that capability in the present context.

RESERVED — DO NOT BUILD.

---

## Extension 2026-08-29 — Observed Execution Cost in the Envelope

Status: RESERVED — architecture/research only. NOT an active build.

### Why this belongs here and not in its own file

A submission proposed that a capability layer should normalize not only what a capability
returns but what it cost the system to execute. The envelope fields above already carry
**rate and cost constraints**. What they do not carry is the observed counterpart. That is
one field class added to an existing list, not a mechanism, so it is recorded here.

### Constraint cost and observed cost are different fields

    CONSTRAINT COST   what this capability is permitted to cost      (pre-execution, policy)
    OBSERVED COST     what this invocation actually cost             (post-execution, evidence)

The first governs admissibility. The second is evidence, and belongs in the envelope
because the same semantic capability invoked through different bindings can cost
materially different amounts — which is precisely the binding-specific difference this
reserve exists to make visible.

Candidate observed fields: invocation cost · provider cost · context overhead · retry
count and retry cost · latency observed against the declared budget · verification cost ·
which binding served it.

### Three owners already share cost; do not become a fourth

    execution-economics.md    owns the unit — COST PER VERIFIED SUCCESSFUL EXECUTION —
                              and the outcome vocabulary including TERMINATED_ON_BOUND
    provider-qualification-and-routing.md
                              owns cost per workload unit as a qualification criterion
    intelligence-resource-governance-layer.md
                              owns total-objective-cost accounting

**This extension defines no cost unit and computes no total.** It records what an
invocation of a semantic capability cost, so that governance comparing two bindings of the
same capability has evidence rather than an assumption. The accounting is consumed from
the owners above.

### Why it matters to admissibility

`GOVERN THE ACTION, NOT THE ADAPTER` still holds — but where two bindings of one capability
differ materially in cost, that is a binding-specific difference of exactly the kind this
reserve already says must be surfaced rather than abstracted away. **Equivalent governance
across bindings does not mean equivalent economics across bindings**, and a system that
records only the former cannot explain the latter.

RESERVED — DO NOT BUILD.
