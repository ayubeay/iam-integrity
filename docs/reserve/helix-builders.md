# RESERVE — HELIX Builders

Status: RESERVED — architecture direction. NOT an active build.
Captured: 2026-08-27. Canonical parent for capability-through-work architecture,
previously existing only in working history.
Adjacent, not merged: `../RESERVE-founder-due-diligence.md` (pre-relationship trust
assessment). HELIX Builders concerns organizational diagnosis and work allocation.

## Core thesis

Not a job board, freelancer marketplace, cofounder directory, or "Tinder for founders."

A governed capability-allocation and team-formation protocol in which relationships
progress from strangers to collaborators to durable teams **on evidence generated
through actual work**.

## The first-order question is not "who are you looking for?"

It is: *what is preventing this project from progressing, what evidence supports that
diagnosis, and what human capability or experiment would reduce that constraint?*

Only then does matching happen.

## 1. Bottleneck-aware matching

Distinguish **DECLARED NEED** (what the founder says) from **INFERRED NEED** (what
project evidence indicates constrains progress).

    project state → existing capabilities → available evidence → current constraints
    → unresolved bottlenecks → missing capability → recommended intervention

Common translations observed in the field:

    "we need another developer"   → delivery-capacity constraint
    "we need a cofounder"         → operational ownership constraint
    "we need marketing"           → distribution constraint
    "we need salespeople"         → qualification / positioning / pricing / conversion
    "we need a project manager"   → execution coordination / responsibility boundaries

The system must expose reasoning and evidence and allow humans to override. It never
asserts that the inferred bottleneck is objective truth.

## 2. Founder-dependency map

For each important organizational function: who owns it, whether ownership is explicit
or implicit, workload concentration, decision authority, undocumented institutional
knowledge, approval bottlenecks, single-person dependencies, execution latency.

**Founder Dependency Index (FDI)** — candidate metric over: share of critical decisions
requiring founder approval, revenue workflows touching the founder, knowledge
concentration, single-person dependencies, delegation completeness, operational
continuity when the founder is unavailable.

The objective is not to minimize founder involvement. It is to distinguish
**strategic founder control** from **accidental operational dependence**.

## 3. Intervention classification

After diagnosis, determine the *smallest appropriate* intervention rather than
defaulting to permanent hiring:

    SPECIALIST · FREELANCER · SPRINT CONTRIBUTOR · FRACTIONAL LEAD
    TECHNICAL ARCHITECT · OPERATOR · EMPLOYEE · BUSINESS PARTNER · COFOUNDER

A cofounder is one possible *result* of diagnosis, never the default starting assumption.

## 4. Bounded collaboration sprint

Instead of DISCOVERY → MATCH → COFOUNDER:

    DISCOVERY → BOTTLENECK DIAGNOSIS → CANDIDATE MATCH → BOUNDED SPRINT
    → EXECUTION EVIDENCE → COMPATIBILITY EVIDENCE → PARTNERSHIP DECISION

Duration configurable (commonly ~3–7 days); duration is not doctrine. A sprint carries
a defined problem, expected artifact, declared responsibilities, time boundary,
dependencies, decision rights, completion criteria and disposition:
CONTINUE / RESTRUCTURE / EXTEND_TEST / PAUSE / EXIT.

The objective is not free speculative labour. It is cheap evidence about whether people
can work together before expensive long-term commitments.

## 5. Collaboration receipt

Extends existing receipt doctrine (`computable-accountability.md`) rather than inventing
a parallel reputation system. Preserves: participants, objective, commitments,
contributions, artifacts, decisions, blockers raised, dependencies, handoff quality,
missed commitments, disagreement and resolution behaviour, outcome, continuation decision.

**Do not reduce this to "87% cofounder compatibility."** Prefer inspectable evidence over
pseudo-psychological certainty.

## 6. Capability-graph team formation

Do not begin from conventional titles. Model:

    problem → required capabilities → required depth → existing team capabilities
    → coverage gaps → candidate capabilities → combinations → coordination cost
    → MINIMUM VIABLE TEAM

MVT does not mean fewest humans at any cost. It means minimum sufficient capability
coverage + acceptable redundancy + manageable coordination complexity. One person may
satisfy several capability nodes.

## 7. Claimed vs observed capability

    CLAIM:     "backend engineer"
    EVIDENCE:  implemented authenticated endpoint → tests passed → review accepted
               → latency target met → artifact preserved → receipt generated

Accumulates as CONTRIBUTOR → CAPABILITY → ARTIFACT → CONTEXT → OUTCOME → RECEIPT.

## 8. AI-augmented work unit

Emerging labour primitive: capability is no longer human hours alone. An employer may
provide a worker with a governed machine workforce — coding/research/testing agents,
compute and API budget, repository permissions, execution authority, review obligations.

    Worker → Role → Human capabilities → Assigned agents → Tool permissions
    → Compute budget → Data permissions → Repository scope → Execution authority
    → Review requirements → Deliverables → Receipts → Outcome

Human accountability principle: *"the AI wrote it"* must not become responsibility
laundering. Preserve agent produced → human inspected → evidence generated → authorized
party approved → execution occurred → outcome measured.

Do not assume AI augmentation implies lower compensation. A capable human coordinating
substantial machine leverage may create more value and warrant more.

Market signals to keep watching: employer-provided model access; explicit agent budgets;
candidates expected to supervise agents; smaller human teams with larger machine
capacity; outcome-based compensation; agent-use disclosure requirements; repository and
production permissions tied to agents.

## 9. Independent technical owner role

Stronger than "prompt engineer," different from "developer who uses AI": understands
inherited systems, establishes architecture, decomposes work, directs developers and
agents, reviews implementations, debugs difficult failures, validates releases, owns
technical continuity.

## Entry/exit symmetry

ENTRY GATE — can this person demonstrate useful collaboration before receiving durable
trust, authority or organizational commitment?

EXIT GATE — has their knowledge, work, access, dependencies and successor context been
transferred before departure is operationally complete?
See `contributor-continuity-handoff-gate.md`.

## Relationship to existing canonical reserves

- `governed-work-attribution.md` — cross-cutting stage/attribution primitive HELIX
  Builders consumes; deliberately not owned here, since WIRE, ShiftTrust and agent
  teams consume it too.
- `contributor-continuity-handoff-gate.md` — departure and replacement continuity.
- `../RESERVE-founder-due-diligence.md` — pre-relationship trust; adjacent layer.
- `organizational-state-transition-governor.md` — durable admission/departure as
  organizational events.
- `opportunity-intelligence-evaluation-engine.md`, `operational-workflow-discovery-engine.md`
  — opportunity and workflow discovery upstream.

## Safety / governance

Must not: claim an inferred bottleneck is objective truth; secretly rank people by
opaque personality judgments; make employment decisions autonomously; infer protected
or sensitive personal characteristics; equate activity with competence; treat one failed
sprint as permanent evidence of inability; expose private collaboration receipts without
authorization; automatically determine equity allocation or create legal partnership.

## Activation

Reserve only. Revisit when HELIX Builders becomes an active development priority, or
when another active project creates concrete need for capability-through-work,
bottleneck diagnosis, or governed multi-contributor formation.

RESERVED. NO ACTIVE BUILD. NO NEW REPOSITORY. NO NEW PRODUCT.
