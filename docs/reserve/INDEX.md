# Reserve Index

Table of contents for the reserve architecture — not a reserve itself.
Generated 2026-07-25. Maintain per IMPLEMENTATION_STATUS doctrine rule 1
(update in the same session that adds or changes a reserve). Machine-readable
sibling: `reserve-index.csv`.

## Taxonomy

Every reserve is one of three kinds:

- **Module** — a concrete system (what exists / will exist). E.g. HELIX, IAM,
  VERITY, OROS, Backpack, API Connect, Commerce Sniper.
- **Layer** — an execution stage or evaluation surface (where it operates
  within the lifecycle). E.g. Context Classification, Economic Attribution,
  Timeline, Routing, Capital Governance.
- **Doctrine** — a rule that constrains how the ecosystem evolves (how it
  should grow). E.g. AI-Era Moat, SURVIVOR, Organizational Separation,
  Ownership vs Execution Rights.

**Layer** column uses the Observation→Strategic Moat sequence nodes
(see `meta-architecture-observation-to-moat.md`) where applicable, so the
index and the capstone stay consistent. **Status:** Doctrines are *Canonical*
(in force now as evaluation lenses); Modules/Layers are *Reserved* (architecture
only, no implementation). This is a first-pass taxonomy — refine as contributors
converge.

## Canonical reserves (iam-integrity/docs/reserve/)

| Reserve | Type | Layer / node | Status | Depends on |
|---|---|---|---|---|
| Meta-Architecture: Observation → Strategic Moat | Doctrine | Global (index) | Canonical | All |
| SURVIVOR Doctrine | Doctrine | Global (Endurance) | Canonical | All |
| AI-Era Moat Doctrine | Doctrine | Global (Strategic Moat) | Canonical | Universal Execution Timeline |
| Ownership Proofs vs Execution Rights | Doctrine | Admissibility | Canonical | — |
| Organizational Separation Doctrine | Doctrine | Global (org) | Canonical | — |
| Proof Before Promotion | Doctrine | Opportunity Evaluation | Canonical | VERITY |
| Recalculation Doctrine | Doctrine | Execution / Routing | Canonical | — |
| HELIX Universal Execution Lifecycle | Doctrine | Execution (all stages) | Canonical | — |
| Operational Workflow Discovery Engine | Module | Workflow Discovery | Reserved | VERITY, IAM, OROS, DRIFT, Commerce Sniper |
| Opportunity Intelligence & Evaluation Engine | Module | Opportunity Evaluation | Reserved | VERITY, OWDE, IAM |
| Commerce Sniper | Module | Workflow Discovery (commerce) | Reserved | VERITY, OROS |
| Hidden Asset Discovery Engine | Module | Workflow Discovery | Reserved | VERITY |
| Capital Admissibility Framework | Layer | Admissibility (capital, project) | Reserved | VERITY, OROS |
| Domain-Aware Capital Intelligence (DACI) | Module | Opportunity Evaluation (capital) | Reserved | Opportunity Intelligence & Evaluation Engine, VERITY |
| Governed Capital Eligibility (GCE) | Layer | Admissibility (capital, execution) | Reserved | vLOID, VERITY, SURVIVOR, Shield Router |
| Future Rights Exchange | Module | Admissibility / settlement (rights) | Reserved | vLOID |
| HELIX Execution Ecosystem | Module | Execution | Reserved | IAM, VERITY, OROS |
| HELIX Exchange Layer | Layer | Match / Route (capability) | Reserved | HELIX Lifecycle, API Connect |
| Execution Placement Engine | Module | Route (placement) | Reserved | — |
| Adaptive Execution Layer | Layer | Execution / Learning | Reserved | evidence gates |
| Universal Execution Interface | Layer | Execution surface | Reserved | IAM, VERITY, vLOID, OROS |
| Execution Assurance Layer | Layer | Execution completion | Reserved | Universal Money Router, HELIX, Shield Router, VERITY |
| HelixShield Execution Governance | Module | Verify (autonomous cyber) | Reserved | IAM, VERITY, vLOID, Universal Execution Timeline |
| Continuous Adversarial Security Graph | Module | Verify (security) | Reserved | VERITY |
| Continuous Security Receipts | Layer | Receipts (security) | Reserved | Receipts substrate |
| Flow Economics Engine | Layer | Economic Attribution | Reserved | HELIX, OROS, Receipts |
| Universal Execution Timeline | Layer | Timeline (execution memory) | Reserved | Receipts |
| Universal Timeline & Semantic Index Engine | Module | Media indexing (shared infra) | Reserved | — |
| VYRE / VYREL Evolution | Module | Post-execution (package / sign) | Reserved | Universal Execution Timeline, Receipts |
| Regime Evidence Engine | Module | Learning (evidence) | Reserved | VERITY |
| Agent DNS Discovery Layer | Layer | Discover (agents) | Reserved | — |
| Autonomous Connectivity Exchange | Module | Match / Route (connectivity) | Reserved | — |
| AI Internet Protocol | Module | Infrastructure | Reserved | — |
| API Trust Exposure Model | Doctrine | Governance (trust) | Canonical | VERITY |
| HELIX Project Lifecycle Exchange | Module | Coordinate (projects) | Reserved | HELIX |
| Hanoi Planner | Module | Coordinate (planning) | Reserved | — |
| HumanOS Personal Operating System | Module | Execution surface (personal) | Reserved | IAM |

## Cross-repo module reserves

| Reserve | Repo · location | Type | Layer / node | Status | Depends on |
|---|---|---|---|---|---|
| Enterprise Knowledge Integrity | api-connect · RESERVE.md §11 | Layer | Verify (knowledge) | Reserved | API Connect, VERITY, Execution Governance, Receipts |
| Capability Intelligence Layer (Phase B/C) | api-connect · RESERVE.md §10 | Layer | Route (capability telemetry) | Reserved | API Connect router |
| Backpack Research Operating System | backpack-engine · RESERVE.md §1 | Module | Learning (research) | Reserved | vLOID, OROS, HELIX, Proof Before Promotion |

## Reading the dependency column

"Depends on" lists the upstream modules/layers a reserve consumes, not a
build order. Doctrines marked "All" are cross-cutting lenses applied to every
reserve rather than runtime dependencies. For the end-to-end progression that
threads these together, see `meta-architecture-observation-to-moat.md`.
