# Reserve Index

> **INDEX COVERAGE WARNING — 2026-08-27.** This index is not currently exhaustive.
> A reconciliation audit found **43 pre-existing canonical reserve files absent from this
> index before the Aug-27 batch** (81 files in `docs/reserve/` against 38 indexed rows).
> New entries from the Aug-27 batch are recorded below, but **absence from this index MUST
> NOT be interpreted as absence of a canonical reserve.** Full index reconciliation remains
> a separate evidence-backed task — see `staging/reserves-2026-08-27.md`.


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

## Aug-27 batch additions (2026-08-27)

Not merged into the table above, which predates them and is itself incomplete — see the
coverage warning. Twenty-two canonical reserves plus one top-level SoundKeep reserve.

| Reserve | Type | Layer / node | Status | Depends on |
|---|---|---|---|---|
| Counterfactual Execution Governor (incl. embodied branch) | Layer | Admissibility (pre-execution consequence) | Reserved | vLOID, VERITY, DRIFT, OROS |
| HELIX Builders | Module | Match / Route (capability) | Reserved | VERITY, IAM, OROS, Governed Work Attribution |
| Governed Work Attribution | Layer | Economic Attribution / Execution | Reserved | Receipts, VERITY, IAM |
| Contributor Continuity / Handoff Gate | Layer | Continuity (organizational + agent) | Reserved | IAM, OROS, VERITY |
| Organizational State Transition Governor | Module | Coordinate (organization) | Reserved | vLOID, IAM, OROS, VERITY |
| Evidence Lifecycle State & Provenance Envelope | Layer | Evidence (temporal state) | Reserved | VERITY, Information Admissibility |
| Agent Metacognition & Calibration Layer | Layer | Learning (reflective control) | Reserved | VERITY, DRIFT, Receipts |
| Repository Execution Intelligence / AAG | Module | Verify (software architecture) | Reserved | vLOID, IAM, VERITY, OROS |
| IAM External Identity-Risk Signal Ingestion | Layer | Admissibility (identity) | Reserved | IAM, VERITY, vLOID |
| Adaptive Infrastructure Topology | Module | Coordinate (physical infrastructure) | Reserved | vLOID, DRIFT, VERITY, KONIGO |
| Emerging Product Pain-Loop Intelligence | Doctrine | Opportunity Evaluation | Canonical | VERITY, Opportunity Intelligence |
| Execution Jurisdiction Gap | Layer | Admissibility (institutional) | Reserved | vLOID, VERITY, IAM, DRIFT |
| Transaction-Gap Financing Primitive | Layer | Admissibility (capital gap) | Reserved | VERITY, vLOID, Capital Admissibility |
| Default-State Admissibility / Inaction Semantics | Doctrine | Admissibility (intent provenance) | Canonical | vLOID, VERITY |
| Instrument Admissibility Envelope | Layer | Admissibility (financial instrument) | Reserved | vLOID, VERITY, DRIFT |
| Executable Asset Semantics | Layer | Admissibility (asset lifecycle) | Reserved | Ownership vs Execution Rights, VERITY |
| Extraordinary Claim Evidence Tree | Layer | Evidence (claim evaluation) | Reserved | Proof Before Promotion, VERITY |
| Prospective Claim Commitment | Layer | Receipts (prospective) | Reserved | Evidence Commitment & Anchoring, VERITY |
| Intelligence Resource Governance Layer | Layer | Execution (intelligence resources) | Reserved | Model Intelligence Router, SIR, Execution Economics, Context Integrity |
| Attention Value Intelligence | Module | Learning (attention) | Reserved | Revealed Preference Measurement, VERITY |
| vLOID Collaborative Validation Doctrine | Doctrine | Global (validation) | Canonical | vLOID |
| SportGPT Intelligence Layer | Module | Opportunity Evaluation (sport) | Reserved | EventPulse (staging), VERITY, Prospective Claim Commitment |
| SoundKeep Intent-to-Patch (docs/soundkeep-intent-to-patch-reserve.md) | Module | Knowledge to Capability (audio) | Reserved | SoundKeep, VKOS, VERITY |

---

## Batch 2026-08-28 — added reserves

The coverage warning above still applies: **this index remains explicitly incomplete.**
These rows record the Batch 2 additions only. They do not close the outstanding deficit
between indexed rows and files present in `docs/reserve/`.

| File | Type | Note |
|---|---|---|
| `physical-system-boundary-accounting.md` | Doctrine | energy-boundary accounting + causal reconstruction; no anomalous-physics claim |
| `protocol-independent-capability-envelope.md` | Layer | semantic capability normalization; transport defers to EAF, stack to UEI |
| `executable-capacity-thinnest-leg.md` | Module | owns thinnest-leg; consumed by DSMI and UDEA |
| `temporal-evidence-admissibility.md` | Layer | child of the evidence-lifecycle envelope |
| `demand-sovereign-market-infrastructure.md` | Doctrine | DSMI |
| `underwater-duration-edge-decay-admissibility.md` | Module | Momentum Sniper / JANUS measurement primitive |
| `loyalty-value-routing.md` | Module | recovered direction; conditional-value routing, bounded to loyalty |
| `architecture-triage-service.md` | Doctrine | service opportunity; no software build |
| `human-fairness-dignity-accountability-institute.md` | Doctrine | HFAI institutional design |
| `human-machine-sovereignty-boundary.md` | Doctrine | non-delegable authority only |

Extensions were appended in place to `computable-accountability.md`,
`proof-before-promotion.md` and `ownership-proofs-vs-execution-rights.md`; they did not
create new index rows because they did not create new reserves.

Batch record: `docs/reserve/staging/reserves-2026-08-28.md`
