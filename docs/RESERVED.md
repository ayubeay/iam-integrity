# RESERVED.md

Architectural primitives discussed but deliberately not implemented.
Each entry has an explicit activation condition.

Discipline rule: ideas live here so they are remembered, not built
prematurely. The same May 6 lesson applies as in /opt/momentum:
conversation-time plausibility is not the same as forward validation.

---

## OROS scope enforcement

**What it is:** OROS reads `scopes/{scope_id}.json` and gates per-action
calls against permissions and constraints. Currently scope contracts
exist on disk and are bound to agents, but no service reads them at
action time.

**Activation condition:** Sonic v1 (or any agent) produces enough
recommendation events that quota enforcement, permission gating, and
amendment policy actually have observable effects. Until then,
enforcement has no surface to act on.

**Why deferred:** Without observable agent activity, "enforcement" is
indistinguishable from "no action ever attempted." Build OROS scope
enforcement when there are real attempted actions to observe enforcement
against.

---

## OROS ORA enforcement

**What it is:** OROS reads `oras/{ora_id}.json` and applies constraint
rules (`require_scope_alignment`, `no_deceptive_output`, etc.) and
enforcement decisions (BLOCK / ADJUST / ADJUST+TRANSITION) during action
evaluation.

**Activation condition:** OROS scope enforcement landed first (since ORA
enforcement piggybacks on the same per-action gating point), AND there
are agent actions whose ORA-compliance can be evaluated in a non-trivial
way.

**Why deferred:** ORA enforcement against a single agent in seed state
with no outbound activity has nothing to enforce against.

---

## JANUS dual-read validator

**What it is:** Adversarial reinterpretation layer. For every action
evaluated by ORA, JANUS performs a second pass that interprets the same
action under different assumptions and checks for divergence.

**Activation condition:** sufficient frequency of governance decisions
to make adversarial validation statistically meaningful. Probably 100+
governance events per day before JANUS produces signal rather than noise.

**Why deferred:** Adversarial validation against a low-volume system
mostly produces false positives. JANUS is a primitive for systems
already at production behavioral density.

---

## ora_default_v2 with janus_rules

**What it is:** Successor ORA contract that includes `janus_rules`
section, enabling agents to be bound to dual-read validation.

**Activation condition:** JANUS validator exists first. ora_default_v2
without JANUS is just a versioned name change with no operational effect.

**Why deferred:** Hard precondition. Listed here so the chain is visible.

---

## Receipt chaining (prior_receipt_id)

**What it is:** Each receipt includes a `prior_receipt_id` field
referencing the previous receipt for the same agent, forming a verifiable
per-agent timeline.

**Activation condition:** Receipt diversity beyond birth + recommendation.
At minimum: settlement receipts, challenge participation receipts,
transition receipts, and ≥5 active agents producing varied event types.

**Why deferred:** Chaining two receipt types on one agent is design
overhead with no operational benefit. The optimal chain shape becomes
clear from observed activity, not from speculation. Designing chain
semantics now means committing to a structure before knowing what
real activity flow looks like.

---

## Off-volume backup

**What it is:** Scheduled snapshot of `agents_index.json` and
`integrity_trail.jsonl` from the Railway volume to S3-compatible
storage, with a tested restore procedure.

**Activation condition:** Production agent count crosses a threshold
where the loss of the volume would represent unrecoverable damage. At
current scale, loss would still be operationally recoverable by
re-minting, but the May 6 wipe demonstrated that continuity claims can
still be permanently broken without backup. The threshold is not
"backup becomes meaningful" — it became meaningful on May 6. The
threshold is "the cost of building backup is now lower than the cost of
the next failure."

**Why deferred:** Backup against a one-agent registry is theatrical.
Plus, designing backup before understanding the access patterns of the
data would commit to a structure (e.g., snapshot frequency, restore
granularity) without information.

**However:** The lack of backup is the single biggest architectural risk
right now. If the Railway volume is corrupted or deleted before this is
built, all production state is permanently lost. This deferral is
deliberate but not comfortable.

---

## Multi-region / multi-replica volume strategy

**What it is:** Distribute agent registry across regions or replicas to
survive single-region failure.

**Activation condition:** Production usage that depends on geographic
availability or has uptime requirements that single-region cannot meet.

**Why deferred:** No production usage. Multi-region adds significant
operational complexity (consistency models, replica reconciliation,
write-coordination) that's premature against current load.

---

## LITMUS auto-ingestion pipeline

**What it is:** Each monitored system (VERITY, IAM, SURVIVOR, GATE,
GHOSTLEDGER, RENTRECLAIM) emits decisions to LITMUS automatically as
they occur, replacing the current manual transport.

**Activation condition:** None — this is buildable now. Listed here as
deferred not because preconditions block it, but because the build queue
prioritizes documentation and durability validation first.

**Why deferred (at time of writing):** Sequencing — document validated
state and stress current durability before extending behavioral density.
Should leave this list and become a build candidate after that
sequencing.

---

## LITMUS_OBSERVER governed agent

**What it is:** LITMUS itself becomes a minted agent with its own scope
contract (read-only across systems, no write, no action initiation) and
ORA contract (`no_deceptive_summary`, `traceable_observation_required`,
etc.).

**Activation condition:** ≥5 active governed agents that LITMUS routinely
observes, AND OROS scope enforcement exists (so that LITMUS_OBSERVER's
read-only scope is actually enforced rather than declared).

**Why deferred:** Right now LITMUS observes effectively zero live
governed agents (because ingestion is manual). Making LITMUS itself a
governed agent before there's anything for it to observe is reversed.

---

## Music-domain VERITY scoring

**What it is:** VERITY scoring weights specific to music agents:
`recommendation_made`, `recommendation_played`, `prediction_correct`,
`taste_transition_detected`, `long_term_alignment_score`. Replaces the
current debate-shaped scoring approximately applied to music agents.

**Activation condition:** Real user interaction with Sonic exists,
producing observable signals (`played`, `skipped`, `predicted_X
happened`). Without these, the new scoring weights have no input.

**Why deferred:** Designing scoring weights against zero feedback signal
means committing to weight definitions before knowing what kinds of
events will actually be observable.

---

## SoundKeep UI

**What it is:** Consumer-facing surface where users see their assigned
Sonic agent and interact with it. Shows current taste state, recent
recommendations, drift visualization, reasoning receipts.

**Activation condition:** Sonic v1 in production producing real
recommendation receipts. Currently: Sonic v1 deployed, has produced
recommendations during testing, but has not been used by any actual user.

**Why deferred (slightly):** Could ship a minimal UI shell now. Was
sequenced after durability and documentation work. Should leave this
list once those land.

---

## Skip and interaction tracking

**What it is:** Extend listen schema to capture `skipped`, `repeat`,
`paused`, `liked`, `played_to_completion` events as inputs to the
taste-state engine.

**Activation condition:** SoundKeep UI exists and is being used (so
that real interaction events have a source).

**Why deferred:** No source of interaction events without a UI.

---

## Exploration mode for Sonic

**What it is:** Sonic recommends tracks outside the user's current taste
cluster, gated by an explicit user opt-in.

**Activation condition:** Sonic v1 in steady use, with users who have
sufficient taste-state data that "outside their cluster" is a meaningful
concept.

**Why deferred:** Exploration mode against a brand-new user with no
taste history would just be random recommendation. Not useful and
wouldn't demonstrate exploration as a primitive.

---

## Receipt schema versioning

**What it is:** Formal version field on every receipt type, with a
migration story for receipts created under earlier versions.

**Activation condition:** A breaking change to receipt structure becomes
necessary. So far the structure has been additive.

**Why deferred:** Versioning before there are any version-1 → version-2
migrations to handle is design overhead.

---

## Per-event activity decay

**What it is:** Reputation/integrity scores decay over time when an agent
is inactive, preventing stale-but-trusted agents from accumulating
unwarranted standing.

**Activation condition:** Agents have observable activity that can be
measured for "recency." Currently no activity timestamps beyond mint.

**Why deferred:** Decay against an essentially-static population is
just slow drift toward zero with no information content.

---

## CYCLE142857 — deterministic cyclic sequencing primitive

A small Python module exposing rotations of the cyclic number 142857
(from 1/7 = 0.142857...) as deterministic ordering, rhythm emphasis,
and state-cycling helpers. The math is real (cyclic number theory);
the framing is generative/mathematical only — no mystical, predictive,
or execution-authority claims.

**Activation condition:** SoundKeep v1 or Sonic runtime develops a
concrete subsystem need for deterministic ordering, repeatable
sequencing, procedural rhythm generation, or sync-safe queue cycling.

**Why deferred:** No current subsystem needs cyclic ordering. Adding
the primitive to the repo without a consumer would be infrastructure
without a consumer — exactly the pattern this document is designed
to prevent. A `[1,4,2,8,5,7]` constant is trivially recoverable when
the consumer appears.

---

## LeadScan v2 — outreach + identity infrastructure reframe

LeadScan v1 sends cold emails to scraped Shopify domains. LeadScan v2
reframes the same pipeline as relationship + workflow intelligence
infrastructure: stack-aware messaging, lifecycle inference, signed
outbound receipts, structured participation memory, identity-verified
sender reputation. Same scraping core, different output product.

**Activation condition:** LeadScan v1 produces evidence about what
actually moves enterprise reply rates — specifically, ≥1 engaged
human reply traced to a specific message variant, target type, or
sender signal. Reframe is informed by behavior, not strategy.

**Why deferred:** v1 has sent ~30 emails, received only autoreplies
(e.g. Biodroga). Reframing v2 before understanding why v1 didn't
convert risks building a more elaborate version of the same failure
mode. The bottleneck is message/target/conversion path, not
architectural sophistication.

---

## Track provenance receipts

Extend the existing signed-receipt pattern (birth, sonic_recommendation)
to track-level events: upload origin, license scope, allowed usage,
remix permission, distribution targets, monetization split. Each
track interaction emits a signed receipt with the same vyre_v1
substrate that birth and recommendation receipts use.

**Activation condition:** A consumer of track-level receipts exists.
Concretely: SoundKeep v1 has a track-upload or track-organization
surface, OR a DJ workflow produces interaction events that would
benefit from signed lineage.

**Why deferred:** No track-level events currently flow through any
production system. Designing the receipt schema before knowing which
events matter (upload? download? remix? play? skip?) would commit to
a structure prematurely. Receipt schema discipline says: wait for
real events.

---

## DJ reputation graph

A reputation primitive for DJs/curators based on observed behavior:
track discovery success, replay value, transition quality, crowd
retention, originality, attribution honesty. Conceptually similar
to VERITY scoring but specialized for music-curation behavior.

**Activation condition:** SoundKeep v1 has ≥10 active DJs producing
real interaction data (session history, transition patterns, crowd
response signals) over a meaningful time window.

**Why deferred:** Reputation against zero observed behavior is
theoretical. The hard problem isn't designing the score; it's having
behavior to score. Same failure mode as Music-domain VERITY scoring
(see above) — listed separately because the inputs are DJ-specific
rather than listener-specific.

---

## Licensing-aware recommendations

Sonic recommendations that include rights metadata per track: can
it be legally downloaded, remixed, redistributed, used commercially,
used in a live set, streamed, monetized? Recommendation receipts
extend to carry "rights traveling with the recommendation."

**Activation condition:** A licensing data source exists (either a
third-party rights API, or a creator-uploaded license tagging system
within SoundKeep). Without one, "licensing-aware" reduces to "guessing
about rights," which is worse than no claim.

**Why deferred:** Rights data is the hard problem, not the receipt
schema. Implementing licensing-aware recommendations against unknown
rights would produce receipts that look authoritative but aren't
grounded. Same discipline as the Ed25519 work — claims need evidence,
including legal claims.

---

## OOBE / SAP autonomous-agent bounty (full workflow build)

SURVIVOR is live on SAP mainnet as an agent (GTZNpo…Af5hx,
did:sap:survivor-execution-agent, active), with the risk-screen tool
published, stake funded, pricing set, and a funded escrow (9zK9…uw3r).
The setup chain works end-to-end. What's deferred is the *autonomous
workflow* the bounty actually rewards: trigger → execution → payment
with no manual steps, Synapse RPC in the execution path, Synapse
Sentinel used at least once, a real AI capability in the loop, and
enough settled escrow volume to rank.

Settlement architecture was mapped empirically: SettlementSecurity has
three modes — SelfReport, CoSigned, DisputeWindow. The escrow was built
as SelfReport (dispute_window 0, no co-signer). create_pending_settlement
rejects SelfReport (InvalidSettlementSecurity 6099); direct settle_calls_v2
on SelfReport returns InvalidAccount 6089 at escrow_v2.rs:333. The
self-serviceable settlement path is a *new* DisputeWindow escrow
(window > 0) settled via pending → finalize; CoSigned needs a second party.

**Activation condition:** OOBE/SAP becomes a confirmed cashflow edge —
either a re-opened bounty window with realistic runway, a paid OOBE
engagement, or the SDK/IDL drift findings convert into a relationship
worth building on. The autonomous-loop build is the bet; protocol
plumbing is done.

**Why deferred:** The bounty is volume-ranked on autonomous settled
activity, not setup work. With ~4 days left, cracking one manual
DisputeWindow settlement wouldn't make the agent competitive against
entrants who built the full autonomous stack. Per doctrine — "model is
table stakes, workflow is the bet" — and this is a workflow build behind
an unconfirmed edge. Detailed receipts and SDK/IDL findings are in
docs/OOBE_BOUNTY_ESCROW_CHAIN_2026_05_30.md. The agent stays live on
mainnet; resume from autonomous-workflow requirements first, not plumbing.

---

## HELIX Execution Ecosystem (full reserve document)

**What it is:** The complete platform-architecture reserve for HELIX as a
truthful execution environment: Marketplace (discovery), Workspace
(collaboration), Execution Rooms (governed work unit), Helix Verified
(execution-based identity), Contribution Economy, Agent Collaboration
Layer, and the module relationships (RACER, HELIXCAN, HelixAtlas,
HelixMeter, Helixwap, HelixShield, vLOID/VERITY/IAM). Includes phased
adoption sequence (Phase 0 internal proof through Phase 5 enterprise),
explicit non-goals, initial wedge candidates, and ethical execution
doctrine (10 principles).

Canonical document: **docs/reserve/helix-execution-ecosystem-reserve.md**

Independent Level 2 global architecture reserve. No Momentum Sniper
dependency; no live-trade prerequisite; grants no execution authority.

**Activation condition:** Tier 1 products (API Connect, SoundKeep, POI
Engine) generate the internal proof artifacts (deployments, receipts,
verification records) required for Phase 0, AND an initial wedge is
selected per section 25 of the reserve document.

**Why deferred:** Launching an empty network inverts the adoption
sequence. The marketplace derives its identity from shipped products
("built by people who actually shipped AI systems"). Per Hexagram 5
(Waiting, Jul 14 oracle): recording the architecture now while
implementation waits for the right cashflow, wedge, and execution window.

---

## HELIX Project Lifecycle Exchange (canonical module reserve)

**What it is:** HELIX Marketplace module treating projects as living
execution assets moving through observable lifecycle states (Idea ->
Building -> Production -> Dormant -> Revived -> Archived), with immutable
receipts on every transition, evidence-based resurrection (cofounder joins,
funding milestone, customer prepays), and a dormant-project exchange
searchable by execution readiness. "Execution paused," never "project
died." Canonical document: **docs/reserve/helix-project-lifecycle-exchange.md**

**Activation condition:** HELIX Workspace operational + active Marketplace
builder community + mature receipts + VERITY/IAM integrated + lifecycle
transitions authenticatable by execution evidence.

**Why deferred:** Extends a platform that is itself reserved (HELIX
ecosystem reserve). Not an MVP feature by its own definition.

---

## Continuous Adversarial Security Graph (canonical module reserve)

**What it is:** HelixShield/HelixAtlas/vLOID extension: living attack
surface graph of every execution entry point, attack surface receipts per
deployment, assumed-breach execution mode with containment receipts,
continuous adversarial agents (ADVERSARY-X, INSIDER, SUPPLYCHAIN,
PROMPT-INJECTOR, ...), dormant infrastructure discovery, security impact
forecasting, blast-radius visualization with replay. Canonical document:
**docs/reserve/continuous-adversarial-security-graph.md**

**Activation condition:** HelixShield exists as a real policy engine with
execution receipts flowing; a production system whose attack surface
justifies continuous modeling.

**Why deferred:** Extends unbuilt HelixShield infrastructure. The doctrine
(security as observable, receipt-driven, simulation-backed) is captured;
implementation waits for the substrate.

---

## Jul 16 evening canonical batch — protocol + product reserves (index)

Seven complete formal reserve documents promoted directly to canonical.
One-line index; each file carries its own activation doctrine:

- **AI Internet Protocol (AIP)** — docs/reserve/ai-internet-protocol.md.
  Protocol-layer research: identity, discovery, trust negotiation,
  receipts, reputation, settlement for AI-to-AI interaction. The missing
  question after DNS/OAuth/OpenAPI: "can I trust this autonomous system to
  act on my behalf?" HELIX may implement, must not define.
- **Agent DNS / AI Discovery Layer** — docs/reserve/agent-dns-discovery-layer.md.
  agent.manifest.json + discovery records + AI passports + domain
  reputation, atop existing DNS. One component of AIP.
- **Continuous Security Receipts (CSR)** — docs/reserve/continuous-security-receipts.md.
  Every security action (human or AI) produces signed receipts; continuous
  compliance over periodic assertion. Sibling of the Adversarial Security
  Graph reserve.
- **HumanOS** — docs/reserve/humanos-personal-operating-system.md.
  INDEPENDENT long-term product: person as platform, identity vault,
  permission ledger, consent center, AI workforce, reputation passport.
  Optional HELIX integration; never a HELIX module.
- **HANOI Planner** — docs/reserve/hanoi-planner.md. Constraint-aware
  recursive execution planning inside vLOID/OROS: minimum VALID path,
  temporary-state doctrine, critical-transition protection, planning +
  execution receipts, failure as state transition. Activates when OROS
  needs real multi-step autonomous planning.
- **Autonomous Connectivity Exchange (ACE)** — docs/reserve/autonomous-connectivity-exchange.md.
  Connectivity as programmable execution marketplace above KONIGO Connect:
  execution contracts, continuous provider scoring, connectivity receipts.
- **Future Rights Exchange** — docs/reserve/future-rights-exchange.md.
  Programmable future ownership infrastructure (vesting first vertical):
  future-rights registry, vesting intelligence, receipt-native settlement.

Staging additions same session (reserves-2026-07-16.md items 14-15):
Founder Attention & Execution Allocation Layer; AI Capability Registry.

---

## Jul 17 canonical batch — doctrine + placement reserves (index)

Eight formal documents promoted directly to canonical:

- **Zircon** — docs/reserve/zircon.md. Engineering and scientific
  knowledge layer: dependency intelligence, future-dependency posture per
  product, three subordinate research programs, engineering receipts.
  Promoted from staging 2026-08-07.
- **Hidden Asset Discovery Engine** — docs/reserve/hidden-asset-discovery-engine.md.
  Zircon capability: secondary outputs (waste heat, idle compute, empty
  routes, unused data) surfaced as economic opportunities before new
  investment. Includes HelixHash Router lineage cross-reference.
- **Recalculation Doctrine** — docs/reserve/recalculation-doctrine.md.
  Constitutional principle: temporary route failure ≠ destination failure;
  recalculate, don't loop. Governs OROS/KONIGO/HelixAtlas/VERITY/IAM/DRIFT.
- **Adaptive Execution Layer** — docs/reserve/adaptive-execution-layer.md.
  Stability is the default, adaptation is earned: evidence-gated, governed,
  receipted behavior change. Four states: stable / observation / candidate
  / approved.
- **Proof Before Promotion Doctrine** — docs/reserve/proof-before-promotion.md.
  Evidence ladder (idea -> hypothesis -> prototype -> benchmark ->
  independent verification -> production -> track record); claims inherit
  the highest COMPLETED stage. "Show me the receipt."
- **API Trust & Exposure Model** — docs/reserve/api-trust-exposure-model.md.
  APIs as execution boundaries; governance separated from transport; future
  classes: Trusted / Agent / Receipt / Policy / Continuity APIs.
- **Capital Admissibility Framework** — docs/reserve/capital-admissibility-framework.md.
  Capital as governed execution resource; milestone -> verification ->
  release -> receipt; execution precedes valuation.
- **Execution Placement Engine** — docs/reserve/execution-placement-engine.md.
  "Execution follows admissibility" generalizing "compute follows energy";
  placement as multidimensional mission optimization.
- **Regime Evidence Engine** — docs/reserve/regime-evidence-engine.md.
  Diagnose WHY performance changed (regime shift vs edge decay vs execution
  degradation vs data anomaly); generalizes MomentumSniper's manual
  criterion discipline.

Staging (reserves-2026-07-17.md): HelixHash Router reinforcement +
history-recovery action; Sonic Discovery Confidence Engine (routed to
SoundKeep reserved architecture); doctrine-trio linkage note (candidate
consolidated Execution Doctrine under LITMUS).

---

## Commerce Sniper (canonical reserve)

**What it is:** Verified commercial-opportunity and execution engine for
real goods and services — demand gap -> source -> verify -> buy -> route
-> sell -> settle. Six-part engine (scanner, demand verifier, VERITY
counterparty trust, margin/route engine, bounded negotiation agent, OROS
settlement), graded recommendations (OBSERVE...REJECT), commercial
receipts for successes AND failures, hard doctrine (verified demand before
capital, net margin not headline spread, no autonomous purchase early,
escrow thresholds, capital preservation first). Phase Zero is a
paper-commerce observation engine with pre-committed accuracy metrics.
Canonical document: **docs/reserve/commerce-sniper.md**

**Activation condition:** Tier 1 production maturity + a chosen narrow
first category with verifiable demand signals + Phase Zero observation
window designed with pre-committed gates (Proof Before Promotion).

**Why deferred:** Extends the execution stack into physical commerce —
requires Universal Money Router, escrow, and mature VERITY counterparty
scoring that do not exist yet. The doctrine is captured; the market will
still be inefficient later.

---

## Universal Execution Interface (canonical reserve)

**What it is:** One reusable execution interface across all products —
"conversation is only the interface; execution is the product." Every
product becomes an execution surface over the same stack (Intent -> IAM ->
VERITY -> vLOID -> OROS -> Domain Engine -> HELIX -> Receipt), with one UI
pattern (accomplish -> understand -> recommend -> approve -> execute ->
receipt) and receipts for denials as first-class outcomes. Includes the
observation/execution BOUNDARY DOCTRINE (Helius vs HeliusSwapProvider vs
HELIX naming distinction; data layers never gain execution authority;
Phase B shims never execute through HELIX). Doctrine: build one execution
infrastructure, expose many execution surfaces. Canonical document:
**docs/reserve/universal-execution-interface.md**

**Activation condition:** vLOID governance mature + HELIX rails
production-ready + at least two domain products sharing common execution
patterns.

**Why deferred:** The layer sits above five subsystems of which several
are themselves reserved. The doctrine (especially the boundary rule) is
binding NOW for design decisions; the interface build waits.

---

## Aug 27 canonical batch — notepad-to-canon reconciliation (index)

Twenty-two formal reserve documents promoted directly to canonical, two existing reserves
extended, one top-level SoundKeep reserve, and six PROPOSED experiment definitions.
One-line index; each file carries its own activation doctrine.

Governing rule for this batch: *canonical concept ownership beats the name we happened to
use in the notepad — preserve responsibility boundaries, not names.* Session provenance and
declined placements: `docs/reserve/staging/reserves-2026-08-27.md`.

- **Counterfactual Execution Governor** — docs/reserve/counterfactual-execution-governor.md.
  Pre-execution consequence forecasting: counterfactual branches, minority-risk
  preservation, least-irreversible intervention, calibration. Includes the embodied /
  robotic-compromise safety branch — authentication proves origin, not physical safety.
- **HELIX Builders** — docs/reserve/helix-builders.md. Capability-through-work formation:
  bottleneck diagnosis before matching, founder-dependency mapping, bounded collaboration
  sprints, collaboration receipts, capability graph / minimum viable team, AI-augmented
  work units. Adjacent to RESERVE-founder-due-diligence.md, not merged into it.
- **Governed Work Attribution** — docs/reserve/governed-work-attribution.md. Cross-cutting:
  stage-gated attribution, outcome attribution graph, opportunity readiness gate,
  persistent contribution attribution, bounded responsibility. Consumed by HELIX Builders,
  WIRE, ShiftTrust and agent teams; owned by none of them.
- **Contributor Continuity / Handoff Gate** — docs/reserve/contributor-continuity-handoff-gate.md.
  Departure is a human event; continuity completion is an operational event. Extends to
  agent replacement and version migration.
- **Organizational State Transition Governor** — docs/reserve/organizational-state-transition-governor.md.
  Organizational events as governed state transitions with an obligation propagation graph.
  Sits above GhostLedger; preserves the GhostLedger/ILF escalation boundary.
- **Evidence Lifecycle State & Provenance Envelope** — docs/reserve/evidence-lifecycle-state-provenance-envelope.md.
  DELETED ≠ UNAVAILABLE ≠ UNVERIFIABLE. Source observability is asymmetric; absence from a
  provider is not proof of deletion.
- **Agent Metacognition & Calibration Layer** — docs/reserve/agent-metacognition-calibration-layer.md.
  Computable self-monitoring, prediction receipts, calibration, epistemic yield. Explicitly
  not chain-of-thought storage and not a claim of machine consciousness.
- **Repository Execution Intelligence / AAG** — docs/reserve/repository-execution-intelligence.md.
  Retrieval ≠ understanding ≠ architectural understanding ≠ authorization ≠ verified
  execution. "Unused in this repository" is not "unused."
- **IAM External Identity-Risk Signal Ingestion** — docs/reserve/iam-external-identity-risk-signals.md.
  Authentication establishes who is presenting an identity; risk intelligence informs how
  much authority it should currently hold. Graduated response, blast radius, recovery.
- **Adaptive Infrastructure Topology** — docs/reserve/adaptive-infrastructure-topology.md.
  Infrastructure functions reconfigured under hazard, governed by physical-world
  admissibility. Synthetic inspiration explicitly preserved as non-evidence.
- **Emerging Product Pain-Loop Intelligence** — docs/reserve/emerging-product-pain-loop-intelligence.md.
  Do not chase what founders are building; study what their existence reveals. L3–L5 depth
  test; a product launch is not validation.
- **Execution Jurisdiction Gap** — docs/reserve/execution-jurisdiction-gap.md. Discover the
  legitimate admissible route or DENY — never manufacture eligibility. Introduces the
  Execution Gap Primitives family.
- **Transaction-Gap Financing Primitive** — docs/reserve/transaction-gap-financing-primitive.md.
  Finance only the measured gap in a verified transaction with a verified repayment source.
  Research only; no capital deployment. Sibling of Execution Jurisdiction Gap.
- **Default-State Admissibility / Inaction Semantics** — docs/reserve/default-state-admissibility.md.
  Defaults may optimize execution but must not counterfeit intent. Governance friction
  budget proportional to consequence.
- **Instrument Admissibility Envelope** — docs/reserve/instrument-admissibility-envelope.md.
  Transaction value ≠ economic exposure. Distinct from Capital Admissibility Framework.
- **Executable Asset Semantics** — docs/reserve/executable-asset-semantics.md. A governed
  asset is not portable merely because ownership is; meaning, constraints, authority chain
  and lifecycle must be portable too. Zero-bilateral-integration test.
- **Extraordinary Claim Evidence Tree** — docs/reserve/extraordinary-claim-evidence-tree.md.
  Operational method under Proof Before Promotion: claim atomization, source independence,
  discriminating predictions, falsification receipts, UNKNOWN preservation.
- **Prospective Claim Commitment** — docs/reserve/prospective-claim-commitment.md. Child of
  Evidence Commitment & Anchoring. Immutability ≠ completeness; claim integrity ≠ selection
  integrity. Includes the SportGPT commitment-ledger application.
- **Intelligence Resource Governance Layer** — docs/reserve/intelligence-resource-governance-layer.md.
  Only what Model Intelligence Router, Sovereign Intelligence Routing, Execution Economics
  and Context Integrity do not own: the NO_MODEL gate, memory admission, progressive
  capability admission, total-objective-cost accounting, waste classification.
- **Attention Value Intelligence** — docs/reserve/attention-value-intelligence.md. What did
  the user actually receive from the attention they spent? Measurement separated from
  interpretation; privacy-preserving by construction.
- **vLOID Collaborative Validation Doctrine** — docs/reserve/vloid-collaborative-validation-doctrine.md.
  Collaborations as heterogeneous validation environments. vLOID adapts around legitimate
  external workflows rather than requiring collaborators to build around it.
- **SportGPT Intelligence Layer** — docs/reserve/sportgpt-intelligence-layer.md. Market
  divergence with explanation and no retrospective intelligence leakage. Divergence ≠
  opportunity. Does not promote the staged EventPulse/SportGPT material.

Extended in place: **Browser Fair Compute** (docs/browser-fair-compute-reserve.md) — scope
widened to Fair Useful Compute; Phases 1 and 2A banked as evidence, 2B–5 reserved.
**Provider Qualification & Workload-Aware Routing** (docs/reserve/provider-qualification-and-routing.md)
— Intelligence Supply Continuity and the Social-Evidence Provider Validation Case.

Top-level: **SoundKeep Intent-to-Patch / Synthesis Knowledge Layer** —
docs/soundkeep-intent-to-patch-reserve.md.

Research: six PROPOSED experiment definitions in
docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md — pointer-level only, referencing the
canonical reserves rather than duplicating them. None has been run.

**Zircon remains reserve-only. Earthwise material excluded from this batch by instruction.**

---

## Batch 2026-08-28 (Batch 2)

Fifteen accumulated candidates reconciled against commit `30e41f1`. Outcome: 10 canonical
reserves, 3 extensions to existing reserves, 4 experiment specs, 2 test specs.

**Canonical reserves**

- `docs/reserve/physical-system-boundary-accounting.md` — energy-boundary accounting and
  causal reconstruction for physical systems. **No claim of anomalous energy or new physics
  is implied.**
- `docs/reserve/protocol-independent-capability-envelope.md` — PICE. Semantic capability
  normalization across invocation protocols. *Govern the action, not the adapter.*
- `docs/reserve/executable-capacity-thinnest-leg.md` — could the opportunity actually have
  been executed at the assumed size and conditions?
- `docs/reserve/temporal-evidence-admissibility.md` — signal decay, half-life, and the five
  separated times. Child of the evidence-lifecycle envelope.
- `docs/reserve/demand-sovereign-market-infrastructure.md` — DSMI. Reusable market primitive;
  no generic marketplace-as-a-service product authorized.
- `docs/reserve/underwater-duration-edge-decay-admissibility.md` — UDEA. Distinguishing pain
  from evidence of failure. **Does not authorize any strategy parameter change.**
- `docs/reserve/loyalty-value-routing.md` — conditional-value routing under changing rules,
  bounded to loyalty and travel. **A technically executable route is not an admissible one.**
- `docs/reserve/architecture-triage-service.md` — sell technical judgment before
  implementation. No software build authorized.
- `docs/reserve/human-fairness-dignity-accountability-institute.md` — HFAI. *Defend the
  human, not the narrative.* **Do not incorporate, fundraise, investigate people, or make
  public accusations.**
- `docs/reserve/human-machine-sovereignty-boundary.md` — HMSB. Capability does not create
  authority. Names the *sovereignty laundering* failure state.

**Extensions**

- `docs/reserve/computable-accountability.md` — Human-Mediated Execution / Decision Influence
  Accountability. *Human execution does not erase machine influence.*
- `docs/reserve/proof-before-promotion.md` — Extraordinary Technology Admissibility and the
  source-credibility separation: *source credibility changes the prior confidence assigned to
  a proposition; it does not determine the proposition's truth value.*
- `docs/reserve/ownership-proofs-vs-execution-rights.md` — Knowledge Execution Rights.

**Specs** — `docs/research/experiments/` (4) and `docs/research/tests/` (2), with their own
indexes and status vocabularies. A test `PASS` is never evidence that an external-world
hypothesis is validated.

**Excluded:** SIGOME (personal ritual practice, not an architecture) · Earthwise · Deliverable
B (EVIDENCE_DISCIPLINE.md and the 2026-08-27 candidate migration). **Zircon remains
reserve-only.**

Batch record: `docs/reserve/staging/reserves-2026-08-28.md`

---

## Batch 3 — Commit A: new canonical reserves (2026-08-29)

Twenty-seven submissions were collision-checked against 106 canonical reserves. **Four
created a new reserve.** The remainder resolved to appends, dissolutions or deferrals and
are recorded with Commit B, which carries the batch record.

- `docs/reserve/decision-leverage-attention-admissibility.md` — attention as a scarce
  governed resource. *Event importance is not attention priority.* Names the silent
  degradation of human review from ATTENTIVE to CEREMONIAL, where the receipt is identical
  at every rung.
- `docs/reserve/operational-sovereignty-dependency-independence.md` — common-control
  ancestry across dependency classes. *Redundancy is not sovereignty; three providers on
  one substrate are one provider with three invoices.* `UNRESOLVED` never defaults to
  `INDEPENDENT`.
- `docs/reserve/trust-graph-provenance-credibility-laundering.md` — trust transitivity,
  credibility laundering, verification scope expansion. *Endorsement count is not
  independent trust roots.* Endorsement-side sibling of
  `docs/reserve/iam-external-identity-risk-signals.md`, which runs the same substrate in
  the opposite direction. **The Trust Independence Ratio stays a research concept; no
  formula is defined.**
- `docs/reserve/multi-embodiment-identity-authority-continuity.md` — execution leases.
  *Identity persists; authority does not travel with it.* Expiry is the default and
  renewal is an event requiring evidence. Cross-links
  `docs/reserve/human-machine-sovereignty-boundary.md` without being hosted by it, per
  that reserve's own scope constraint.

**Batch record:** lands with Commit B.

---

## Batch 3 — Commit B: extensions (2026-08-29)

Fifteen extensions were appended in place to existing reserves. Each carries a named, dated
section and a **"Why this belongs here and not in its own file"** subsection.

**Evidence and independence**

- `docs/reserve/independent-validation-capability-promotion.md` — Effective Evidence
  Multiplicity. `N_effective ≤ N_raw`, **with no formula**: the Evidence Independence Graph
  should inform the quantity from ancestry, not collapse it to a ratio.
- `docs/reserve/extraordinary-claim-evidence-tree.md` — Correction-Induced Misconception.
  *A correction is not a subtraction*; it can install a new misconception rather than
  removing an old one.
- `docs/reserve/invariant-precomputation.md` — Cognitive Artifacts as a Reuse Class.
  Reusing a judgment is not reusing a computation: its inputs can be unchanged while its
  warrant has expired.

**Execution and runtime**

- `docs/reserve/hanoi-planner.md` — Execution Bounds, Deficit Classification & Verification
  Debt. `TERMINATED_ON_BOUND` is distinct from failure; deferred verification is a
  Temporary State with an exit condition.
- `docs/reserve/execution-economics.md` — Bounded Execution and the Cost of Stopping.
  `TECHNICAL_FAILURE ≠ GOVERNED_TERMINATION` · `STOPPED ≠ FAILED`.
- `docs/reserve/intelligence-resource-governance-layer.md` — Continuation Admission &
  Self-Constructed Capabilities. A capability the agent builds is admitted by the same gate
  as one the system exposed, or not at all.
- `docs/reserve/helixshield-execution-governance.md` — Governance Distance. *A governance
  mechanism that can only instruct has not governed.* The nine-level scale's intermediate
  levels remain **UNKNOWN** and were not invented.

**Providers, capability and instruments**

- `docs/reserve/provider-qualification-and-routing.md` — Dependency Trust Degradation &
  Obligation Preservation. `INFRASTRUCTURE FAILURE ≠ EVIDENCE OF PARTICIPANT FAILURE`.
- `docs/reserve/protocol-independent-capability-envelope.md` — Observed Execution Cost.
  Constraint cost and observed cost are different fields; no fourth cost accounting.
- `docs/reserve/instrument-admissibility-envelope.md` — Opportunity-Cluster Identity. Seven
  agents on one catalyst are one opportunity expressed seven times, not seven exposures.

**Capital and markets**

- `docs/reserve/capital-admissibility-framework.md` — Outcome Optionality as a sixth
  admissibility dimension. *What futures become unreachable if this capital is accepted?*
- `docs/reserve/domain-aware-capital-intelligence.md` — Capital Relationship Intelligence.
  `DETECT ≠ SOLICIT`; diagnosis precedes matching.
- `docs/reserve/transaction-gap-financing-primitive.md` — Boundary case: inventory demand is
  not a transaction. **This extension narrows application and authorizes nothing new.**
- `docs/reserve/commerce-sniper.md` — Belief-Commerce. *A commercially successful claim is
  not necessarily an epistemically successful claim.*
- `docs/reserve/demand-sovereign-market-infrastructure.md` — Demand-Induced Supply
  Formation. `INTEREST ≠ COMMITMENT ≠ TRANSACTION`.

**Deferred unwritten:** energy metering / ARAL · Regulatory Reconstruction Layer · forecast
provenance. **Dissolved into existing owners:** six submissions, listed in the batch record.

Batch record: `docs/reserve/staging/reserves-2026-08-29.md`
