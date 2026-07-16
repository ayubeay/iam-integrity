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
