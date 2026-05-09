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
