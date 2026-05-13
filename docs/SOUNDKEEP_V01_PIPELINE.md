# SoundKeep v0.1 — Pipeline Architecture

Companion to `SOUNDKEEP_V01_SCOPE.md`. The scope doc says **what** v0.1
tests and what stays excluded. This doc says **how** the system is
structured, stage by stage, with explicit responsibilities and
non-goals at each stage.

## Architectural commitments

1. SoundKeep v0.1 lives at `iam-integrity/soundkeep/`. Not a new repo.
   Not a new Railway project.
2. It has its own continuation engine. Sonic v1 stays intact.
3. Candidate source is a **curated seed graph**, not external APIs.
   API integration is RESERVED for v0.2.
4. Signed interaction receipts reuse the existing vyre_v1 / Ed25519
   substrate for continuity with the broader stack.
5. No new agents minted per DJ in v0.1. DJs are tracked by lightweight
   session identifiers, not full agent identity. Agent minting per DJ
   is RESERVED.

## Pipeline overview

```
DJ Input
  ↓
[1] Normalization
  ↓
[2] Continuation Engine
  ↓
[3] Scoring Layer
  ↓
[4] Result Surface (Web UI)
  ↓
[5] Interaction Capture
  ↓
[6] Signed Observation Receipts
```

Six stages. Each has a single responsibility. Each has explicit
inputs, outputs, and non-goals.

---

## Stage 1 — Normalization

**Responsibility:** Convert messy DJ input into canonical references
in the seed graph.

**Input:**
- Free-text strings: artist names, track titles, genre labels, vibes
- Optional: URLs from Spotify, YouTube, SoundCloud, Apple Music (parsed
  for metadata only — not used for catalog lookup in v0.1)

**Output:**
- A list of canonical seed graph node IDs the user "anchored" on
- A list of unresolved inputs (text that did not match anything)
- Confidence per resolution

**Logic:**
- Fuzzy match against seed graph entries (artist name aliases, genre
  variants like "afrobeats" vs "afro-beats" vs "afrobeat")
- Genre/vibe tags map to clusters in the graph, not individual tracks

**Explicitly NOT in this stage:**
- No external API lookups
- No track audio analysis
- No "did you mean" suggestions in v0.1 (just resolve or fail)
- No adaptive learning from user behavior in v0.1

---

## Stage 2 — Continuation Engine

**Responsibility:** Generate a pathway through the seed graph that
feels intentional — not just "tracks similar to your inputs."

**Input:**
- Anchored seed graph node IDs from Stage 1
- Optional user-stated intent ("warm up set", "peak time", "after-hours")
  — but v0.1 does not require this; if absent, the engine produces a
  generic continuation

**Output:**
- An ordered sequence of candidate node IDs (length: 10-20)
- For each candidate: the reasoning tag(s) that brought it into the
  pathway (e.g., "energy step from peak to wind-down", "regional
  bridge: SA -> Nigeria", "underground adjacency from Skepta")

**Logic:**
- Traverse adjacency edges in the seed graph
- Edge types include: genre adjacency, energy progression, regional
  bridge, era continuation, underground/mainstream distance, remix
  lineage
- Each pathway is a small walk through the graph that maintains
  coherence (not random jumps)
- v0.1 algorithm: weighted-walk with explicit transition rules,
  not ML

**Explicitly NOT in this stage:**
- No collaborative filtering
- No "users who liked X also liked Y"
- No popularity scoring
- No personalization based on past behavior (v0.1 has no user history)
- No neural recommendation models
- No external data sources

---

## Stage 3 — Scoring Layer

**Responsibility:** Rank candidates within the pathway by how strongly
each justifies its position.

**Input:**
- Candidate sequence from Stage 2 with reasoning tags

**Output:**
- Same sequence, with a score per candidate (0.0-1.0)
- Top reasoning tag per candidate, surfaced to user as the "why this"
  line in the UI

**Logic:**
- Score combines: edge weight strength, number of justifying reasons,
  position-appropriate energy match
- v0.1 scoring is deterministic, not learned

**Explicitly NOT in this stage:**
- No user feedback loop in v0.1 (scoring does not update from clicks)
- No A/B testing infrastructure
- No global popularity bias

---

## Stage 4 — Result Surface (Web UI)

**Responsibility:** Present the pathway to the DJ in a way that
makes the continuation legible.

**Input:**
- Scored pathway from Stage 3

**Output (to user):**
- Ordered list of recommendations
- Each entry shows: artist, title, genre, energy, "why this" line,
  outbound preview links (YouTube, SoundCloud, Spotify, Apple Music
  — generated as search URLs, not API calls)
- Save / Skip / Group / Revisit buttons per entry
- Session persistence: returning to the page reloads saved state

**Output (to backend):**
- DJ session identifier (lightweight UUID, not a minted agent)
- Pathway impression event (what was shown, when, in what order)

**Explicitly NOT in this stage:**
- No audio playback
- No track previews hosted by SoundKeep
- No image/cover art generation
- No social features
- No login/auth in v0.1 (session-based only)
- No styling beyond functional clarity — visual polish is RESERVED

---

## Stage 5 — Interaction Capture

**Responsibility:** Record observable DJ behavior on the result
surface.

**Input:**
- User actions: save, skip, group, revisit, outbound link click,
  return visit, session end

**Output:**
- Structured interaction events with: session ID, action type, target
  candidate ID, timestamp, pathway context

**Logic:**
- Events buffered client-side and posted to backend
- Backend appends to interaction log (file or lightweight DB —
  decided in implementation, not here)

**Explicitly NOT in this stage:**
- No third-party analytics (no Google Analytics, no Mixpanel, no
  Segment)
- No identity correlation across sessions in v0.1
- No real-time dashboards — analysis is post-hoc

---

## Stage 6 — Signed Observation Receipts

**Responsibility:** Emit signed, durable evidence of what each DJ
session produced.

**Input:**
- Aggregated interaction events for a session (or a pathway)

**Output:**
- A signed receipt of type `soundkeep_session` or
  `soundkeep_pathway` (exact type decided in implementation)
- Receipt structure includes: session ID, pathway shown, interactions
  recorded, timestamps, signed with existing vyre_v1 / Ed25519
  substrate

**Logic:**
- Reuses signing primitives from `signing.py`
- Receipts written to existing `integrity_trail.jsonl` OR a separate
  SoundKeep trail (decided in implementation — but if separate,
  same format)

**Explicitly NOT in this stage:**
- No new signing keys
- No new verification infrastructure
- No on-chain anchoring (RESERVED)
- No public receipt browsing in v0.1

---

## What is NOT in the pipeline

The following are deliberate exclusions, not oversights:

- **No agent per DJ.** DJs are session entities in v0.1, not minted
  agents. Agent-per-DJ is meaningful only when reputation/continuity
  across the broader stack matters, which is post-validation.
- **No taste-state computation.** Sonic v1 has `taste_state` with genre
  distribution and energy distribution. SoundKeep v0.1 does NOT compute
  this — its hypothesis is about pathway intent, not aggregate taste.
  Taste-state ingestion can be RESERVED.
- **No collaboration features.** Single-DJ-per-session only.
- **No persistence of pathways across DJs.** Each session is isolated.
- **No "trending" or "popular" surfaces.** v0.1 is not a content
  discovery surface; it is a continuation surface.

## The seed graph

The seed graph is the most consequential v0.1 decision. It is:

- Built from existing `data/sonic_catalog_seed.json` (200 tracks,
  8 genres) as a starting point
- Expanded manually with explicit adjacency edges (not ML-derived)
- Target size for v0.1: 500-2000 entries
- Edges typed by: genre adjacency, energy progression, regional
  bridge, era continuation, underground distance, remix lineage
- Stored as: structured JSON or simple graph file (Neo4j is RESERVED)
- Maintained by hand initially — automation is RESERVED

The seed graph is the actual locus of "intentional curation" the
hypothesis depends on. A poorly-built graph guarantees a poorly-built
test, regardless of how good the engine is.

## Build sequence

In rough order of dependence:

1. Seed graph expansion (`soundkeep/data/seed_graph.json`)
2. Continuation engine (`soundkeep/continuation/walker.py`)
3. Scoring layer (`soundkeep/continuation/scoring.py`)
4. Normalization (`soundkeep/normalize.py`)
5. Web UI scaffolding (`soundkeep/templates/`, `soundkeep/static/`)
6. FastAPI routes (`soundkeep/routes.py`, mounted on existing app)
7. Interaction capture endpoint
8. Receipt emission integration

Stages 1-3 can be tested in isolation before any UI exists. That
is the right order — the engine is the differentiator.

## What success looks like at the architecture level

The pipeline succeeds architecturally if:

- Each stage can be tested independently
- A change to the seed graph or scoring does not require changes
  elsewhere
- Receipts are produced for every meaningful interaction
- The backend has zero dependencies on external music APIs
- Total implementation should remain structurally small and auditable (target: <=1500 LOC excluding seed graph data)

## Activation conditions for future stages

Anything beyond the six stages above — taste state, agent identity,
collaborative features, external metadata, ML scoring, payment,
mobile — is governed by the same activation-condition discipline
as RESERVED.md. No additions during v0.1 implementation without
revisiting this document.
