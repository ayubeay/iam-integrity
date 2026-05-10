# ATTRIBUTION_NOTES.md

What the live system has actually shown us. This file is observed history,
not architectural interpretation. Every claim here points to a specific
event, deploy, or measurement.

Discipline rule: prefer under-claiming. "Observed under condition X" is
stronger than "validated." Architectural framings ("continuity infrastructure",
"compositional continuity") are hypotheses about what the evidence enables,
not properties the evidence has demonstrated.

---

## May 6 — Production registry wipe

**What happened:**

Sonic v1 cognition loop (commit `da581e9`) deployed to Railway. The deploy
succeeded — code shipped, endpoints registered, uvicorn booted clean.

Calling `/agents/agent_fb864655eb48b7c2/sonic/recommend` returned:

```
"failed to read agents_index: [Errno 2] No such file or directory:
 '/app/agents_index.json'"
```

`/agents/summary` showed `indexed_agents: 0`. The previously-minted agent
record was gone.

**Root cause:**

`agents_index.json`, `integrity_trail.jsonl`, `scopes/`, `oras/` were all
gitignored as runtime artifacts. They were being written to `/app`
(working directory) inside the container. Each Railway deploy clones a
fresh container from source — and source doesn't include the gitignored
files. So the deploy started from an empty registry while the previous
deploy's container (with its registry state) was destroyed.

**What was lost:**

Two pre-durability agents are permanently gone from the production registry:

- `agent_2b9d1666793fc844` — first music agent on the platform (minted
  before signing existed). Confirmed 404 on May 9.

- `agent_fb864655eb48b7c2` — first cryptographically attested mint-born
  agent. Birth receipt was real and signed at the time. Receipt itself
  was also lost when `integrity_trail.jsonl` was wiped. Confirmed 404 on
  May 9.

The signed receipts for those two agents were captured in chat logs at the
time, so cryptographic verification of their birth could in principle still
be performed — but the platform itself can no longer resolve those IDs.
This is a real continuity rupture, not a recoverable state.

**Lesson (which became the durability patch):**

A platform that claims "identity continuity" cannot survive its own
deploys. The May 6 wipe demonstrated that the previous architecture had
no separation between code state and operational state. Mutable runtime
state and immutable code template were both being treated as
deploy-time inputs.

---

## May 7 — Durability patch and first redeploy continuity test

**Patch (commit `9dfc0e1`):**

Introduced `paths.py` as single source of truth for all data paths.
Established two roots:

- `CODE_ROOT` — where source ships, contains static templates
- `DATA_ROOT` — configurable via `IAM_DATA_ROOT` env var, contains
  mutable runtime state

Static (CODE_ROOT): `archetypes.json`, `agents_zodiac.json`,
`data/sonic_catalog_seed.json`, `challenges/`.

Mutable (DATA_ROOT): `agents_index.json`, `agents_seed.json`,
`integrity_trail.jsonl`, `scopes/`, `oras/`, `data/survivor_data.json`,
`data/helixcan_snapshot.json`.

Local default behavior preserved: when `IAM_DATA_ROOT` is unset,
DATA_ROOT defaults to CODE_ROOT.

**Railway configuration:**

- Volume created, mounted at `/data`
- Environment variable `IAM_DATA_ROOT=/data` set on web service

**The redeploy test:**

1. Deploy `7214d6a7` came up with the durability patch active.
2. Minted `agent_af611f1f5ac50670` via `/agents/mint`.
3. Confirmed it resolved on `/agents/{id}` and birth receipt was signed
   in the trail with `receipt_hash:
   sha256:b352466e42035ebdcc3effec0381d6b2292b8488ce918c673e731716eb72aff2`.
4. Pushed empty commit `e2cc4e5` to trigger redeploy.
5. Deploy `965e55c2` came up — different deploy ID, fresh container,
   same persistent volume.
6. Re-queried `/agents/agent_af611f1f5ac50670`. Returned full agent
   record with identical scope_contract_id, ora_contract_id, created_at.
7. Re-queried `/integrity/recent`. Returned identical receipt_hash and
   identical signature.

**What this proved:**

One mint-born agent persisted across one production redeploy on the same
volume in the same region. Receipt cryptographic anchor (hash + signature)
remained byte-identical pre/post redeploy.

**What this did not prove:**

- Volume migration (Railway moving volume between hosts)
- Region failover
- Multi-replica behavior
- Volume corruption recovery
- Recovery from off-volume backup (no off-volume backup exists yet)
- Multi-agent persistence (only one agent tested)
- Sonic recommendation receipt persistence (only birth receipt tested)
- Sequential redeploy stress (only one redeploy executed)

---

## May 10 — ~82 hour resolve

**Observation:**

`agent_af611f1f5ac50670` queried May 10 ~morning EDT. Returned full agent
record with `created_at: 1778125970.7004893` (May 7 ~00:30 UTC).

**What this adds:**

The agent remained resolvable approximately 82 hours after its mint and
through whatever Railway lifecycle activity occurred in that window
(replicas, container churn, scheduled processes — exact count not
recorded). This strengthens the May 7 redeploy test by extending the
observation window.

**What this does not add:**

- No deliberate stress was applied during this window
- No specific count of redeploys is known (the agent survived "whatever
  happened" rather than a specific test)
- Other agents were not checked, so multi-agent persistence remains
  untested

---

## May 7 — LITMUS dashboard state

**What's running:**

LITMUS Firewall + PoW Observatory at `localhost:8082` (FastAPI, SQLite-backed).
Started via `~/litmus-firewall/start.sh`. 117 decisions visible on the
dashboard, 87 allowed / 10 denied / 20 review across 8 active policies
and 6 monitored systems.

**What's real about it:**

The 117 decisions represent real evaluations that occurred — LITMUS
applied policies, doctrine logic fired, allow/deny outcomes are genuine
results.

**What's manual about it:**

The path from system → LITMUS database is not auto-ingested. Each
decision was loaded into `firewall.db` by hand. The dashboard renders
correctly because the underlying data is real, but the live ingestion
pipeline (VERITY → LITMUS, SURVIVOR → LITMUS, etc.) does not exist.

**Honest framing:**

"117 verified decisions, manually transported." Not "live observation
pipeline." Not "demo data." The substrate is real; the transport is not
yet automated.

---

## Open observations not yet investigated

These were noticed but not resolved during the May 6/7 sessions:

**Authorize traffic flood:**
Railway deploy logs (`67c55b8a`, `965e55c2`) showed 50+ rapid `POST /authorize`
requests from internal IPs (`100.64.0.x`) during normal deploys. Source
not identified. Could be another internal service hammering the endpoint,
a runaway loop, or expected health-check behavior. Not investigated.

**Indexer behavior on empty volume:**
After the durability patch, `seed_agents` count dropped from 8 to 0 on
the first deploy with empty volume. Expected, since `verity_indexer.py`
rebuilds `agents_seed.json` from the trail and the volume started empty.
Behavior should self-correct as activity accumulates. Not yet observed
self-correcting because production traffic is low.

---

## Architectural framings that the evidence does not yet support

Several framings have appeared in advisor conversations that are not yet
backed by direct observation. Naming them here so they don't quietly
become accepted truth:

- "Continuity infrastructure achieved" — evidence is one redeploy plus
  ~58h survival of one agent. This is a foundation for continuity claims,
  not a demonstration of continuity infrastructure.

- "Identity is no longer tied to process lifetime" — true for one agent
  across one container handoff. Not yet generalized.

- "Compositional continuity" — would require receipt chaining and
  multi-event lineage. Neither exists yet.

- "Bounded execution continuity" — implies enforcement of bounds. Current
  ORA contract is binding-only; OROS-side enforcement does not exist.

These framings may turn out to be accurate descriptions later. For now,
they remain hypotheses about what the current architecture may support,
not properties established by testing.
