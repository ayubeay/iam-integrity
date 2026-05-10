# VALIDATION_MATRIX.md

Architectural claims mapped to validation status. The point of this file
is preventing claim-evidence drift: every claim states what it would mean
to be true, what evidence currently exists, and what the status implies.

Status definitions:

- **VALIDATED** — direct observation under realistic conditions, repeatable
- **PARTIALLY VALIDATED** — observed under narrow conditions, not yet
  generalized
- **UNVALIDATED** — claim plausible from architecture but no test exists
- **UNTESTED** — capability or property not yet attempted
- **UNCLAIMED** — explicitly not a current property of the system

Discipline rule: when in doubt between two statuses, pick the lower one.

---

## Identity and persistence

### Agents survive container redeploy on the same persistent volume

**Status: PARTIALLY VALIDATED**

Evidence: `agent_af611f1f5ac50670` minted on deploy `7214d6a7`, resolved
identically on deploy `965e55c2` after empty-commit redeploy. Same volume,
same region.

What's missing: only one agent tested, only one redeploy executed.
Multi-agent and sequential-redeploy stress not performed.

To advance to VALIDATED: ≥3 agents, ≥5 sequential redeploys, all agents
resolve identically after each redeploy.

---

### Agents persist across longer runtime windows

**Status: PARTIALLY VALIDATED**

Evidence: `agent_af611f1f5ac50670` resolved ~82 hours after mint, through
unspecified Railway lifecycle activity in the intervening window.

What's missing: no controlled measurements of what happened in the window.
No multi-agent baseline. No comparison agent that *failed* to persist
(needed to know whether persistence is robust or fragile but lucky).

To advance: deliberate scheduled checks at 24h / 72h / 1wk / 2wk
intervals across multiple agents.

---

### Volume survives infrastructure migration

**Status: UNTESTED**

Railway can migrate volumes between physical hosts during their lifecycle.
This has not been triggered or tested. If it occurs unexpectedly, behavior
is unknown.

To test: would require either Railway-initiated migration (uncontrolled)
or a deliberate destroy-and-restore cycle (currently no off-volume
backup, so this would be destructive).

---

### Off-volume backup of agent registry exists

**Status: UNCLAIMED**

No off-volume backup mechanism exists. If the Railway volume is lost
(corruption, accidental deletion, region failure), all production agent
state is gone permanently.

The pre-durability state of two agents (`agent_2b9d1666793fc844` and
`agent_fb864655eb48b7c2`) demonstrates this failure mode at a smaller
scale: when the runtime working directory was wiped on May 6, those
agents became unrecoverable.

To advance: implement scheduled trail snapshot to S3-compatible storage,
plus restore procedure validated by drill.

---

## Cryptographic anchor

### Birth receipts are Ed25519-signed

**Status: VALIDATED**

Evidence: every birth receipt observed in production includes `signed: true`,
`verification_status: SIGNED`, signer `vyre_v1`, and a 128-character hex
signature. The platform verify_key
`b76cff64645d37f725dd8c923c591e328ff25863950844247b9aba2efb4cfaa9` is
returned by `/survivor/verify-key` and is consistent across receipts.

Local offline verification against canonicalized payloads succeeded using
the public verify key and detached signatures emitted by the platform.

---

### Birth receipts are independently verifiable

**Status: VALIDATED**

Evidence: receipts have been verified end-to-end during local testing
using `nacl.signing.VerifyKey` against canonical-JSON of the payload.
Hash and signature both validate.

---

### Receipt cryptographic anchor survives redeploy unchanged

**Status: PARTIALLY VALIDATED**

Evidence: `receipt_hash:
sha256:b352466e42035ebdcc3effec0381d6b2292b8488ce918c673e731716eb72aff2`
and the corresponding signature were byte-identical when queried before
and after the May 7 redeploy.

What's missing: only one receipt observed across redeploy. Sonic
recommendation receipts not tested across redeploy.

---

### Sonic recommendation receipts persist across redeploy

**Status: UNTESTED**

The infrastructure should support this (recommendation receipts use the
same trail file as birth receipts), but it was never explicitly verified.

To validate: produce a sonic_recommendation receipt, redeploy, query the
trail, confirm receipt hash unchanged.

---

### Receipt chain ancestry (prior_receipt_id)

**Status: UNCLAIMED**

Receipts are currently standalone. No `prior_receipt_id` field, no chain
semantics, no replay model. Claims of "verifiable identity timeline" or
"compositional continuity" are not supported.

---

## Governance binding and enforcement

### Agents are bound to scope contracts at mint

**Status: VALIDATED**

Evidence: every minted agent record contains `scope_contract_id`. Scope
contract files exist on the volume in `scopes/{scope_id}.json`. The
binding is recorded inside the signed birth receipt payload.

---

### Agents are bound to ORA contracts at mint

**Status: VALIDATED**

Evidence: every minted agent record contains `ora_contract_id:
ora_default_v1`. The default ORA contract exists at
`oras/ora_default_v1.json`. Binding is recorded in the signed birth
receipt payload.

---

### Scope permissions are enforced at action time

**Status: UNCLAIMED**

Currently the `/agents/{id}/sonic/recommend` endpoint records a
`scope_check` field claiming `passed: true` with a note: *"permission
verified at receipt time; quota enforcement deferred to OROS"*.

This is a compliance claim, not enforcement. No service reads the scope
contract to gate execution. Daily recommendation count is not actually
counted. The constraint exists in the contract; the enforcement does not.

To advance: OROS must read scope contracts and gate per-action calls.

---

### ORA constraint rules are enforced

**Status: UNCLAIMED**

Same shape as scope enforcement. Receipts include `ora_compliance_claims`
listing `require_scope_alignment`, `no_deceptive_output`,
`no_hidden_state_mutation`, `traceable_reasoning_required`, and
`ora_enforcement_status: deferred_to_oros`.

The "deferred_to_oros" framing is intentional. ORA contracts are bound
and recorded but not enforced. Future OROS work would need to read
`oras/{ora_id}.json` and apply enforcement decisions.

---

### JANUS dual-read validation

**Status: UNCLAIMED**

JANUS validator does not exist. `ora_default_v2` (which would include
janus_rules) is reserved on activation precondition: JANUS must exist
first. See RESERVED.md.

---

## Sonic v1 cognition loop

### Recommendations are deterministic

**Status: VALIDATED**

Evidence: identical listen history produces identical recommended track
IDs across runs. Self-test in `sonic_loop.py` includes determinism
assertion that passes locally.

---

### Taste state computation reflects listen history

**Status: PARTIALLY VALIDATED**

Evidence: synthetic test history with weighted afrobeats listens produced
`top_genre: afrobeats` with weighted distribution `{afrobeats: 0.86,
r&b: 0.09, hip-hop: 0.05}`. Drift score correctly detected hip-hop
→ afrobeats transition with `drift_score: 0.5819`.

What's missing: only synthetic test data. No real user listen history
processed.

---

### Recommendations exclude already-listened tracks

**Status: VALIDATED**

Evidence: `_resolve_listened_track_ids` excludes matched tracks from
ranking pool. Self-test verifies excluded tracks do not appear in output.

---

### Recommendations are calibrated to user taste in production

**Status: UNTESTED**

No real user has interacted with Sonic. No feedback signal exists.
Whether the recommendations are subjectively good for any actual person
is unknown.

---

## LITMUS firewall

### LITMUS dashboard renders policy decisions

**Status: VALIDATED**

Evidence: dashboard at localhost:8082 renders 117 decisions with policy,
risk, system, and source provenance fields. UI confirmed working May 7.

---

### LITMUS auto-ingests decisions from monitored systems

**Status: UNCLAIMED**

No live ingestion pipeline exists. The 117 decisions in `firewall.db`
were loaded manually. VERITY, IAM, SURVIVOR, GATE, GHOSTLEDGER,
RENTRECLAIM are listed as monitored systems but do not currently emit
to LITMUS.

To advance: implement writers in each monitored system's decision path
that append to LITMUS database.

---

## Process and architecture

### Static templates are separable from mutable state

**Status: VALIDATED**

Evidence: `paths.py` enforces this separation. Local test in `/tmp/iam_test`
with `IAM_DATA_ROOT=/tmp/test_volume` confirmed: catalog read from
CODE_ROOT, runtime artifacts (agents_index.json, integrity_trail.jsonl,
oras/, scopes/) written to DATA_ROOT, code dir remained clean.

Production: same configuration with volume at `/data`, env var set,
deploy `965e55c2` continues to operate cleanly.

---

### Documentation prevents claim-evidence drift

**Status: UNVALIDATED — by definition cannot be validated by self-reference**

The hypothesis is that maintaining ATTRIBUTION_NOTES, VALIDATION_MATRIX,
and RESERVED prevents architectural drift over time. Whether this is
true requires future observation: at the next milestone, are framings
better-grounded in evidence than before? Compare against the May 6/7
session transcripts where advisor framings outpaced test results.
