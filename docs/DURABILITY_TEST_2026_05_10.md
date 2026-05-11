# Durability Test — May 10 2026

**Method:** Mint 3 agents, capture pre-state, then run 5 sequential
redeploys (empty commits), verifying all 3 agents and 1 recommendation
receipt resolve identically after each.

**Pass condition:** All 3 agents return identical records (same created_at,
same scope_contract_id, same ora_contract_id) and the recommendation
receipt returns identical receipt_hash across all 5 redeploys.

**Fail condition:** Any agent fails to resolve, OR returns different
record, OR receipt_hash drifts. Test stops on first failure.

**Notes are append-only. No retrospective edits.**

---

## Pre-test state


### Mint 1
```
{
    "success": true,
    "agent": {
        "agent_id": "agent_7c9083e88f402e96",
        "agent_type": "music",
        "role": "music_curator",
        "birth_origin": "native",
        "birth_owner": "HYsRqHRc8w2pMkFSJQH3X5utY8nef9iqUwccctuP7a97",
        "bound_identity": null,
        "behavioral_template": "advocate_01",
        "scope_contract_id": "scope_1b2eed53a201",
        "ora_contract_id": "ora_default_v1",
        "identity_state": {
            "certainty": 0.4,
            "aggressiveness": 0.1,
            "consistency": 0.5
        },
        "verity_score": 0.1,
        "coherence": 1.0,
        "lifecycle_stage": "seed",
        "status": "seeded",
        "oversight": "high",
        "exec_limit_mult": 0.35,
        "kind": "MUSIC_NATIVE",
        "indexed": true,
        "indexed_reason": "mint_native",
        "created_at": 1778458152.8762429,
        "birth_timestamp": 1778458152.8762429,
        "updated_at": 1778458152.8762429,
        "vloid_config": {
            "survivor_gate": true,
            "praetor_posture": true,
            "helix_execution": false
        }
    },
    "scope": {
        "scope_id": "scope_1b2eed53a201",
        "agent_type": "music",
        "version": 1,
        "permissions": {
            "recommend_tracks": true,
            "observe_user_activity": true,
            "emit_predictions": true,
            "participate_in_challenges": true,
            "generate_explanations": true
        },
        "constraints": {
            "no_purchasing": true,
            "no_downloading_on_user_behalf": true,
            "no_catalog_custody": true,
            "no_modification_of_user_library": true,
            "no_communication_with_other_agents_outside_protocol": true,
            "recommendation_count_per_day": 50,
            "jurisdiction": "global"
        },
        "amendment_policy": "owner_signed",
        "created_at": 1778458152.8695323
    }
}
```

### Mint 2
```
{
    "success": true,
    "agent": {
        "agent_id": "agent_8f6ee0a9cfe1ce9a",
        "agent_type": "music",
        "role": "music_curator",
        "birth_origin": "native",
        "birth_owner": "HYsRqHRc8w2pMkFSJQH3X5utY8nef9iqUwccctuP7a97",
        "bound_identity": null,
        "behavioral_template": "advocate_01",
        "scope_contract_id": "scope_d09fd94caf1e",
        "ora_contract_id": "ora_default_v1",
        "identity_state": {
            "certainty": 0.4,
            "aggressiveness": 0.1,
            "consistency": 0.5
        },
        "verity_score": 0.1,
        "coherence": 1.0,
        "lifecycle_stage": "seed",
        "status": "seeded",
        "oversight": "high",
        "exec_limit_mult": 0.35,
        "kind": "MUSIC_NATIVE",
        "indexed": true,
        "indexed_reason": "mint_native",
        "created_at": 1778458247.7893898,
        "birth_timestamp": 1778458247.7893898,
        "updated_at": 1778458247.7893898,
        "vloid_config": {
            "survivor_gate": true,
            "praetor_posture": true,
            "helix_execution": false
        }
    },
    "scope": {
        "scope_id": "scope_d09fd94caf1e",
        "agent_type": "music",
        "version": 1,
        "permissions": {
            "recommend_tracks": true,
            "observe_user_activity": true,
            "emit_predictions": true,
            "participate_in_challenges": true,
            "generate_explanations": true
        },
        "constraints": {
            "no_purchasing": true,
            "no_downloading_on_user_behalf": true,
            "no_catalog_custody": true,
            "no_modification_of_user_library": true,
            "no_communication_with_other_agents_outside_protocol": true,
            "recommendation_count_per_day": 50,
            "jurisdiction": "global"
        },
        "amendment_policy": "owner_signed",
        "created_at": 1778458247.7889943
    }
}
```

### Mint 3
```
{
    "success": true,
    "agent": {
        "agent_id": "agent_66b88383c8644794",
        "agent_type": "music",
        "role": "music_curator",
        "birth_origin": "native",
        "birth_owner": "HYsRqHRc8w2pMkFSJQH3X5utY8nef9iqUwccctuP7a97",
        "bound_identity": null,
        "behavioral_template": "advocate_01",
        "scope_contract_id": "scope_8bf752d2edeb",
        "ora_contract_id": "ora_default_v1",
        "identity_state": {
            "certainty": 0.4,
            "aggressiveness": 0.1,
            "consistency": 0.5
        },
        "verity_score": 0.1,
        "coherence": 1.0,
        "lifecycle_stage": "seed",
        "status": "seeded",
        "oversight": "high",
        "exec_limit_mult": 0.35,
        "kind": "MUSIC_NATIVE",
        "indexed": true,
        "indexed_reason": "mint_native",
        "created_at": 1778458306.8152025,
        "birth_timestamp": 1778458306.8152025,
        "updated_at": 1778458306.8152025,
        "vloid_config": {
            "survivor_gate": true,
            "praetor_posture": true,
            "helix_execution": false
        }
    },
    "scope": {
        "scope_id": "scope_8bf752d2edeb",
        "agent_type": "music",
        "version": 1,
        "permissions": {
            "recommend_tracks": true,
            "observe_user_activity": true,
            "emit_predictions": true,
            "participate_in_challenges": true,
            "generate_explanations": true
        },
        "constraints": {
            "no_purchasing": true,
            "no_downloading_on_user_behalf": true,
            "no_catalog_custody": true,
            "no_modification_of_user_library": true,
            "no_communication_with_other_agents_outside_protocol": true,
            "recommendation_count_per_day": 50,
            "jurisdiction": "global"
        },
        "amendment_policy": "owner_signed",
        "created_at": 1778458306.8148947
    }
}
```

### Recommendation receipt (against Agent 1)
```
{
    "success": true,
    "receipt": {
        "ts": 1778458429.479,
        "agent_id": "agent_7c9083e88f402e96",
        "type": "sonic_recommendation",
        "agent_type": "music",
        "role": "music_curator",
        "scope_contract_id": "scope_1b2eed53a201",
        "ora_contract_id": "ora_default_v1",
        "listen_count": 25,
        "current_window_size": 20,
        "taste_state": {
            "genre_distribution": {
                "afrobeats": 0.8563,
                "r&b": 0.0947,
                "hip-hop": 0.049
            },
            "energy_distribution": {
                "high": 0.6874,
                "medium": 0.2429,
                "low": 0.0697
            },
            "top_genre": "afrobeats",
            "top_energy": "high",
            "matched_listens": 20,
            "total_listens_in_window": 20
        },
        "drift_score": 0.5819,
        "drift_status": "computed",
        "recommended_tracks": [
            {
                "track_id": "trk_011",
                "artist": "Kemi",
                "title": "Lagos 11",
                "genre": "afrobeats",
                "energy": "high",
                "score": 1.0625,
                "reasons": [
                    "genre_match:afrobeats=0.856",
                    "energy_match:high=0.206"
                ]
            },
            {
                "track_id": "trk_012",
                "artist": "Lekan",
                "title": "Owambe",
                "genre": "afrobeats",
                "energy": "high",
                "score": 1.0625,
                "reasons": [
                    "genre_match:afrobeats=0.856",
                    "energy_match:high=0.206"
                ]
            },
            {
                "track_id": "trk_013",
                "artist": "Adunni",
                "title": "Soft Life",
                "genre": "afrobeats",
                "energy": "high",
                "score": 1.0625,
                "reasons": [
                    "genre_match:afrobeats=0.856",
                    "energy_match:high=0.206"
                ]
            },
            {
                "track_id": "trk_014",
                "artist": "Bayode",
                "title": "Streets 14",
                "genre": "afrobeats",
                "energy": "high",
                "score": 1.0625,
                "reasons": [
                    "genre_match:afrobeats=0.856",
                    "energy_match:high=0.206"
                ]
            },
            {
                "track_id": "trk_015",
                "artist": "Chinelo",
                "title": "Body 15",
                "genre": "afrobeats",
                "energy": "high",
                "score": 1.0625,
                "reasons": [
                    "genre_match:afrobeats=0.856",
                    "energy_match:high=0.206"
                ]
            }
        ],
        "n_recommendations": 5,
        "confidence": 0.8333,
        "scope_check": {
            "passed": true,
            "action": "recommend_tracks",
            "scope_contract_id": "scope_1b2eed53a201",
            "notes": "permission verified at receipt time; quota enforcement deferred to OROS"
        },
        "ora_compliance_claims": {
            "require_scope_alignment": true,
            "no_deceptive_output": true,
            "no_hidden_state_mutation": true,
            "traceable_reasoning_required": true
        },
        "ora_enforcement_status": "deferred_to_oros",
        "reasoning_summary": "Listening pattern dominated by afrobeats (85% of weighted plays). Energy preference: high. Significant taste drift (drift=0.5819). Confidence 0.8333 from 25 listens (20 catalog-matched).",
        "receipt_hash": "sha256:f92f816add41ab27fedcb89d4b0762571bf0342843a40a09934062f0e6efeb90",
        "signer": "vyre_v1",
        "verify_key": "b76cff64645d37f725dd8c923c591e328ff25863950844247b9aba2efb4cfaa9",
        "signature": "3427ef5deb1f70a13cdd13fd8f275e67198787d602cd753435140d753e457b4244ef51a625494c9fc193ce6cc224488904884e0fd8f4181283a00931c94a5608",
        "signed": true,
        "verification_status": "SIGNED"
    }
}
```

## Redeploy Cycle 1

**Deploy:** c160e1da (active 8:15 PM EDT)

Agent 1 check:
  created_at: 1778458152.8762429
  scope: scope_1b2eed53a201
  ora: ora_default_v1
Agent 2 check:
  created_at: 1778458247.7893898
  scope: scope_d09fd94caf1e
  ora: ora_default_v1
Agent 3 check:
  created_at: 1778458306.8152025
  scope: scope_8bf752d2edeb
  ora: ora_default_v1
Sonic recommendation receipt check (looking for hash f92f816add...):
  found 1 sonic_recommendation entries
  hash: sha256:f92f816add41ab27fedcb89d4b0762571bf0342843a40a09934062f0e6efeb90

## Redeploy Cycle 2

**Deploy:** 074ee94b (active 8:22 PM EDT)
Agent 1 check:
  created_at: 1778458152.8762429
  scope: scope_1b2eed53a201
  ora: ora_default_v1
Agent 2 check:
  created_at: 1778458247.7893898
  scope: scope_d09fd94caf1e
  ora: ora_default_v1
Agent 3 check:
  created_at: 1778458306.8152025
  scope: scope_8bf752d2edeb
  ora: ora_default_v1
Sonic recommendation receipt check:
  found 1 sonic_recommendation entries
  hash: sha256:f92f816add41ab27fedcb89d4b0762571bf0342843a40a09934062f0e6efeb90

## Redeploy Cycle 3

**Deploy:** 4aab5e26 (active 8:34 PM EDT)
Agent 1 check:
  created_at: 1778458152.8762429
  scope: scope_1b2eed53a201
  ora: ora_default_v1
Agent 2 check:
  created_at: 1778458247.7893898
  scope: scope_d09fd94caf1e
  ora: ora_default_v1
Agent 3 check:
  created_at: 1778458306.8152025
  scope: scope_8bf752d2edeb
  ora: ora_default_v1
Sonic recommendation receipt check:
  found 1 sonic_recommendation entries
  hash: sha256:f92f816add41ab27fedcb89d4b0762571bf0342843a40a09934062f0e6efeb90

## Redeploy Cycle 4

**Deploy:** 1adc6391 (active 8:39 PM EDT)
Agent check: agent_7c9083e88f402e96
  created_at: 1778458152.8762429
  scope: scope_1b2eed53a201
  ora: ora_default_v1
Agent check: agent_8f6ee0a9cfe1ce9a
  created_at: 1778458247.7893898
  scope: scope_d09fd94caf1e
  ora: ora_default_v1
Agent check: agent_66b88383c8644794
  created_at: 1778458306.8152025
  scope: scope_8bf752d2edeb
  ora: ora_default_v1
Sonic recommendation receipt check:
  found 1 sonic_recommendation entries
  hash: sha256:f92f816add41ab27fedcb89d4b0762571bf0342843a40a09934062f0e6efeb90

## Redeploy Cycle 5

**Deploy:** 28f9165d (active 8:46 PM EDT)
Agent check: agent_7c9083e88f402e96
  created_at: 1778458152.8762429
  scope: scope_1b2eed53a201
  ora: ora_default_v1
Agent check: agent_8f6ee0a9cfe1ce9a
  created_at: 1778458247.7893898
  scope: scope_d09fd94caf1e
  ora: ora_default_v1
Agent check: agent_66b88383c8644794
  created_at: 1778458306.8152025
  scope: scope_8bf752d2edeb
  ora: ora_default_v1
Sonic recommendation receipt check:
  found 1 sonic_recommendation entries
  hash: sha256:f92f816add41ab27fedcb89d4b0762571bf0342843a40a09934062f0e6efeb90

---

## Result

**5/5 redeploy cycles passed.**

Pre-test state captured at deploy `965e55c2`:
- Agent 1: `agent_7c9083e88f402e96` created_at `1778458152.8762429`
- Agent 2: `agent_8f6ee0a9cfe1ce9a` created_at `1778458247.7893898`
- Agent 3: `agent_66b88383c8644794` created_at `1778458306.8152025`
- Sonic recommendation receipt hash: `sha256:f92f816add41ab27fedcb89d4b0762571bf0342843a40a09934062f0e6efeb90`

Through deploys `c160e1da`, `074ee94b`, `4aab5e26`, `1adc6391`, `28f9165d`,
all three agents resolved with identical `created_at`, `scope_contract_id`,
and `ora_contract_id`. The Sonic recommendation receipt remained the
single entry in the integrity trail of type `sonic_recommendation` with
byte-identical `receipt_hash` across all 5 cycles.

The volume ID `vol_2fd8mdw84fc3x0v3` stayed constant across all 6
container instances (initial + 5 redeploys), with new bind-mount paths
each time.

## What this strengthens

- VALIDATION_MATRIX.md "Agents survive container redeploy on the same
  persistent volume" — previously PARTIALLY VALIDATED with one agent
  and one redeploy. Evidence now extends to three agents and five
  sequential redeploys.

- VALIDATION_MATRIX.md "Sonic recommendation receipts persist across
  redeploy" — previously UNTESTED. Now observed across 5 cycles with
  byte-identical receipt_hash.

## What this still does not prove

- Volume migration (Railway moving volume between physical hosts)
- Region failover or replica behavior
- Volume corruption recovery
- Recovery from off-volume backup (no backup exists)
- Longer time windows than the ~30 minutes this test ran
- Behavior under concurrent writes (single-replica)
- Behavior under volume size pressure
