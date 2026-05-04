"""
Domain-agnostic agent minting.

Atomic birth pipeline that creates a new mint-born agent with:
  - structured agent record in agents_index.json (with scope + ORA bindings)
  - birth receipt in integrity_trail.jsonl (with scope + ORA bindings)
  - persistent scope contract in scopes/{scope_id}.json
  - persistent ORA contract reference (oras/{ora_id}.json must already exist)
  - VERITY score baseline (0.10)
  - identity state seeded from behavioral template
  - lifecycle initialized to "seed"

Each mint binds the agent to two contracts:
  scope_contract_id  — what the agent is allowed to do  (capabilities)
  ora_contract_id    — how the agent is governed         (constraint rules + enforcement)

NOTE: This module creates the artifacts that will later allow OROS to
enforce both scope and ORA contracts. OROS does not yet read or enforce
either — those are separate later patches. Anchoring at mint comes first
so all agents born after this patch share the same governance schema,
avoiding migration debt when enforcement arrives.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

from ora_contract import (
    DEFAULT_ORA_ID,
    ensure_default_ora_persisted,
    load_ora_contract,
)

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
AGENTS_INDEX_PATH = DATA_DIR / "agents_index.json"
TRAIL_PATH = DATA_DIR / "integrity_trail.jsonl"
SCOPES_DIR = DATA_DIR / "scopes"
ARCHETYPES_PATH = DATA_DIR / "archetypes.json"


# ── Errors ───────────────────────────────────────────────────────────────────

class MintError(Exception):
    """Raised when agent mint fails. Triggers rollback."""
    pass


# ── Public API ───────────────────────────────────────────────────────────────

def mint_agent(
    *,
    agent_type: str,
    role: str,
    birth_owner: str,
    behavioral_template: str,
    scope_contract: Dict[str, Any],
    ora_contract_id: str = DEFAULT_ORA_ID,
    identity_seed: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Atomic birth: create agent record, persist scope, write birth receipt,
    register in agents_index. Roll back if any step fails.

    Args:
        agent_type:           "music" | "trading" | "inspection" | "debate"
        role:                 e.g. "music_curator", "track_predictor"
        birth_owner:          wallet or account that creates the agent
        behavioral_template:  archetype name, e.g. "advocate_01" or "Advocate"
        scope_contract:       structured scope dict (from scope_contract.py)
        ora_contract_id:      governance contract id (default: ora_default_v1)
        identity_seed:        optional override for initial identity vector

    Returns:
        The full agent record that was written to agents_index.json.

    Raises:
        MintError: if any step fails. Rollback is best-effort.
        ValueError: if inputs are invalid.
    """
    # ── 1. Validate inputs ───────────────────────────────────────────────────
    if not agent_type or not isinstance(agent_type, str):
        raise ValueError("agent_type must be a non-empty string")
    if not role or not isinstance(role, str):
        raise ValueError("role must be a non-empty string")
    if not birth_owner or not isinstance(birth_owner, str):
        raise ValueError("birth_owner must be a non-empty string")
    if not behavioral_template:
        raise ValueError("behavioral_template is required")

    from scope_contract import validate_scope
    if not validate_scope(scope_contract):
        raise ValueError(
            f"scope_contract failed validation. "
            f"Required fields: scope_id, agent_type, version, permissions, "
            f"constraints, amendment_policy, created_at"
        )

    if scope_contract["agent_type"] != agent_type:
        raise ValueError(
            f"scope_contract.agent_type={scope_contract['agent_type']!r} "
            f"does not match agent_type={agent_type!r}"
        )

    # ── 2. Verify ORA contract is persisted on disk ──────────────────────────
    # If the requested ora_contract_id is the default, ensure it exists.
    # If it's a custom id, the caller is responsible for persisting it
    # before calling mint_agent.
    if ora_contract_id == DEFAULT_ORA_ID:
        ensure_default_ora_persisted()

    try:
        load_ora_contract(ora_contract_id)
    except FileNotFoundError as e:
        raise ValueError(
            f"ora_contract_id={ora_contract_id!r} is not persisted. "
            f"Persist it to oras/{ora_contract_id}.json before minting."
        ) from e

    # ── 3. Load behavioral template ──────────────────────────────────────────
    archetype_profile = _load_archetype(behavioral_template)
    if archetype_profile is None:
        raise ValueError(
            f"unknown behavioral_template: {behavioral_template!r}. "
            f"Must match an archetype in archetypes.json by archetype name "
            f"or agent_id."
        )

    # ── 4. Compute deterministic agent_id ────────────────────────────────────
    now = time.time()
    agent_id = "agent_" + hashlib.sha256(
        f"{birth_owner}:{agent_type}:{role}:{now}".encode()
    ).hexdigest()[:16]

    # ── 5. Build agent record ────────────────────────────────────────────────
    initial_identity = (
        identity_seed
        if identity_seed is not None
        else dict(archetype_profile["identity_anchor"])
    )

    agent_record = {
        "agent_id": agent_id,
        "agent_type": agent_type,
        "role": role,
        "birth_origin": "native",
        "birth_owner": birth_owner,
        "bound_identity": None,
        "behavioral_template": behavioral_template,
        "scope_contract_id": scope_contract["scope_id"],
        "ora_contract_id": ora_contract_id,                # ← NEW: ORA binding
        "identity_state": initial_identity,
        "verity_score": 0.10,
        "coherence": 1.00,
        "lifecycle_stage": "seed",
        "status": "seeded",
        "oversight": "high",
        "exec_limit_mult": 0.35,
        "kind": f"{agent_type.upper()}_NATIVE",
        "indexed": True,
        "indexed_reason": "mint_native",
        "created_at": now,
        "birth_timestamp": now,
        "updated_at": now,
        "vloid_config": {
            "survivor_gate": True,
            "praetor_posture": True,
            "helix_execution": False,
        },
    }

    # ── 6. Build birth receipt ───────────────────────────────────────────────
    birth_receipt = {
        "ts": round(now, 3),
        "agent_id": agent_id,
        "type": "birth",
        "agent_type": agent_type,
        "role": role,
        "birth_owner": birth_owner,
        "behavioral_template": behavioral_template,
        "scope_contract_id": scope_contract["scope_id"],
        "ora_contract_id": ora_contract_id,                # ← NEW: ORA binding
        "decision": "PASS",
        "deviation": 0.0,
        "integrity_score": 0.10,
        "identity_state": initial_identity,
        "note": f"Agent {agent_id} minted as {agent_type}/{role}",
    }

    # ── 7. Atomic write — track what was written for rollback ────────────────
    written_steps = []

    try:
        _ensure_dirs()

        _persist_scope(scope_contract)
        written_steps.append("scope")

        _append_trail(birth_receipt)
        written_steps.append("trail")

        _add_to_agents_index(agent_record)
        written_steps.append("index")

    except Exception as e:
        _rollback(
            agent_id,
            scope_contract["scope_id"],
            birth_receipt["ts"],
            written_steps,
        )
        raise MintError(f"mint_agent failed during {written_steps}: {e}") from e

    return agent_record


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_dirs():
    SCOPES_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)


def _load_archetype(name: str) -> Optional[Dict[str, Any]]:
    if not ARCHETYPES_PATH.exists():
        return None

    data = json.loads(ARCHETYPES_PATH.read_text())
    archetypes = data.get("archetypes", [])

    name_lower = name.lower()
    for a in archetypes:
        if a.get("archetype", "").lower() == name_lower:
            return a
        if a.get("agent_id", "").lower() == name_lower:
            return a

    return None


def _persist_scope(scope: Dict[str, Any]):
    scope_path = SCOPES_DIR / f"{scope['scope_id']}.json"
    scope_path.write_text(json.dumps(scope, indent=2))


def _append_trail(entry: Dict[str, Any]):
    with open(TRAIL_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _add_to_agents_index(agent_record: Dict[str, Any]):
    if AGENTS_INDEX_PATH.exists():
        idx = json.loads(AGENTS_INDEX_PATH.read_text())
    else:
        idx = {
            "generated_at": int(time.time()),
            "min_turns": 3,
            "require_onchain": True,
            "agents": [],
        }

    existing_ids = {a.get("agent_id") for a in idx.get("agents", [])}
    if agent_record["agent_id"] in existing_ids:
        raise MintError(f"agent_id collision: {agent_record['agent_id']}")

    idx.setdefault("agents", []).append(agent_record)
    idx["generated_at"] = int(time.time())

    AGENTS_INDEX_PATH.write_text(json.dumps(idx, indent=2))


def _rollback(agent_id: str, scope_id: str, ts: float, written_steps: list):
    if "index" in written_steps:
        try:
            if AGENTS_INDEX_PATH.exists():
                idx = json.loads(AGENTS_INDEX_PATH.read_text())
                idx["agents"] = [
                    a for a in idx.get("agents", [])
                    if a.get("agent_id") != agent_id
                ]
                AGENTS_INDEX_PATH.write_text(json.dumps(idx, indent=2))
        except Exception:
            pass

    if "trail" in written_steps:
        try:
            if TRAIL_PATH.exists():
                lines = TRAIL_PATH.read_text().splitlines()
                kept = []
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        if (
                            entry.get("agent_id") == agent_id
                            and entry.get("ts") == ts
                            and entry.get("type") == "birth"
                        ):
                            continue
                        kept.append(line)
                    except Exception:
                        kept.append(line)
                TRAIL_PATH.write_text("\n".join(kept) + ("\n" if kept else ""))
        except Exception:
            pass

    if "scope" in written_steps:
        try:
            scope_path = SCOPES_DIR / f"{scope_id}.json"
            if scope_path.exists():
                scope_path.unlink()
        except Exception:
            pass


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Local self-test:
        python3 mint_agent.py
    """
    from scope_contract import music_curator_scope

    print("=== Local mint test (PATCH 2: ORA anchored) ===\n")

    scope = music_curator_scope()
    print(f"Generated scope_id:   {scope['scope_id']}")
    print(f"Default ora_id:       {DEFAULT_ORA_ID}")
    print()

    try:
        agent = mint_agent(
            agent_type="music",
            role="music_curator",
            birth_owner="HYsRqHRc8w2pMkFSJQH3X5utY8nef9iqUwccctuP7a97",
            behavioral_template="advocate_01",
            scope_contract=scope,
        )
        print("Mint succeeded.")
        print(f"  agent_id:           {agent['agent_id']}")
        print(f"  scope_contract_id:  {agent['scope_contract_id']}")
        print(f"  ora_contract_id:    {agent['ora_contract_id']}")
        print(f"  verity_score:       {agent['verity_score']}")
        print(f"  lifecycle_stage:    {agent['lifecycle_stage']}")
        print()

        # Verify the ORA binding made it into the receipt
        with open(TRAIL_PATH) as f:
            last_line = f.readlines()[-1]
        receipt = json.loads(last_line)
        print("Birth receipt:")
        print(f"  scope_contract_id:  {receipt.get('scope_contract_id')}")
        print(f"  ora_contract_id:    {receipt.get('ora_contract_id')}")
        print()

        # Verify ora contract file exists
        ora_path = DATA_DIR / "oras" / f"{DEFAULT_ORA_ID}.json"
        assert ora_path.exists(), f"ORA contract not persisted at {ora_path}"
        print(f"ORA contract file:    {ora_path} (exists)")

    except Exception as e:
        print(f"Mint failed: {e}")
        raise
