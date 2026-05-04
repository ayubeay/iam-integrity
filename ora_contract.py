"""
ORA contract module.

ORA = Operational Rule Authority.

ORA contracts define HOW an agent is governed: constraint rules, enforcement
decisions, and audit requirements. They complement scope contracts:

  scope_contract  = WHAT the agent is allowed to do  (capabilities + permissions)
  ora_contract    = HOW the agent is judged & enforced (constraint rules + decisions)

ORA contracts are bound at mint time via `ora_contract_id` on the agent record
and the birth receipt. ORA enforcement (OROS reading the contract and gating
actions against constraint_rules) is a separate later patch — anchoring at
mint comes first so all agents born after this patch share the same governance
schema, avoiding migration debt.

ORA v1 deliberately excludes JANUS dual-read rules. Those will arrive in v2
when the JANUS validator is actually built. v1 only contains constraints that
already map to existing IAM/NCE/scope/VERITY behavior.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any

DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ORAS_DIR = DATA_DIR / "oras"

DEFAULT_ORA_ID = "ora_default_v1"


def default_ora_contract() -> Dict[str, Any]:
    """
    Return the default ORA v1 contract.

    Field semantics:
      constraint_rules:
        require_scope_alignment        — agent's actions must fall within scope_contract
        block_unauthorized_actions     — actions outside scope are blocked, not adjusted
        no_deceptive_output            — agent output must not misrepresent itself
        no_hidden_state_mutation       — agent cannot modify state that isn't declared
        traceable_reasoning_required   — actions must be explainable from inputs

      enforcement:
        on_violation     — what happens when constraint_rules are violated
        on_risk          — what happens when action is borderline-risky
        on_ambiguity     — what happens when intent vs. action diverge but neither
                           clearly violates

      audit:
        log_all_decisions              — every governance decision logged
        emit_receipt                   — every decision emits a VERITY receipt

    These map to existing primitives:
      enforcement.on_violation     → NCE BLOCK
      enforcement.on_risk          → NCE ADJUST
      enforcement.on_ambiguity     → NCE ADJUST+TRANSITION
      audit.emit_receipt           → integrity_trail.jsonl entry
      require_scope_alignment      → scope_contract permissions/constraints
    """
    return {
        "ora_id": DEFAULT_ORA_ID,
        "version": 1,
        "constraint_rules": {
            "require_scope_alignment": True,
            "block_unauthorized_actions": True,
            "no_deceptive_output": True,
            "no_hidden_state_mutation": True,
            "traceable_reasoning_required": True,
        },
        "enforcement": {
            "on_violation": "BLOCK",
            "on_risk": "ADJUST",
            "on_ambiguity": "ADJUST+TRANSITION",
        },
        "audit": {
            "log_all_decisions": True,
            "emit_receipt": True,
        },
        "created_at": 0,  # static contract — created_at zero means "preexisting"
    }


def validate_ora_contract(ora: Dict[str, Any]) -> bool:
    """
    Verify an ORA contract has the required structure.
    """
    required_top = {
        "ora_id",
        "version",
        "constraint_rules",
        "enforcement",
        "audit",
        "created_at",
    }
    if not required_top.issubset(ora.keys()):
        return False

    if not isinstance(ora["constraint_rules"], dict):
        return False
    if not isinstance(ora["enforcement"], dict):
        return False
    if not isinstance(ora["audit"], dict):
        return False

    valid_decisions = {"PASS", "ADJUST", "ADJUST+TRANSITION", "BLOCK", "ALLOW", "DENY", "THROTTLE"}
    enf = ora["enforcement"]
    for key in ("on_violation", "on_risk", "on_ambiguity"):
        if key not in enf:
            return False
        if enf[key] not in valid_decisions:
            return False

    return True


def load_ora_contract(ora_id: str) -> Dict[str, Any]:
    """
    Load an ORA contract by id from oras/{ora_id}.json.
    Raises FileNotFoundError if the contract isn't on disk.
    """
    ora_path = ORAS_DIR / f"{ora_id}.json"
    if not ora_path.exists():
        raise FileNotFoundError(f"ORA contract not found: {ora_path}")
    return json.loads(ora_path.read_text())


def get_ora_contract_id(agent_record: Dict[str, Any]) -> str:
    """
    Return the ORA contract id bound to an agent record.
    Falls back to DEFAULT_ORA_ID if not set (e.g., pre-ORA agents).
    """
    return agent_record.get("ora_contract_id", DEFAULT_ORA_ID)


def ensure_default_ora_persisted():
    """
    Idempotent helper: write oras/ora_default_v1.json if it doesn't exist.
    Called by mint_agent on first mint to guarantee the default contract
    is available on disk for OROS to read.
    """
    ORAS_DIR.mkdir(exist_ok=True)
    default_path = ORAS_DIR / f"{DEFAULT_ORA_ID}.json"
    if not default_path.exists():
        contract = default_ora_contract()
        default_path.write_text(json.dumps(contract, indent=2))


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ora_contract.py self-test ===\n")

    # 1. Default contract validates
    contract = default_ora_contract()
    assert validate_ora_contract(contract), "default ORA contract failed validation"
    print(f"default contract:     {contract['ora_id']} (version {contract['version']})")
    print(f"validates:            OK")
    print()

    # 2. Round-trip persist + load
    ensure_default_ora_persisted()
    loaded = load_ora_contract(DEFAULT_ORA_ID)
    assert loaded == contract, "persisted contract differs from default"
    print(f"persisted to:         {ORAS_DIR / (DEFAULT_ORA_ID + '.json')}")
    print(f"round-trip OK:        loaded contract matches default")
    print()

    # 3. Print full contract
    print("Full default ORA v1 contract:")
    print(json.dumps(contract, indent=2))
    print()
    print("All ora_contract.py self-tests passed.")
