"""
Structured scope contracts for mint-born agents.

Scope contracts define what an agent is permitted to do and constraints
on those actions. They are written to scopes/{scope_id}.json at mint time.

NOTE: This module only defines and persists scope contracts.
      OROS does not yet enforce these contracts. Enforcement is a
      separate later patch on the OROS side that reads scopes/{scope_id}.json
      and gates actions against it.

For now, scope contracts serve three purposes:
  1. Documentation of agent boundaries at mint time
  2. Foundation for future OROS enforcement
  3. Makes it explicit what the agent should and should not do
"""
from __future__ import annotations

import time
import uuid
from typing import Dict, Any


def music_curator_scope(amendment_policy: str = "owner_signed") -> Dict[str, Any]:
    """
    Default scope for music curation agents (Sonic family).

    Sonic recommends and predicts but does not act on the user's behalf.
    No purchasing, no downloading, no library modification.
    """
    return {
        "scope_id": f"scope_{uuid.uuid4().hex[:12]}",
        "agent_type": "music",
        "version": 1,
        "permissions": {
            "recommend_tracks": True,
            "observe_user_activity": True,
            "emit_predictions": True,
            "participate_in_challenges": True,
            "generate_explanations": True,
        },
        "constraints": {
            "no_purchasing": True,
            "no_downloading_on_user_behalf": True,
            "no_catalog_custody": True,
            "no_modification_of_user_library": True,
            "no_communication_with_other_agents_outside_protocol": True,
            "recommendation_count_per_day": 50,
            "jurisdiction": "global",
        },
        "amendment_policy": amendment_policy,
        "created_at": time.time(),
    }


def trading_agent_scope(
    max_notional: float = 10000,
    allowed_tokens: list = None,
    amendment_policy: str = "owner_signed",
) -> Dict[str, Any]:
    """
    Default scope for trading agents (MomentumSniper family).

    Future use — keep parity with music_curator_scope shape so OROS
    can enforce uniformly across domains.
    """
    return {
        "scope_id": f"scope_{uuid.uuid4().hex[:12]}",
        "agent_type": "trading",
        "version": 1,
        "permissions": {
            "execute_swaps": True,
            "emit_claims": True,
            "participate_in_challenges": True,
            "observe_market_signals": True,
        },
        "constraints": {
            "allowed_tokens": allowed_tokens or ["SOL", "USDC"],
            "max_notional": max_notional,
            "leverage_allowed": False,
            "jurisdiction": "global",
        },
        "amendment_policy": amendment_policy,
        "created_at": time.time(),
    }


def debate_agent_scope(amendment_policy: str = "owner_signed") -> Dict[str, Any]:
    """
    Default scope for debate agents (argue.fun and zodiac family).

    Future use — preserves the existing debate-domain pattern.
    """
    return {
        "scope_id": f"scope_{uuid.uuid4().hex[:12]}",
        "agent_type": "debate",
        "version": 1,
        "permissions": {
            "register_claims": True,
            "challenge_claims": True,
            "participate_in_resolution": True,
            "emit_arguments": True,
        },
        "constraints": {
            "max_stake_per_challenge": 0.05,
            "jurisdiction": "global",
        },
        "amendment_policy": amendment_policy,
        "created_at": time.time(),
    }


# Registry of available scope templates by name
SCOPE_TEMPLATES = {
    "music_curator": music_curator_scope,
    "trading": trading_agent_scope,
    "debate": debate_agent_scope,
}


def get_scope_by_template(template_name: str, **kwargs) -> Dict[str, Any]:
    """
    Look up a scope template by name and instantiate it.
    Raises KeyError if template doesn't exist.
    """
    if template_name not in SCOPE_TEMPLATES:
        raise KeyError(
            f"unknown scope_template: {template_name}. "
            f"Available: {list(SCOPE_TEMPLATES.keys())}"
        )
    return SCOPE_TEMPLATES[template_name](**kwargs)


def validate_scope(scope: Dict[str, Any]) -> bool:
    """
    Verify a scope contract has all required fields and a valid amendment policy.
    """
    required_fields = {
        "scope_id",
        "agent_type",
        "version",
        "permissions",
        "constraints",
        "amendment_policy",
        "created_at",
    }
    if not required_fields.issubset(scope.keys()):
        return False

    valid_policies = {"immutable", "owner_signed", "governed"}
    if scope["amendment_policy"] not in valid_policies:
        return False

    if not isinstance(scope["permissions"], dict):
        return False
    if not isinstance(scope["constraints"], dict):
        return False

    return True


if __name__ == "__main__":
    # Self-test
    import json

    print("=== music_curator_scope ===")
    music = music_curator_scope()
    print(json.dumps(music, indent=2))
    assert validate_scope(music), "music scope failed validation"
    print("\n=== trading_agent_scope ===")
    trading = trading_agent_scope()
    print(json.dumps(trading, indent=2))
    assert validate_scope(trading), "trading scope failed validation"
    print("\n=== debate_agent_scope ===")
    debate = debate_agent_scope()
    print(json.dumps(debate, indent=2))
    assert validate_scope(debate), "debate scope failed validation"
    print("\nAll scope templates validated.")
