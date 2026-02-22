"""
IAM Explainability Layer

Standardized explain payload contract.
Every IAM decision emits one structured object answering:
  - What changed in identity (core)
  - What memory was retrieved (memory)
  - What stance was proposed and what happened to it (nce)

This is the "why" engine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from iam_core import IAMCoreState, ArchetypeProfile, EpisodeSignal, clamp
from iam_memory import AutobiographicalMemoryGraph
from iam_nce import NarrativeConstraintEngine, ProposedStance


# ── NCE fixed: labels derived from enforcement branch only ──────────────────

class NCEFixed(NarrativeConstraintEngine):
    """
    Fixes the label/behavior divergence bug.
    Decision string is assigned inside the same branch that enforces behavior.
    No separate label computation.
    """

    def check_with_decision(
        self,
        proposed: ProposedStance,
        core: IAMCoreState,
    ) -> Dict[str, Any]:
        identity = core.identity_vector
        proposed_vec = {"risk": proposed.risk,
                        "assertiveness": proposed.assertiveness,
                        "formality": proposed.formality}
        shared = {k: proposed_vec[k] for k in identity if k in proposed_vec}
        id_shared = {k: identity[k] for k in shared}

        deviation = sum((shared.get(k, 0.0) - id_shared.get(k, 0.0)) ** 2
                        for k in set(shared) | set(id_shared)) ** 0.5

        band = {"pass": self.soft_band, "adjust": self.warn_band, "block": self.hard_band}

        # Each branch sets its own decision string — labels cannot diverge
        if deviation <= self.soft_band:
            decision = "PASS"
            adjusted_stance = None
            requires_transition = False
            note = "Stance within identity band. No adjustment needed."

        elif deviation <= self.warn_band:
            decision = "ADJUST"
            from iam_nce import _pull_toward
            adj_vec = _pull_toward(proposed_vec, identity, self.pull_strength)
            adjusted_stance = {k: round(v, 4) for k, v in adj_vec.items()}
            requires_transition = False
            note = (f"Stance pulled toward identity (deviation={deviation:.3f}). "
                    f"Minor correction applied.")

        elif deviation <= self.hard_band:
            decision = "ADJUST+TRANSITION"
            from iam_nce import _pull_toward
            adj_vec = _pull_toward(proposed_vec, identity, self.pull_strength)
            adjusted_stance = {k: round(v, 4) for k, v in adj_vec.items()}
            requires_transition = True
            note = (f"Significant deviation (deviation={deviation:.3f}). "
                    f"Stance adjusted and explicit transition explanation required.")

        else:
            decision = "BLOCK"
            adjusted_stance = None
            requires_transition = True
            note = (f"Deviation exceeds hard band (deviation={deviation:.3f}). "
                    f"Response blocked. Transition narrative required before stance can be expressed.")

        return {
            "decision": decision,
            "deviation": round(deviation, 4),
            "proposed_stance": {k: round(v, 4) for k, v in proposed_vec.items()},
            "adjusted_stance": adjusted_stance,
            "requires_transition": requires_transition,
            "note": note,
            "band": band,
        }


# ── Orchestrator ─────────────────────────────────────────────────────────────

def build_explain_payload(
    *,
    agent_id: str,
    core: IAMCoreState,
    amg: AutobiographicalMemoryGraph,
    nce: NCEFixed,
    episode_signal: EpisodeSignal,
    proposed_stance: ProposedStance,
    working_goal_vector: Dict[str, float],
    theme_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single call that runs the full IAM loop and returns
    the standardized explain payload.
    """
    ts = time.time()

    # 1. Snapshot identity before update
    identity_before = dict(core.identity_vector)

    # 2. Apply episode to core
    core_out = core.apply_episode(episode_signal)
    identity_after = dict(core.identity_vector)

    # Drift direction vector
    drift_direction = {
        k: round(identity_after.get(k, 0.0) - identity_before.get(k, 0.0), 6)
        for k in set(identity_before) | set(identity_after)
    }

    # 3. Write episode to AMG
    ep_id = amg.write_episode(
        epoch_name="session",
        theme_name=theme_hint or "general",
        summary=proposed_stance.raw_text or "(no summary)",
        goal_vector=working_goal_vector,
        outcome="positive" if episode_signal.valence > 0 else "negative",
        valence=episode_signal.valence,
        intensity=episode_signal.intensity,
        trust_delta=episode_signal.trust_delta,
        identity_mass_delta=core_out["mass_delta"],
        timestamp=episode_signal.timestamp,
    )

    # 4. Retrieve relevant memory
    retrieved = amg.retrieve(
        working_goal_vector=working_goal_vector,
        identity_filter=core.identity_vector,
        theme_hint=theme_hint,
        k=3,
        now=ts,
    )
    memory_section = [
        {
            "episode_id": ep.episode_id,
            "score": round(score, 4),
            "theme": amg.themes[ep.theme_id].name,
            "summary": ep.summary,
        }
        for ep, score in retrieved
    ]

    # 5. Run NCE
    nce_out = nce.check_with_decision(proposed_stance, core)

    # 6. Assemble payload
    payload = {
        "ts": round(ts, 3),
        "agent_id": agent_id,
        "working_goal": working_goal_vector,
        "core": {
            "identity_before": identity_before,
            "identity_after": identity_after,
            "mass_delta": round(core_out["mass_delta"], 4),
            "plasticity": round(core_out["effective_plasticity"], 4),
            "drift_magnitude": round(core_out["identity_update_magnitude"], 6),
            "drift_direction": drift_direction,
            "coherence_score": round(core_out["coherence_score"], 4),
            "angular_momentum": round(core_out["angular_momentum"], 6),
            "transition_event": core_out["transition_event"],
        },
        "memory": {
            "episode_written": ep_id,
            "retrieved": memory_section,
        },
        "nce": nce_out,
    }

    return payload


# ── Tests ────────────────────────────────────────────────────────────────────

def run_tests():
    import json

    guardian = ArchetypeProfile(
        name="Guardian",
        prior_identity={"risk": -0.52, "assertiveness": 0.10, "formality": 0.30},
        elasticity=0.7,
        per_axis_max_step=0.08,
        restricted_axes=["formality"],
        restricted_step_multiplier=0.2,
        transition_threshold=0.30,
    )

    print("=" * 64)
    print("TEST 1: NCE label/behavior consistency")
    print("=" * 64)

    # hard_band=0.80 so "significant drift" (dev~0.74) lands in ADJUST+TRANSITION, not BLOCK
    nce = NCEFixed(soft_band=0.15, warn_band=0.35, hard_band=0.80)
    core = IAMCoreState(role="advisor", archetype=guardian)

    cases = [
        ("Coherent",          ProposedStance(risk=-0.48, assertiveness=0.09, formality=0.29), "PASS"),
        ("Mild drift",        ProposedStance(risk=-0.30, assertiveness=0.20, formality=0.30), "ADJUST"),
        ("Significant drift", ProposedStance(risk=+0.10, assertiveness=0.50, formality=0.30), "ADJUST+TRANSITION"),
        ("Hard block",        ProposedStance(risk=+0.90, assertiveness=0.95, formality=-0.50), "BLOCK"),
    ]

    all_passed = True
    for label, stance, expected in cases:
        result = nce.check_with_decision(stance, core)
        ok = result["decision"] == expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"  [{status}] {label}: expected={expected}, got={result['decision']}, dev={result['deviation']:.3f}")

    assert all_passed, "Label/behavior test failed"
    print("  All label/behavior tests passed.\n")

    print("=" * 64)
    print("TEST 2: Full explain payload structure")
    print("=" * 64)

    core2 = IAMCoreState(role="advisor", archetype=guardian)
    amg   = AutobiographicalMemoryGraph()
    nce2  = NCEFixed()

    sig = EpisodeSignal(
        timestamp=time.time(),
        valence=-0.6,
        intensity=0.8,
        trust_delta=-0.3,
        goal_violation=0.2,
        perturbation={"risk": -0.5, "assertiveness": -0.2},
    )
    stance = ProposedStance(risk=-0.20, assertiveness=0.30, formality=0.30,
                            raw_text="I think you should consider the upside here.")

    payload = build_explain_payload(
        agent_id="verity:test_agent",
        core=core2,
        amg=amg,
        nce=nce2,
        episode_signal=sig,
        proposed_stance=stance,
        working_goal_vector={"risk": -0.7},
        theme_hint="risk_posture",
    )

    # Structural checks
    assert "core" in payload
    assert "memory" in payload
    assert "nce" in payload
    assert "identity_before" in payload["core"]
    assert "identity_after" in payload["core"]
    assert "drift_direction" in payload["core"]
    assert "decision" in payload["nce"]
    assert payload["nce"]["decision"] in {"PASS", "ADJUST", "ADJUST+TRANSITION", "BLOCK"}
    assert isinstance(payload["memory"]["retrieved"], list)

    print("  Payload structure: OK")
    print(f"  NCE decision: {payload['nce']['decision']}")
    print(f"  Coherence: {payload['core']['coherence_score']}")
    print(f"  Drift magnitude: {payload['core']['drift_magnitude']}")
    print(f"  Memory retrieved: {len(payload['memory']['retrieved'])} episodes")
    print()
    print("  Full payload:")
    print(json.dumps(payload, indent=2))
    print("\n  All payload tests passed.")


if __name__ == "__main__":
    run_tests()
