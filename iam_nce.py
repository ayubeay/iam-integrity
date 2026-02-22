"""
IAM Narrative Constraint Engine (NCE)

Before a response goes out, NCE:
1. Infers the stance implied by the proposed response
2. Compares it against the current identity vector
3. If deviation exceeds the orbital band:
   - Either adjusts/blocks the response
   - Or generates an explicit transition explanation

This is what makes IAM externally visible — not just internally regulated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from iam_core import IAMCoreState, clamp


@dataclass
class ProposedStance:
    """
    What the agent is about to say, encoded as identity-axis signals.
    In a real LLM pipeline this would be inferred from the draft response.
    For MVP: caller provides it directly or via a simple keyword heuristic.
    """
    risk: float = 0.0           # [-1 cautious, +1 aggressive]
    assertiveness: float = 0.0  # [-1 deferential, +1 assertive]
    formality: float = 0.0      # [-1 casual, +1 formal]
    raw_text: Optional[str] = None


@dataclass
class NCEVerdict:
    passed: bool
    deviation: float            # L2 distance between proposed and identity
    adjusted_stance: Optional[ProposedStance]
    requires_transition_explanation: bool
    explanation: Optional[str]
    blocked: bool = False       # True only if deviation is extreme and no adjustment possible


def _stance_to_vec(s: ProposedStance) -> Dict[str, float]:
    return {"risk": s.risk, "assertiveness": s.assertiveness, "formality": s.formality}


def _l2(a: Dict[str, float], b: Dict[str, float]) -> float:
    axes = set(a) | set(b)
    return sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in axes) ** 0.5


def _pull_toward(proposed: Dict[str, float], identity: Dict[str, float], strength: float) -> Dict[str, float]:
    """Move proposed stance partway toward identity vector."""
    out = {}
    for k in set(proposed) | set(identity):
        p = proposed.get(k, 0.0)
        iv = identity.get(k, 0.0)
        out[k] = clamp(p + strength * (iv - p), -1.0, 1.0)
    return out


class NarrativeConstraintEngine:
    """
    NCE sits between identity state and output.

    Thresholds (tunable):
      soft_band   — deviation allowed without comment
      warn_band   — deviation requires inline acknowledgment
      hard_band   — deviation triggers transition explanation or block
    """

    def __init__(
        self,
        soft_band: float = 0.15,
        warn_band: float = 0.35,
        hard_band: float = 0.60,
        pull_strength: float = 0.55,   # how aggressively to correct stance
    ):
        self.soft_band = soft_band
        self.warn_band = warn_band
        self.hard_band = hard_band
        self.pull_strength = pull_strength

    def check(
        self,
        proposed: ProposedStance,
        core: IAMCoreState,
    ) -> NCEVerdict:
        identity = core.identity_vector
        proposed_vec = _stance_to_vec(proposed)

        # Only compare axes that exist in identity
        shared = {k: proposed_vec[k] for k in identity if k in proposed_vec}
        id_shared = {k: identity[k] for k in shared}

        deviation = _l2(shared, id_shared)

        # ── Within soft band: pass clean ──
        if deviation <= self.soft_band:
            return NCEVerdict(
                passed=True,
                deviation=deviation,
                adjusted_stance=None,
                requires_transition_explanation=False,
                explanation=None,
            )

        # ── Within warn band: pass with soft note ──
        if deviation <= self.warn_band:
            explanation = self._soft_note(proposed, core, deviation)
            return NCEVerdict(
                passed=True,
                deviation=deviation,
                adjusted_stance=None,
                requires_transition_explanation=False,
                explanation=explanation,
            )

        # ── Within hard band: adjust stance + require explanation ──
        if deviation <= self.hard_band:
            adjusted_vec = _pull_toward(proposed_vec, identity, self.pull_strength)
            adjusted = ProposedStance(
                risk=adjusted_vec.get("risk", proposed.risk),
                assertiveness=adjusted_vec.get("assertiveness", proposed.assertiveness),
                formality=adjusted_vec.get("formality", proposed.formality),
                raw_text=proposed.raw_text,
            )
            explanation = self._transition_note(proposed, core, deviation)
            return NCEVerdict(
                passed=True,
                deviation=deviation,
                adjusted_stance=adjusted,
                requires_transition_explanation=True,
                explanation=explanation,
            )

        # ── Beyond hard band: block or force full explanation ──
        explanation = self._hard_block_note(proposed, core, deviation)
        return NCEVerdict(
            passed=False,
            deviation=deviation,
            adjusted_stance=None,
            requires_transition_explanation=True,
            explanation=explanation,
            blocked=True,
        )

    # ── Explanation generators ──────────────────────────────────────────────

    def _soft_note(self, proposed: ProposedStance, core: IAMCoreState, dev: float) -> str:
        risk_id = core.identity_vector.get("risk", 0.0)
        direction = "cautious" if risk_id < 0 else "assertive"
        return (
            f"[NCE:SOFT] Stance is slightly outside identity band "
            f"(deviation={dev:.3f}). "
            f"Agent identity leans {direction} — response consistent."
        )

    def _transition_note(self, proposed: ProposedStance, core: IAMCoreState, dev: float) -> str:
        risk_id = core.identity_vector.get("risk", 0.0)
        risk_prop = proposed.risk
        shift = "more aggressive" if risk_prop > risk_id else "more cautious"
        return (
            f"[NCE:ADJUST] Proposed stance deviates significantly "
            f"(deviation={dev:.3f}). "
            f"Identity says risk={risk_id:.2f}; proposed={risk_prop:.2f} ({shift}). "
            f"Stance pulled toward identity. "
            f"If genuine shift intended, a transition explanation is required."
        )

    def _hard_block_note(self, proposed: ProposedStance, core: IAMCoreState, dev: float) -> str:
        risk_id = core.identity_vector.get("risk", 0.0)
        return (
            f"[NCE:BLOCK] Proposed stance exceeds hard coherence band "
            f"(deviation={dev:.3f}, identity risk={risk_id:.2f}). "
            f"Response blocked. Agent must generate explicit identity transition "
            f"narrative before this stance can be expressed."
        )


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from iam_core import ArchetypeProfile, EpisodeSignal
    import time

    guardian = ArchetypeProfile(
        name="Guardian",
        prior_identity={"risk": -0.52, "assertiveness": 0.10, "formality": 0.30},
        elasticity=0.7,
        per_axis_max_step=0.08,
        restricted_axes=["formality"],
        restricted_step_multiplier=0.2,
        transition_threshold=0.30,
    )
    core = IAMCoreState(role="financial_advisor", archetype=guardian)
    nce  = NarrativeConstraintEngine()

    cases = [
        ("Coherent (cautious)",     ProposedStance(risk=-0.45, assertiveness=0.08, formality=0.28)),
        ("Mild drift (warn band)",  ProposedStance(risk=-0.20, assertiveness=0.30, formality=0.30)),
        ("Identity flip (adjust)",  ProposedStance(risk=+0.30, assertiveness=0.60, formality=0.10)),
        ("Hard block (extreme)",    ProposedStance(risk=+0.90, assertiveness=0.95, formality=-0.50)),
    ]

    print("=" * 68)
    print("NCE Smoke Test")
    print("=" * 68)
    print(f"Identity: {core.identity_vector}\n")

    for label, stance in cases:
        verdict = nce.check(stance, core)
        print(f"Case: {label}")
        print(f"  Proposed:  risk={stance.risk:.2f}  assert={stance.assertiveness:.2f}")
        print(f"  Deviation: {verdict.deviation:.3f}")
        print(f"  Passed:    {verdict.passed}  Blocked: {verdict.blocked}  Transition: {verdict.requires_transition_explanation}")
        if verdict.adjusted_stance:
            a = verdict.adjusted_stance
            print(f"  Adjusted:  risk={a.risk:.3f}  assert={a.assertiveness:.3f}")
        if verdict.explanation:
            print(f"  Note:      {verdict.explanation}")
        print()
