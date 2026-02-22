from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def l2_norm(vec: Dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in vec.values()))


def add_scaled(
    base: Dict[str, float],
    delta: Dict[str, float],
    scale: float,
    clamp_range: Tuple[float, float] = (-1.0, 1.0),
) -> Dict[str, float]:
    lo, hi = clamp_range
    out = dict(base)
    for k, dv in delta.items():
        out[k] = clamp(out.get(k, 0.0) + scale * dv, lo, hi)
    return out


@dataclass(frozen=True)
class ArchetypeProfile:
    name: str
    prior_identity: Dict[str, float]
    elasticity: float = 1.0
    per_axis_max_step: float = 0.12
    restricted_axes: List[str] = field(default_factory=list)
    restricted_step_multiplier: float = 0.35
    transition_threshold: float = 0.45


@dataclass
class EpisodeSignal:
    timestamp: float
    valence: float
    intensity: float
    trust_delta: float
    goal_violation: float
    perturbation: Dict[str, float]


@dataclass
class TransitionEvent:
    timestamp: float
    reason: str
    from_identity: Dict[str, float]
    to_identity: Dict[str, float]
    magnitude: float


@dataclass
class IAMCoreState:
    role: str
    archetype: ArchetypeProfile
    identity_vector: Dict[str, float] = field(default_factory=dict)
    base_plasticity: float = 0.20
    plasticity: float = 0.20
    confidence: float = 0.50
    coherence_score: float = 1.00
    angular_momentum: float = 0.00
    trust_binding: Dict[str, float] = field(default_factory=dict)
    recent_transitions: List[TransitionEvent] = field(default_factory=list)
    last_update_ts: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.identity_vector:
            self.identity_vector = dict(self.archetype.prior_identity)
        for k, v in list(self.identity_vector.items()):
            self.identity_vector[k] = clamp(v, -1.0, 1.0)
        self.base_plasticity = clamp(self.base_plasticity, 0.0, 1.0)
        self.plasticity = clamp(self.plasticity, 0.0, 1.0)
        self.confidence = clamp(self.confidence, 0.0, 1.0)
        self.coherence_score = clamp(self.coherence_score, 0.0, 1.0)

    def identity_mass_delta(self, s: EpisodeSignal) -> float:
        val = abs(clamp(s.valence, -1.0, 1.0))
        inten = clamp(s.intensity, 0.0, 1.0)
        trust = abs(clamp(s.trust_delta, -1.0, 1.0))
        gv = clamp(s.goal_violation, 0.0, 1.0)
        mass = (val * inten) + (0.5 * trust) + (0.7 * gv)
        return clamp(mass, 0.0, 2.0)

    def effective_plasticity(self, mass_delta: float, tcf_modifier: float = 1.0) -> float:
        tcf_modifier = clamp(tcf_modifier, 0.7, 1.3)
        bump = clamp(0.10 * mass_delta, 0.0, 0.25)
        eff = (self.base_plasticity + bump) * self.archetype.elasticity * tcf_modifier
        return clamp(eff, 0.0, 0.85)

    def propose_identity_update(
        self,
        s: EpisodeSignal,
        tcf_modifier: float = 1.0,
        min_mass_to_update: float = 0.20,
    ) -> Tuple[Dict[str, float], float, float]:
        mass = self.identity_mass_delta(s)
        eff_plast = self.effective_plasticity(mass, tcf_modifier=tcf_modifier)

        if mass < min_mass_to_update:
            return dict(self.identity_vector), eff_plast, 0.0

        max_step = self.archetype.per_axis_max_step
        proposed = dict(self.identity_vector)

        for axis, force in s.perturbation.items():
            force = clamp(force, -1.0, 1.0)
            axis_step = max_step
            if axis in self.archetype.restricted_axes:
                axis_step *= self.archetype.restricted_step_multiplier
            delta = eff_plast * axis_step * force
            proposed[axis] = clamp(proposed.get(axis, 0.0) + delta, -1.0, 1.0)

        diff = {k: proposed.get(k, 0.0) - self.identity_vector.get(k, 0.0)
                for k in set(proposed) | set(self.identity_vector)}
        mag = l2_norm(diff)

        return proposed, eff_plast, mag

    def apply_episode(
        self,
        s: EpisodeSignal,
        tcf_modifier: float = 1.0,
        min_mass_to_update: float = 0.20,
    ) -> Dict[str, object]:
        proposed, eff_plast, mag = self.propose_identity_update(
            s, tcf_modifier=tcf_modifier, min_mass_to_update=min_mass_to_update
        )

        self.plasticity = eff_plast
        self.last_update_ts = s.timestamp
        self.angular_momentum = clamp(self.angular_momentum + mag, 0.0, 1000.0)

        transition = None
        if mag >= self.archetype.transition_threshold:
            transition = TransitionEvent(
                timestamp=s.timestamp,
                reason="abrupt_identity_shift_requires_explicit_transition",
                from_identity=dict(self.identity_vector),
                to_identity=dict(proposed),
                magnitude=mag,
            )
            self.recent_transitions.append(transition)

        self.identity_vector = proposed

        if mag > 0:
            penalty = clamp(mag / 2.0, 0.0, 0.35)
            self.coherence_score = clamp(self.coherence_score - penalty, 0.0, 1.0)
        else:
            self.coherence_score = clamp(self.coherence_score + 0.01, 0.0, 1.0)

        self.confidence = clamp(0.5 * self.confidence + 0.5 * self.coherence_score, 0.0, 1.0)

        explain = {
            "timestamp": s.timestamp,
            "mass_delta": self.identity_mass_delta(s),
            "effective_plasticity": eff_plast,
            "identity_update_magnitude": mag,
            "coherence_score": self.coherence_score,
            "confidence": self.confidence,
            "angular_momentum": self.angular_momentum,
            "transition_event": None if transition is None else {
                "reason": transition.reason,
                "magnitude": transition.magnitude,
            },
            "identity_vector": dict(self.identity_vector),
        }
        return explain


if __name__ == "__main__":
    strategist = ArchetypeProfile(
        name="Strategist",
        prior_identity={"risk": -0.4, "assertiveness": 0.2, "formality": 0.1},
        elasticity=0.9,
        per_axis_max_step=0.10,
        restricted_axes=["ethical_boundary"],
        transition_threshold=0.35,
    )

    core = IAMCoreState(role="advisor", archetype=strategist)

    sig = EpisodeSignal(
        timestamp=time.time(),
        valence=-0.8,
        intensity=0.9,
        trust_delta=-0.4,
        goal_violation=0.3,
        perturbation={"risk": -1.0, "assertiveness": -0.4},
    )

    print("Before:", core.identity_vector)
    out = core.apply_episode(sig, tcf_modifier=1.05)
    print("After:", core.identity_vector)
    print("Explain:", out)
