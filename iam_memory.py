from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import math
import uuid


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def dot(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(a.get(k, 0.0) * b.get(k, 0.0) for k in a.keys())


def norm(v: Dict[str, float]) -> float:
    return math.sqrt(sum(x * x for x in v.values()))


def cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return clamp(dot(a, b) / (na * nb), -1.0, 1.0)


@dataclass
class Episode:
    episode_id: str
    timestamp: float
    epoch_id: str
    theme_id: str
    summary: str
    artifacts: Dict[str, str] = field(default_factory=dict)
    goal_vector: Dict[str, float] = field(default_factory=dict)
    outcome: str = "unknown"
    valence: float = 0.0
    intensity: float = 0.0
    trust_delta: float = 0.0
    identity_mass_delta: float = 0.0
    orbital_stability: float = 0.8
    decay_rate: float = 0.01


@dataclass
class Theme:
    theme_id: str
    name: str
    created_at: float
    episodes: List[str] = field(default_factory=list)


@dataclass
class Epoch:
    epoch_id: str
    name: str
    start_ts: float
    end_ts: Optional[float] = None
    themes: Dict[str, str] = field(default_factory=dict)


@dataclass
class AutobiographicalMemoryGraph:
    epochs: Dict[str, Epoch] = field(default_factory=dict)
    themes: Dict[str, Theme] = field(default_factory=dict)
    episodes: Dict[str, Episode] = field(default_factory=dict)
    active_epoch_id: Optional[str] = None

    def ensure_epoch(self, name: str, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        if self.active_epoch_id and self.epochs[self.active_epoch_id].name == name:
            return self.active_epoch_id
        epoch_id = f"ep_{uuid.uuid4().hex[:10]}"
        self.epochs[epoch_id] = Epoch(epoch_id=epoch_id, name=name, start_ts=now)
        self.active_epoch_id = epoch_id
        return epoch_id

    def ensure_theme(self, epoch_id: str, theme_name: str, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        epoch = self.epochs[epoch_id]
        if theme_name in epoch.themes:
            return epoch.themes[theme_name]
        theme_id = f"th_{uuid.uuid4().hex[:10]}"
        self.themes[theme_id] = Theme(theme_id=theme_id, name=theme_name, created_at=now)
        epoch.themes[theme_name] = theme_id
        return theme_id

    def write_episode(
        self,
        epoch_name: str,
        theme_name: str,
        summary: str,
        *,
        goal_vector: Dict[str, float],
        outcome: str,
        valence: float,
        intensity: float,
        trust_delta: float,
        identity_mass_delta: float,
        artifacts: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None,
    ) -> str:
        ts = time.time() if timestamp is None else timestamp
        epoch_id = self.ensure_epoch(epoch_name, now=ts)
        theme_id = self.ensure_theme(epoch_id, theme_name, now=ts)
        ep_id = f"ev_{uuid.uuid4().hex[:12]}"
        stability = 0.65 + 0.25 * clamp(identity_mass_delta / 2.0, 0.0, 1.0) - 0.15 * clamp(intensity - 0.8, 0.0, 1.0)
        episode = Episode(
            episode_id=ep_id,
            timestamp=ts,
            epoch_id=epoch_id,
            theme_id=theme_id,
            summary=summary,
            artifacts=artifacts or {},
            goal_vector=dict(goal_vector),
            outcome=outcome,
            valence=clamp(valence, -1.0, 1.0),
            intensity=clamp(intensity, 0.0, 1.0),
            trust_delta=clamp(trust_delta, -1.0, 1.0),
            identity_mass_delta=clamp(identity_mass_delta, 0.0, 2.0),
            orbital_stability=clamp(stability, 0.05, 1.0),
            decay_rate=0.01,
        )
        self.episodes[ep_id] = episode
        self.themes[theme_id].episodes.append(ep_id)
        return ep_id

    def retrieve(
        self,
        *,
        working_goal_vector: Dict[str, float],
        identity_filter: Dict[str, float],
        theme_hint: Optional[str] = None,
        k: int = 5,
        now: Optional[float] = None,
    ) -> List[Tuple[Episode, float]]:
        now = time.time() if now is None else now
        candidates: List[Episode] = []

        if self.active_epoch_id:
            epoch = self.epochs[self.active_epoch_id]
            if theme_hint and theme_hint in epoch.themes:
                theme_id = epoch.themes[theme_hint]
                candidates = [self.episodes[eid] for eid in self.themes[theme_id].episodes]
            else:
                for tid in epoch.themes.values():
                    candidates.extend(self.episodes[eid] for eid in self.themes[tid].episodes)
        else:
            candidates = list(self.episodes.values())

        scored: List[Tuple[Episode, float]] = []
        for ep in candidates:
            g_align = (cosine_sim(ep.goal_vector, working_goal_vector) + 1.0) / 2.0
            age_days = max(0.0, (now - ep.timestamp) / 86400.0)
            decay = math.exp(-ep.decay_rate * age_days)
            id_align = (cosine_sim(identity_filter, ep.goal_vector) + 1.0) / 2.0
            mass = clamp(ep.identity_mass_delta / 2.0, 0.0, 1.0)
            score = mass * g_align * ep.orbital_stability * decay * (0.5 + 0.5 * id_align)
            if score > 0.0:
                scored.append((ep, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
