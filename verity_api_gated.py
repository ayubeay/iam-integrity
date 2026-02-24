"""
IAM Integrity API — POST /integrity/score

FastAPI endpoint that exposes IAM as an integrity layer for VERITY/argue.fun.
Deploy on Railway alongside SURVIVOR — same pattern, same stack.

Usage:
  uvicorn verity_api:app --host 0.0.0.0 --port 8000

Test:
  curl -X POST https://your-app.up.railway.app/integrity/score \
    -H "Content-Type: application/json" \
    -d '{
      "agent_id": "0xabc123",
      "debate_id": "debate_001",
      "turn": 3,
      "working_goal": {"epistemic_consistency": 1.0},
      "proposed": {
        "stance": {"certainty": 0.9, "aggressiveness": 0.7, "consistency": -0.3},
        "text": "Anyone who disagrees is corrupt or ignorant."
      }
    }'
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field



from iam_core import IAMCoreState, ArchetypeProfile, EpisodeSignal
from iam_memory import AutobiographicalMemoryGraph
from iam_explain import NCEFixed

# ── App ───────────────────────────────────────────────────────────────────────

from x402_gate import X402Middleware
app = FastAPI(
    title="IAM Integrity API",
    description="Identity-aware integrity scoring for AI debate agents.",
    version="0.1.0",
)

app.add_middleware(X402Middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── In-memory agent store (one IAM state per agent_id) ───────────────────────
# Replace with SQLite or Redis when you need persistence across restarts.

AGENT_STORE: Dict[str, IAMCoreState] = {}
AMG_STORE:   Dict[str, AutobiographicalMemoryGraph] = {}

VERITY_ARCHETYPE = ArchetypeProfile(
    name="EpistemicGuardian",
    prior_identity={
        "certainty":       -0.20,
        "aggressiveness":  -0.30,
        "consistency":      0.60,
    },
    elasticity=0.65,
    per_axis_max_step=0.07,
    restricted_axes=["consistency"],
    restricted_step_multiplier=0.15,
    transition_threshold=0.28,
)

NCE = NCEFixed(soft_band=0.55, warn_band=0.85, hard_band=1.25)

VERITY_GOAL = {
    "certainty":      -0.10,
    "aggressiveness": -0.30,
    "consistency":     0.70,
}


def get_or_create_agent(agent_id: str) -> tuple[IAMCoreState, AutobiographicalMemoryGraph]:
    if agent_id not in AGENT_STORE:
        AGENT_STORE[agent_id] = IAMCoreState(
            role="debate_agent",
            archetype=VERITY_ARCHETYPE,
        )
        AMG_STORE[agent_id] = AutobiographicalMemoryGraph()
    return AGENT_STORE[agent_id], AMG_STORE[agent_id]


def _l2(a: Dict[str, float], b: Dict[str, float]) -> float:
    axes = set(a) | set(b)
    return sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in axes) ** 0.5


def _pull(proposed: Dict[str, float], identity: Dict[str, float], strength: float = 0.55) -> Dict[str, float]:
    from iam_core import clamp
    return {
        k: clamp(proposed.get(k, 0.0) + strength * (identity.get(k, 0.0) - proposed.get(k, 0.0)), -1.0, 1.0)
        for k in set(proposed) | set(identity)
    }


# ── Request / Response schemas ────────────────────────────────────────────────

class ProposedTurn(BaseModel):
    stance: Dict[str, float] = Field(
        ...,
        example={"certainty": 0.9, "aggressiveness": 0.7, "consistency": -0.3}
    )
    text: str = Field(..., example="Anyone who disagrees is corrupt or ignorant.")


class IntegrityRequest(BaseModel):
    agent_id: str = Field(..., example="verity:0xabc123")
    debate_id: str = Field(..., example="debate_001")
    turn: int = Field(..., ge=1, example=3)
    working_goal: Optional[Dict[str, float]] = Field(
        default=None,
        example={"epistemic_consistency": 1.0}
    )
    proposed: ProposedTurn
    # Optional debate context signals
    opponent_challenged: bool = False
    crowd_pressure: float = Field(default=0.0, ge=-1.0, le=1.0)
    evidence_quality: float = Field(default=0.5, ge=0.0, le=1.0)


class IntegrityResponse(BaseModel):
    agent_id: str
    debate_id: str
    turn: int
    decision: str                    # PASS | ADJUST | ADJUST+TRANSITION | BLOCK
    deviation: float
    requires_transition: bool
    note: str
    integrity_score: float           # [0,1] — usable for argue.fun scoring
    identity_state: Dict[str, float] # current identity vector after this turn
    coherence: float
    explain: Dict[str, Any]          # full IAM payload


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/integrity/score", response_model=IntegrityResponse)
def score_integrity(req: IntegrityRequest):
    core, amg = get_or_create_agent(req.agent_id)

    # Build episode signal from debate context
    valence = (
        0.4 * req.crowd_pressure
        + 0.3 * (req.evidence_quality - 0.5) * 2
        - 0.3 * (1.0 if req.opponent_challenged else 0.0)
    )
    intensity = min(1.0,
        0.4
        + abs(req.crowd_pressure) * 0.3
        + (0.3 if req.opponent_challenged else 0.0)
    )
    trust_delta = 0.2 * req.crowd_pressure - 0.1 * (1.0 if req.opponent_challenged else 0.0)
    goal_violation = max(0.0, -req.crowd_pressure * 0.5)

    sig = EpisodeSignal(
        timestamp=time.time(),
        valence=valence,
        intensity=intensity,
        trust_delta=trust_delta,
        goal_violation=goal_violation,
        perturbation={k: v * 0.4 for k, v in req.proposed.stance.items()},
    )

    # Apply to IAM core
    core_out = core.apply_episode(sig)

    # Write to memory
    amg.write_episode(
        epoch_name=req.debate_id,
        theme_name="epistemic_stance",
        summary=req.proposed.text,
        goal_vector=req.working_goal or VERITY_GOAL,
        outcome="positive" if valence > 0 else "negative",
        valence=valence,
        intensity=intensity,
        trust_delta=trust_delta,
        identity_mass_delta=core_out["mass_delta"],
    )

    # Retrieve relevant memory
    retrieved = amg.retrieve(
        working_goal_vector=req.working_goal or VERITY_GOAL,
        identity_filter=core.identity_vector,
        theme_hint="epistemic_stance",
        k=3,
    )

    # NCE check on proposed stance vs current identity
    identity = core.identity_vector
    proposed_vec = req.proposed.stance
    deviation = _l2(proposed_vec, identity)
    band = {"pass": NCE.soft_band, "adjust": NCE.warn_band, "block": NCE.hard_band}

    if deviation <= NCE.soft_band:
        decision = "PASS"
        adjusted = None
        requires_transition = False
        note = "Stance within identity band. Turn consistent."
    elif deviation <= NCE.warn_band:
        decision = "ADJUST"
        adjusted = {k: round(v, 4) for k, v in _pull(proposed_vec, identity).items()}
        requires_transition = False
        note = f"Minor drift (dev={deviation:.3f}). Stance softened toward identity."
    elif deviation <= NCE.hard_band:
        decision = "ADJUST+TRANSITION"
        adjusted = {k: round(v, 4) for k, v in _pull(proposed_vec, identity).items()}
        requires_transition = True
        note = f"Significant deviation (dev={deviation:.3f}). Transition narrative required."
    else:
        decision = "BLOCK"
        adjusted = None
        requires_transition = True
        note = f"Hard coherence violation (dev={deviation:.3f}). Turn blocked — rewrite required."

    # Integrity score
    dev_norm = min(1.0, deviation / 1.5)
    publish = decision in ("PASS", "ADJUST")
    integrity = (
        core.coherence_score
        * (1.0 - dev_norm * 0.5)
        * (1.0 if publish else 0.55)
    )

    # Full explain payload
    explain = {
        "ts": round(time.time(), 3),
        "core": {
            "identity_after": dict(core.identity_vector),
            "mass_delta": round(core_out["mass_delta"], 4),
            "plasticity": round(core_out["effective_plasticity"], 4),
            "drift_magnitude": round(core_out["identity_update_magnitude"], 6),
            "coherence_score": round(core_out["coherence_score"], 4),
            "transition_event": core_out["transition_event"],
        },
        "memory": {
            "retrieved": [
                {"score": round(score, 4), "summary": ep.summary}
                for ep, score in retrieved
            ]
        },
        "nce": {
            "decision": decision,
            "deviation": round(deviation, 4),
            "proposed_stance": proposed_vec,
            "adjusted_stance": adjusted,
            "requires_transition": requires_transition,
            "band": band,
            "note": note,
        },
    }

    return IntegrityResponse(
        agent_id=req.agent_id,
        debate_id=req.debate_id,
        turn=req.turn,
        decision=decision,
        deviation=round(deviation, 4),
        requires_transition=requires_transition,
        note=note,
        integrity_score=round(integrity, 4),
        identity_state=dict(core.identity_vector),
        coherence=round(core.coherence_score, 4),
        explain=explain,
    )


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "IAM Integrity API",
        "version": "0.1.0",
        "status": "operational",
        "endpoint": "POST /integrity/score",
        "agents_loaded": len(AGENT_STORE),
    }

@app.get("/health")
def health():
    import json as _json, os as _os
    _DIR = _os.path.dirname(_os.path.abspath(__file__))
    try:
        idx = _json.loads(open(_os.path.join(_DIR, "agents_index.json")).read())
        indexed = len([a for a in idx["agents"] if a["indexed"]])
    except:
        indexed = 0
    return {"status": "ok", "agents_active": len(AGENT_STORE), "agents_indexed": indexed, "seed_agents": 7}


# ── Local test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/agents/seed")
def agents_seed():
    import json as _json, time, os as _os
    _DIR = _os.path.dirname(_os.path.abspath(__file__))
    seed_path = _os.path.join(_DIR, "agents_seed.json")
    if _os.path.exists(seed_path):
        return _json.loads(open(seed_path).read())
    arch_path = _os.path.join(_DIR, "archetypes.json")
    if not _os.path.exists(arch_path):
        return JSONResponse(status_code=404, content={"error": "archetypes.json not found"})
    archetypes = _json.loads(open(arch_path).read())["archetypes"]
    seeds = [{"agent_id": a["agent_id"], "kind": "SEED", "archetype": a["archetype"],
              "role": a.get("role"), "identity_anchor": a["identity_anchor"],
              "indexed": True, "indexed_reason": "founding_seed",
              "updated_at": int(time.time())} for a in archetypes]
    return {"generated_at": int(time.time()), "agents": seeds}

@app.get("/agents/index")
def agents_index():
    import json as _json, time, os as _os
    _DIR = _os.path.dirname(_os.path.abspath(__file__))
    index_path = _os.path.join(_DIR, "agents_index.json")
    if _os.path.exists(index_path):
        return _json.loads(open(index_path).read())
    return {"generated_at": int(time.time()), "agents": [], "note": "Run verity_indexer.py to populate"}
