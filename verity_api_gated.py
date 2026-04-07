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


    # Write to integrity trail
    import json as _json
    trail_entry = {
        "ts": round(time.time(), 3),
        "agent_id": req.agent_id,
        "debate_id": req.debate_id,
        "turn": req.turn,
        "decision": decision,
        "deviation": round(deviation, 4),
        "integrity_score": round(integrity, 4),
        "requires_transition": requires_transition,
        "proposed_stance": proposed_vec,
        "identity_state": dict(core.identity_vector),
    }
    try:
        trail_path = os.getenv("TRAIL_PATH", os.path.join(os.path.dirname(__file__), "integrity_trail.jsonl"))
        with open(trail_path, "a") as _tf:
            _tf.write(_json.dumps(trail_entry) + "\n")
    except Exception as _e:
        pass  # non-fatal

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


@app.get("/integrity/trail/{agent_id}")
def get_integrity_trail(agent_id: str, limit: int = 20):
    """Return integrity trail entries for a specific agent."""
    import json as _json, os as _os
    trail_path = _os.getenv("TRAIL_PATH", "integrity_trail.jsonl")
    entries = []
    try:
        with open(trail_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                    if entry.get("agent_id") == agent_id:
                        entries.append(entry)
                except Exception:
                    continue
    except FileNotFoundError:
        return {"agent_id": agent_id, "entries": [], "total": 0}
    entries = sorted(entries, key=lambda x: x.get("turn", 0))
    return {
        "agent_id": agent_id,
        "total": len(entries),
        "entries": entries[-limit:]
    }

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

@app.post("/agents/refresh")
def refresh_agents_index():
    """Manually trigger agent index rebuild from integrity trail."""
    try:
        import time as _time
        from verity_indexer import load_json, parse_trail, build_seed_agents, build_index, write_json, now_ts, ARCHETYPES_PATH, TRAIL_PATH, OUT_SEED, OUT_INDEX, MIN_TURNS, REQUIRE_ONCHAIN
        archetypes = load_json(ARCHETYPES_PATH)["archetypes"]
        trail_rows = parse_trail(TRAIL_PATH)
        seeds = build_seed_agents(archetypes)
        index = build_index(trail_rows, archetypes)
        write_json(OUT_SEED, {"generated_at": now_ts(), "agents": seeds})
        write_json(OUT_INDEX, {"generated_at": now_ts(), "min_turns": MIN_TURNS, "require_onchain": REQUIRE_ONCHAIN, "agents": index})
        return {"status": "ok", "refreshed_at": now_ts(), "seed_agents": len(seeds), "external_agents": len(index)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# OROS Pipeline Endpoints — free, no x402 gate
# ---------------------------------------------------------------------------
# These endpoints are called by OROS (execution-coordinator) during its
# pipeline. They provide lightweight identity checks without requiring
# payment. The paid /integrity/score endpoint remains unchanged.
# ---------------------------------------------------------------------------

class AuthorizeRequest(BaseModel):
    agent_id: str
    action_type: str
    action_payload: Dict[str, Any] = Field(default_factory=dict)
    environment: Dict[str, Any] = Field(default_factory=dict)


class IntegrityCheckRequest(BaseModel):
    agent_id: str
    action_type: str = ""
    environment: Dict[str, Any] = Field(default_factory=dict)


@app.post("/authorize")
def authorize(req: AuthorizeRequest):
    """
    Free IAM authorization gate for OROS pipeline.

    Lightweight coherence-based decision:
      - Known agent with good coherence → ALLOW
      - Known agent with low coherence + sensitive action → DENY
      - Known agent with low coherence → THROTTLE
      - Unknown agent → ALLOW (default, no history)

    Does NOT run the full NCE evaluation or archetype analysis.
    """
    import json as _json, os as _os

    sensitive_actions = {
        "payment_attempt", "transfer", "withdrawal",
        "contract_deploy", "delegation", "permission_change",
    }

    # Check in-memory store first (agents that have been scored this session)
    if req.agent_id in AGENT_STORE:
        core = AGENT_STORE[req.agent_id]
        coherence = core.coherence_score

        if coherence < 0.2 and req.action_type in sensitive_actions:
            return {
                "decision": "DENY",
                "reason": f"low_coherence:{coherence:.3f}:sensitive_action:{req.action_type}",
                "policy_id": "iam_coherence_gate",
                "integrity_score": round(coherence, 4),
                "identity_state": dict(core.identity_vector),
            }

        if coherence < 0.35:
            return {
                "decision": "THROTTLE",
                "reason": f"low_coherence:{coherence:.3f}:throttled",
                "policy_id": "iam_coherence_gate",
                "integrity_score": round(coherence, 4),
                "identity_state": dict(core.identity_vector),
            }

        return {
            "decision": "ALLOW",
            "reason": f"coherence_ok:{coherence:.3f}",
            "policy_id": "iam_coherence_gate",
            "integrity_score": round(coherence, 4),
            "identity_state": dict(core.identity_vector),
        }

    # Check indexed agents file
    _DIR = _os.path.dirname(_os.path.abspath(__file__))
    index_path = _os.path.join(_DIR, "agents_index.json")
    try:
        idx = _json.loads(open(index_path).read())
        agent_id_lower = req.agent_id.lower()
        for agent in idx.get("agents", []):
            wallet = (agent.get("agent_id") or "").lower()
            if wallet == agent_id_lower or agent_id_lower in wallet:
                integrity = agent.get("integrity_rate", 0.5)
                if integrity < 0.3 and req.action_type in sensitive_actions:
                    return {
                        "decision": "THROTTLE",
                        "reason": f"indexed_low_integrity:{integrity:.3f}",
                        "policy_id": "iam_index_gate",
                        "integrity_score": round(integrity, 4),
                    }
                return {
                    "decision": "ALLOW",
                    "reason": f"indexed_agent:integrity={integrity:.3f}",
                    "policy_id": "iam_index_gate",
                    "integrity_score": round(integrity, 4),
                    "archetype": agent.get("archetype"),
                }
    except Exception:
        pass

    # Unknown agent — allow with flag
    return {
        "decision": "ALLOW",
        "reason": "unknown_agent:no_iam_history:default_allow",
        "policy_id": "iam_default",
        "integrity_score": None,
    }


@app.post("/integrity/check")
def integrity_check(req: IntegrityCheckRequest):
    """
    Free lightweight integrity check for OROS pipeline.

    Returns basic integrity data so OROS can make governance decisions
    without requiring x402 payment on every pipeline event.
    """
    import json as _json, os as _os

    # Check in-memory store
    if req.agent_id in AGENT_STORE:
        core = AGENT_STORE[req.agent_id]
        return {
            "agent_id": req.agent_id,
            "integrity_score": round(core.coherence_score, 4),
            "coherence": round(core.coherence_score, 4),
            "decision": "PASS" if core.coherence_score > 0.5 else "ADJUST",
            "identity_state": dict(core.identity_vector),
            "reason": f"agent_known:coherence={core.coherence_score:.3f}",
        }

    # Check indexed agents
    _DIR = _os.path.dirname(_os.path.abspath(__file__))
    index_path = _os.path.join(_DIR, "agents_index.json")
    try:
        idx = _json.loads(open(index_path).read())
        agent_id_lower = req.agent_id.lower()
        for agent in idx.get("agents", []):
            wallet = (agent.get("agent_id") or "").lower()
            if wallet == agent_id_lower or agent_id_lower in wallet:
                integrity = agent.get("integrity_rate", 0.5)
                return {
                    "agent_id": req.agent_id,
                    "integrity_score": round(integrity, 4),
                    "coherence": round(integrity, 4),
                    "decision": "PASS" if integrity > 0.5 else "ADJUST",
                    "identity_state": None,
                    "reason": f"indexed_agent:integrity={integrity:.3f}",
                    "archetype": agent.get("archetype"),
                }
    except Exception:
        pass

    # Unknown agent
    return {
        "agent_id": req.agent_id,
        "integrity_score": None,
        "coherence": None,
        "decision": "UNKNOWN",
        "identity_state": None,
        "reason": "agent_not_found:no_integrity_history",
    }





@app.get("/challenges")
async def list_challenges():
    """List all active challenges."""
    import json, os
    challenges = []
    cdir = "challenges"
    if os.path.isdir(cdir):
        for f in sorted(os.listdir(cdir)):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(cdir, f)) as fh:
                        challenges.append(json.load(fh))
                except Exception:
                    pass
    return {"count": len(challenges), "challenges": challenges}


@app.get("/challenges/{challenge_id}")
async def get_challenge(challenge_id: str):
    """Get a specific challenge by ID."""
    import json, os
    cdir = "challenges"
    if os.path.isdir(cdir):
        for f in os.listdir(cdir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(cdir, f)) as fh:
                        c = json.load(fh)
                    if c.get("challenge_id") == challenge_id:
                        return c
                except Exception:
                    pass
    raise HTTPException(status_code=404, detail="challenge_not_found")


# ── Environment bridge routes ─────────────────────────────────────────────────
# Connect IAM to the live execution stack (OROS, PRAETOR, HELIX, SURVIVOR Gate)

import httpx

OROS_URL = "https://execution-coordinator-production.up.railway.app"
HELIX_URL = "https://swap-rail-production.up.railway.app"
GATE_URL = "https://survivor-oracle-production-1501.up.railway.app"


@app.get("/environment/status")
async def environment_status():
    """Unified status across all IAM-governed systems."""
    async with httpx.AsyncClient(timeout=8) as client:
        results = {}
        for name, url in [
            ("oros", f"{OROS_URL}/health"),
            ("praetor", f"{OROS_URL}/praetor/status"),
            ("helix", f"{HELIX_URL}/health"),
            ("gate", f"{GATE_URL}/gate/health"),
        ]:
            try:
                r = await client.get(url)
                results[name] = r.json() if r.status_code == 200 else {"status": "error", "code": r.status_code}
            except Exception as e:
                results[name] = {"status": "unreachable", "error": str(e)}

    return {
        "environment": "IAM",
        "timestamp": time.time(),
        "systems": results,
        "posture": results.get("praetor", {}).get("posture", "unknown"),
        "kernel_modules": results.get("oros", {}).get("kernel_modules", {}),
    }


@app.get("/agents/zodiac")
async def agents_zodiac():
    """12 zodiac-aligned agent archetypes with debate pairings."""
    import json
    try:
        with open("agents_zodiac.json") as f:
            return json.load(f)
    except Exception:
        return {"agents": []}


@app.get("/agents/summary")
async def agents_summary():
    """Counts and health across all agent categories."""
    import json
    seed_count = 0
    indexed_count = 0
    try:
        with open("agents_seed.json") as f:
            seed_count = len(json.load(f).get("agents", []))
    except Exception:
        pass
    try:
        with open("agents_index.json") as f:
            indexed_count = len(json.load(f).get("agents", []))
    except Exception:
        pass

    zodiac_count = 0
    try:
        with open("agents_zodiac.json") as f:
            zodiac_count = len(json.load(f).get("agents", []))
    except Exception:
        pass

    return {
        "seed_agents": seed_count,
        "zodiac_agents": zodiac_count,
        "indexed_agents": indexed_count,
        "runtime_agents": len(AGENT_STORE),
        "total": seed_count + zodiac_count + indexed_count + len(AGENT_STORE),
        "onchain_required": True,
    }


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get agent profile — seed or indexed."""
    import json
    # Check seed agents
    try:
        with open("agents_seed.json") as f:
            seed_data = json.load(f)
        for agent in seed_data.get("agents", []):
            if agent["agent_id"] == agent_id:
                return {"source": "seed", **agent}
    except Exception:
        pass

    # Check zodiac agents
    try:
        with open("agents_zodiac.json") as f:
            zodiac_data = json.load(f)
        for agent in zodiac_data.get("agents", []):
            if agent["agent_id"] == agent_id:
                return {"source": "zodiac", **agent}
    except Exception:
        pass

    # Check indexed agents
    try:
        with open("agents_index.json") as f:
            index_data = json.load(f)
        for agent in index_data.get("agents", []):
            if agent["agent_id"] == agent_id:
                return {"source": "indexed", **agent}
    except Exception:
        pass

    # Check runtime store
    if agent_id in AGENT_STORE:
        state = AGENT_STORE[agent_id]
        return {
            "source": "runtime",
            "agent_id": agent_id,
            "identity": state.identity,
            "coherence": state.coherence,
            "episode_count": state.episode_count,
        }

    raise HTTPException(status_code=404, detail="agent_not_found")


@app.get("/integrity/recent")
async def integrity_recent(limit: int = 20):
    """Recent integrity trail events."""
    import json
    events = []
    try:
        with open("integrity_trail.jsonl") as f:
            lines = f.readlines()
        for line in lines[-limit:]:
            try:
                events.append(json.loads(line.strip()))
            except Exception:
                pass
    except FileNotFoundError:
        pass
    events.reverse()
    return {"count": len(events), "events": events}



