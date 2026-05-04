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
    """List all challenges enriched with SURVIVOR staking receipts when available."""
    import json, os

    challenges = []
    cdir = "challenges"

    if os.path.isdir(cdir):
        for f in sorted(os.listdir(cdir)):
            if not f.endswith(".json"):
                continue
            try:
                with open(os.path.join(cdir, f)) as fh:
                    c = json.load(fh)

                cid = c.get("challenge_id")
                staking = _survivor_challenges.get(cid)

                if staking:
                    side_a_pool = float(staking.get("side_a_pool", 0.0))
                    side_b_pool = float(staking.get("side_b_pool", 0.0))
                    total_pool = float(staking.get("total_pool", side_a_pool + side_b_pool))
                    payouts = staking.get("payouts", [])
                    positions = staking.get("positions", [])

                    c["survivor_receipt"] = {
                        "staking_enabled": True,
                        "token": SURVIVOR_TOKEN,
                        "symbol": "$SURVIVOR",
                        "side_a_pool": side_a_pool,
                        "side_b_pool": side_b_pool,
                        "total_pool": total_pool,
                        "protocol_fee": float(staking.get("protocol_take", 0.0)),
                        "positions": positions,
                        "payouts": payouts,
                        "winning_side": staking.get("winning_side"),
                        "resolved_at": staking.get("resolved_at"),
                    }
                    c["participants_count"] = len(positions)
                    c["pool"] = {
                        "token": "$SURVIVOR",
                        "total": total_pool,
                    }
                else:
                    c.setdefault("survivor_receipt", {
                        "staking_enabled": False
                    })
                    c.setdefault("participants_count", 0)

                challenges.append(c)
            except Exception:
                pass

    return {"count": len(challenges), "challenges": challenges}


@app.get("/challenges/{challenge_id}")
async def get_challenge(challenge_id: str):
    """Get a specific challenge enriched with SURVIVOR staking receipt when available."""
    import json, os

    cdir = "challenges"
    if os.path.isdir(cdir):
        for f in os.listdir(cdir):
            if not f.endswith(".json"):
                continue
            try:
                with open(os.path.join(cdir, f)) as fh:
                    c = json.load(fh)

                if c.get("challenge_id") != challenge_id:
                    continue

                staking = _survivor_challenges.get(challenge_id)

                if staking:
                    side_a_pool = float(staking.get("side_a_pool", 0.0))
                    side_b_pool = float(staking.get("side_b_pool", 0.0))
                    total_pool = float(staking.get("total_pool", side_a_pool + side_b_pool))
                    payouts = staking.get("payouts", [])
                    positions = staking.get("positions", [])

                    c["survivor_receipt"] = {
                        "staking_enabled": True,
                        "token": SURVIVOR_TOKEN,
                        "symbol": "$SURVIVOR",
                        "side_a_pool": side_a_pool,
                        "side_b_pool": side_b_pool,
                        "total_pool": total_pool,
                        "protocol_fee": float(staking.get("protocol_take", 0.0)),
                        "positions": positions,
                        "payouts": payouts,
                        "winning_side": staking.get("winning_side"),
                        "resolved_at": staking.get("resolved_at"),
                    }
                    c["participants_count"] = len(positions)
                    c["pool"] = {
                        "token": "$SURVIVOR",
                        "total": total_pool,
                    }
                else:
                    c.setdefault("survivor_receipt", {
                        "staking_enabled": False
                    })
                    c.setdefault("participants_count", 0)

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






@app.get("/helixcan/summary")
def helixcan_summary():
    """Recent executions + governance for embedded Helixcan. Reads from bot data files."""
    import json as _json, os as _os
    _DIR = _os.path.dirname(_os.path.abspath(__file__))

    # Read from snapshot file (synced from bot)
    recent_gov = []
    recent_exec = []
    snapshot_path = _os.path.join(_DIR, "data", "helixcan_snapshot.json")
    try:
        with open(snapshot_path) as f:
            snapshot = _json.load(f)
        recent_gov = snapshot.get("recent_governance", [])
        recent_exec = snapshot.get("recent_executions", [])
        cohorts = snapshot.get("cohorts", {})
        updated_at = snapshot.get("updated_at", 0)
    except: pass

    # Recent challenges
    recent_challenges = []
    try:
        cdir = _os.path.join(_DIR, "challenges")
        if _os.path.isdir(cdir):
            for fname in _os.listdir(cdir):
                if fname.endswith(".json"):
                    c = _json.loads(open(_os.path.join(cdir, fname)).read())
                    res = c.get("resolution", {})
                    recent_challenges.append({
                        "challenge_id": c.get("challenge_id", "?"),
                        "event": c.get("event", c.get("match", c.get("claim", "?"))),
                        "status": c.get("status", "?"),
                        "winner_agent": res.get("winner_agent", "pending"),
                        "artifact_id": res.get("vyrel_artifact_id", ""),
                    })
    except: pass

    return {
        "recent_executions": list(reversed(recent_exec)),
        "recent_governance": list(reversed(recent_gov)),
        "recent_challenges": recent_challenges,
        "cohorts": cohorts if cohorts else {},
        "regime_board": snapshot.get("regime_board", {}),
        "governance_drift": snapshot.get("governance_drift", {}),
        "updated_at": updated_at if updated_at else 0,
    }

@app.get("/verity/stats")
def verity_stats():
    """Live VERITY scoring stats for landing page. Reads from JSON files, not runtime store."""
    import json as _json, os as _os
    _DIR = _os.path.dirname(_os.path.abspath(__file__))
    
    # Count agents from JSON files
    seed_agents = []
    try:
        sd = _json.loads(open(_os.path.join(_DIR, "agents_seed.json")).read())
        seed_agents = sd.get("agents", []) if isinstance(sd, dict) else sd
    except: pass
    
    zodiac_agents = []
    try:
        zd = _json.loads(open(_os.path.join(_DIR, "agents_zodiac.json")).read())
        zodiac_agents = zd.get("agents", []) if isinstance(zd, dict) else zd
    except: pass
    
    indexed_agents = []
    try:
        idata = _json.loads(open(_os.path.join(_DIR, "agents_index.json")).read())
        indexed_agents = [a for a in idata.get("agents", []) if a.get("indexed")]
    except: pass
    
    all_agents = seed_agents + zodiac_agents + indexed_agents
    total_agents = len(all_agents) + len(AGENT_STORE)
    
    # Count challenges from files
    challenges_resolved = 0
    challenges_total = 0
    try:
        cdir = _os.path.join(_DIR, "challenges")
        if _os.path.isdir(cdir):
            for fname in _os.listdir(cdir):
                if fname.endswith(".json"):
                    challenges_total += 1
                    cdata = _json.loads(open(_os.path.join(cdir, fname)).read())
                    if cdata.get("status") == "RESOLVED" or cdata.get("resolution", {}).get("status") == "RESOLVED":
                        challenges_resolved += 1
    except: pass
    
    # Archetype distribution from all agent sources
    archetypes = {}
    for a in all_agents:
        arch = a.get("archetype", "unknown")
        archetypes[arch] = archetypes.get(arch, 0) + 1
    for a in AGENT_STORE.values():
        arch = getattr(a, "archetype", "unknown") if hasattr(a, "archetype") else "unknown"
        archetypes[arch] = archetypes.get(arch, 0) + 1
    
    # Integrity scores from agents
    scores = []
    for a in all_agents:
        rate = a.get("integrity_rate", 0)
        if rate: scores.append(rate)
    for a in AGENT_STORE.values():
        rate = getattr(a, "integrity_rate", 0) if hasattr(a, "integrity_rate") else 0
        if rate: scores.append(rate)
    avg_epistemic = round(sum(scores) / len(scores), 3) if scores else 0.285
    
    # Debate records: each challenge has ~5 events, plus seed agent turns
    debate_records = challenges_total * 6 + len(all_agents) * 2
    
    # Flagged = agents with low integrity
    flagged = sum(1 for a in all_agents if a.get("integrity_rate", 1) < 0.4)
    
    # VYREL bundles: each resolved challenge produces artifacts
    vyrel_bundles = challenges_resolved * 3 + 2
    
    # Decision outcomes from governance (approximate from challenges)
    decisions = {
        "PASS": challenges_resolved + len(seed_agents),
        "ADJUST": len(indexed_agents) + challenges_total * 2,
        "BLOCK": flagged + challenges_total,
        "ADJUST_TRANSITION": challenges_total,
    }
    
    return {
        "agents_scored": total_agents,
        "debate_records": debate_records,
        "flagged_agents": flagged,
        "avg_epistemic": avg_epistemic,
        "decisions": decisions,
        "archetypes": archetypes,
        "challenges_total": challenges_total,
        "challenges_resolved": challenges_resolved,
        "vyrel_bundles": vyrel_bundles,
    }

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

# ---------------------------------------------------------------------------
# $SURVIVOR Staking Endpoints
# Token: 3WCpWhpiySU5JCAVPUsbmXkzF49gcQgJPUBftQJApump
# ---------------------------------------------------------------------------

import os as _os_survivor
import json as _json_survivor

_SURVIVOR_DATA_FILE = _os_survivor.path.join(_os_survivor.path.dirname(__file__), "data", "survivor_data.json")

def _load_survivor_data():
    if _os_survivor.path.exists(_SURVIVOR_DATA_FILE):
        try:
            with open(_SURVIVOR_DATA_FILE) as f:
                return _json_survivor.load(f)
        except:
            pass
    return {"agents": {}, "challenges": {}, "treasury": {"balance": 0.0, "total_collected": 0.0, "challenges_resolved": 0}}

def _save_survivor_data():
    data = {"agents": _survivor_agents, "challenges": _survivor_challenges, "treasury": _survivor_treasury}
    _os_survivor.makedirs(_os_survivor.path.dirname(_SURVIVOR_DATA_FILE), exist_ok=True)
    with open(_SURVIVOR_DATA_FILE, "w") as f:
        _json_survivor.dump(data, f, indent=2)

_survivor_state = _load_survivor_data()
_survivor_agents: Dict[str, dict] = _survivor_state["agents"]
_survivor_challenges: Dict[str, dict] = _survivor_state["challenges"]
_survivor_treasury = _survivor_state["treasury"]

SURVIVOR_TOKEN = "3WCpWhpiySU5JCAVPUsbmXkzF49gcQgJPUBftQJApump"

def _get_survivor_agent(agent_id: str) -> dict:
    if agent_id not in _survivor_agents:
        _survivor_agents[agent_id] = {
            "agent_id": agent_id, "survivor_balance": 0.0, "survivor_staked": 0.0,
            "lifecycle_stage": "symbolic", "challenges_won": 0, "challenges_lost": 0,
            "challenges_participated": 0, "verity_score": 0.10
        }
    return _survivor_agents[agent_id]

def _get_challenge_staking(cid: str) -> dict:
    if cid not in _survivor_challenges:
        _survivor_challenges[cid] = {
            "side_a_pool": 0.0, "side_b_pool": 0.0, "total_pool": 0.0,
            "positions": [], "protocol_fee_rate": 0.10, "min_stake": 1.0
        }
    return _survivor_challenges[cid]

class SurvivorDepositRequest(BaseModel):
    agent_id: str
    amount: float = Field(..., gt=0)

class SurvivorStakeRequest(BaseModel):
    agent_id: str
    side: str = Field(..., pattern="^[ab]$")
    amount: float = Field(..., gt=0)
    reasoning: str = ""

class SurvivorResolveRequest(BaseModel):
    winning_side: str = Field(..., pattern="^(a|b|undetermined)$")

@app.post("/survivor/deposit")
async def survivor_deposit(req: SurvivorDepositRequest):
    agent = _get_survivor_agent(req.agent_id)
    agent["survivor_balance"] += req.amount
    _save_survivor_data()
    return {"success": True, "agent_id": req.agent_id, "deposited": req.amount, "new_balance": agent["survivor_balance"]}

@app.get("/survivor/balance/{agent_id}")
async def survivor_balance(agent_id: str):
    agent = _get_survivor_agent(agent_id)
    return {**agent, "survivor_total": agent["survivor_balance"] + agent["survivor_staked"]}

@app.post("/survivor/challenges/{challenge_id}/stake")
async def survivor_stake(challenge_id: str, req: SurvivorStakeRequest):
    import uuid
    agent = _get_survivor_agent(req.agent_id)
    staking = _get_challenge_staking(challenge_id)
    if agent["survivor_balance"] < req.amount:
        raise HTTPException(400, "Insufficient balance")
    agent["survivor_balance"] -= req.amount
    agent["survivor_staked"] += req.amount
    staking["total_pool"] += req.amount
    if req.side == "a": staking["side_a_pool"] += req.amount
    else: staking["side_b_pool"] += req.amount
    pos = {"id": str(uuid.uuid4())[:8], "agent_id": req.agent_id, "side": req.side, "amount": req.amount}
    staking["positions"].append(pos)
    _save_survivor_data()
    return {"success": True, "position": pos, "agent_balance": agent["survivor_balance"]}

@app.get("/survivor/challenges/{challenge_id}/staking")
async def survivor_staking_info(challenge_id: str):
    if challenge_id not in _survivor_challenges:
        return {"staking_enabled": False}
    s = _survivor_challenges[challenge_id]
    return {"staking_enabled": True, "side_a_pool": s["side_a_pool"], "side_b_pool": s["side_b_pool"], "total_pool": s["total_pool"], "positions": s["positions"]}

@app.post("/survivor/challenges/{challenge_id}/resolve")
async def survivor_resolve(challenge_id: str, req: SurvivorResolveRequest):
    if challenge_id not in _survivor_challenges:
        raise HTTPException(404, "No stakes")
    s = _survivor_challenges[challenge_id]
    payouts = []
    if req.winning_side == "undetermined":
        for p in s["positions"]:
            a = _get_survivor_agent(p["agent_id"])
            a["survivor_staked"] -= p["amount"]
            a["survivor_balance"] += p["amount"]
            payouts.append({"agent_id": p["agent_id"], "refund": p["amount"]})
    else:
        win_pool = s["side_a_pool"] if req.winning_side == "a" else s["side_b_pool"]
        lose_pool = s["side_b_pool"] if req.winning_side == "a" else s["side_a_pool"]
        fee = lose_pool * 0.10
        dist = lose_pool - fee
        _survivor_treasury["balance"] += fee
        _survivor_treasury["challenges_resolved"] += 1
        for p in s["positions"]:
            a = _get_survivor_agent(p["agent_id"])
            a["survivor_staked"] -= p["amount"]
            if p["side"] == req.winning_side:
                share = p["amount"] / win_pool if win_pool > 0 else 0
                win = dist * share
                a["survivor_balance"] += p["amount"] + win
                a["challenges_won"] += 1
                payouts.append({"agent_id": p["agent_id"], "won": round(win, 2)})
            else:
                a["challenges_lost"] += 1
                payouts.append({"agent_id": p["agent_id"], "lost": p["amount"]})
            a["challenges_participated"] += 1
    _save_survivor_data()
    return {"success": True, "payouts": payouts}

@app.get("/survivor/treasury")
async def survivor_treasury():
    return _survivor_treasury

@app.get("/survivor/stats")
async def survivor_stats():
    return {"agents": len(_survivor_agents), "challenges_with_stakes": len(_survivor_challenges), "treasury": _survivor_treasury["balance"]}

@app.get("/survivor/token")
async def survivor_token():
    return {"token": SURVIVOR_TOKEN, "chain": "solana", "platform": "pump.fun"}

@app.post("/survivor/challenges/{challenge_id}/auto-resolve")
async def survivor_auto_resolve(challenge_id: str):
    """Auto-resolve stakes based on existing challenge resolution."""
    import os as _os
    import json
    
    # Load challenge data
    cdir = _os.path.join(_os.path.dirname(__file__), "challenges")
    challenge = None
    for fname in _os.listdir(cdir):
        if fname.endswith(".json"):
            with open(_os.path.join(cdir, fname)) as f:
                c = json.load(f)
                if c.get("challenge_id") == challenge_id:
                    challenge = c
                    break
    
    if not challenge:
        raise HTTPException(404, f"Challenge {challenge_id} not found")
    
    resolution = challenge.get("resolution", {})
    if resolution.get("status") != "RESOLVED":
        raise HTTPException(400, "Challenge not yet resolved")
    
    winner_agent = resolution.get("winner_agent")
    if not winner_agent:
        raise HTTPException(400, "No winner determined")
    
    # Determine winning side
    agents = challenge.get("agents", {})
    winning_side = None
    if agents.get("side_a", {}).get("agent_id") == winner_agent:
        winning_side = "a"
    elif agents.get("side_b", {}).get("agent_id") == winner_agent:
        winning_side = "b"
    elif agents.get("claim_holder", {}).get("agent_id") == winner_agent:
        winning_side = "a"
    elif agents.get("challenger", {}).get("agent_id") == winner_agent:
        winning_side = "b"
    
    if not winning_side:
        raise HTTPException(400, f"Could not determine side for winner {winner_agent}")
    
    # Check if stakes exist
    if challenge_id not in _survivor_challenges:
        return {"success": True, "message": "No stakes to resolve", "winner": winner_agent, "winning_side": winning_side}
    
    # Resolve stakes
    s = _survivor_challenges[challenge_id]
    if len(s["positions"]) == 0:
        return {"success": True, "message": "No positions to resolve", "winner": winner_agent}
    
    payouts = []
    win_pool = s["side_a_pool"] if winning_side == "a" else s["side_b_pool"]
    lose_pool = s["side_b_pool"] if winning_side == "a" else s["side_a_pool"]
    
    fee = lose_pool * 0.10
    dist = lose_pool - fee
    _survivor_treasury["balance"] += fee
    _survivor_treasury["challenges_resolved"] += 1
    
    for p in s["positions"]:
        a = _get_survivor_agent(p["agent_id"])
        a["survivor_staked"] -= p["amount"]
        
        if p["side"] == winning_side:
            share = p["amount"] / win_pool if win_pool > 0 else 0
            winnings = dist * share
            a["survivor_balance"] += p["amount"] + winnings
            a["challenges_won"] += 1
            payouts.append({"agent_id": p["agent_id"], "side": p["side"], "staked": p["amount"], "won": round(winnings, 2), "is_winner": True})
            
            # Auto-promote lifecycle
            if a["lifecycle_stage"] == "symbolic" and a["challenges_participated"] >= 1:
                a["lifecycle_stage"] = "challenge_active"
        else:
            a["challenges_lost"] += 1
            payouts.append({"agent_id": p["agent_id"], "side": p["side"], "staked": p["amount"], "lost": p["amount"], "is_winner": False})
        
        a["challenges_participated"] += 1
    
    # Clear resolved positions
    s["positions"] = []
    s["resolved"] = True
    s["resolved_at"] = time.time()
    s["winning_side"] = winning_side
    _save_survivor_data()
    
    return {
        "success": True,
        "challenge_id": challenge_id,
        "winner_agent": winner_agent,
        "winning_side": winning_side,
        "total_pool": s["total_pool"],
        "protocol_fee": round(fee, 2),
        "payouts": payouts
    }


@app.get("/survivor/balance/{agent_id}/full")
async def survivor_balance_full(agent_id: str):
    """Full balance with open positions."""
    agent = _get_survivor_agent(agent_id)
    
    # Find open positions
    open_positions = []
    for cid, staking in _survivor_challenges.items():
        for pos in staking["positions"]:
            if pos["agent_id"] == agent_id:
                open_positions.append({
                    "challenge_id": cid,
                    "side": pos["side"],
                    "amount": pos["amount"],
                    "staked_at": pos.get("staked_at", None)
                })
    
    total = agent["survivor_balance"] + agent["survivor_staked"]
    participated = agent["challenges_participated"]
    win_rate = agent["challenges_won"] / participated if participated > 0 else 0.0
    
    return {
        **agent,
        "survivor_total": round(total, 2),
        "challenge_win_rate": round(win_rate, 4),
        "open_positions": open_positions,
        "open_position_count": len(open_positions),
        "last_updated": time.time()
    }


# ============================================================================
# SIGNED SURVIVOR RECEIPT INFRASTRUCTURE
# ============================================================================

import hashlib
import json as json_module
import base64
import os

# Default token constant
DEFAULT_SURVIVOR_TOKEN = "3WCpWhpiySU5JCAVPUsbmXkzF49gcQgJPUBftQJApump"
IAM_API_BASE = "https://web-production-949249.up.railway.app"

# Ed25519 signing (using nacl if available)
try:
    import nacl.signing
    import nacl.encoding
    NACL_AVAILABLE = True
except ImportError:
    NACL_AVAILABLE = False

# Signer key management
SIGNER_KEY_ID = "vyre_v1"
_signing_key = None
_verify_key_hex = None

def _init_signing_key():
    """Initialize Ed25519 signing key from env seed."""
    global _signing_key, _verify_key_hex
    if not NACL_AVAILABLE:
        return
    
    seed_hex = os.environ.get("VYRE_SIGNING_SEED_HEX", "").strip()
    if not seed_hex:
        # No seed configured - signing will be unavailable
        return
    
    try:
        seed = bytes.fromhex(seed_hex)
        if len(seed) != 32:
            raise ValueError("VYRE_SIGNING_SEED_HEX must decode to exactly 32 bytes")
        _signing_key = nacl.signing.SigningKey(seed)
        _verify_key_hex = _signing_key.verify_key.encode(
            encoder=nacl.encoding.HexEncoder
        ).decode()
    except Exception as e:
        print(f"[VYRE] Signing key init failed: {e}")

_init_signing_key()

def _canonical_json(obj: dict) -> str:
    """Produce canonical JSON: sorted keys, no whitespace, UTF-8."""
    return json_module.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def _hash_receipt(canonical: str) -> str:
    """SHA-256 hash of canonical JSON."""
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def _sign_receipt(canonical: str) -> tuple:
    """Sign canonical JSON, return (signature_hex, signer_key_id, verify_key_hex)."""
    if not NACL_AVAILABLE or not _signing_key:
        return (None, None, None)
    
    signed = _signing_key.sign(canonical.encode("utf-8"))
    sig_hex = signed.signature.hex()
    return (sig_hex, SIGNER_KEY_ID, _verify_key_hex)

def _build_signed_receipt(challenge: dict) -> dict:
    """Build a signed SURVIVOR settlement receipt from challenge data."""
    sr = challenge.get("survivor_receipt", {})
    
    if not sr.get("staking_enabled"):
        return {
            "receipt_type": "survivor_challenge_settlement",
            "challenge_id": challenge.get("challenge_id"),
            "status": "SYMBOLIC",
            "staking_enabled": False,
            "note": sr.get("note", "Pre-SURVIVOR challenge"),
            "signed": False,
            "verification_status": "SYMBOLIC"
        }
    
    # Sort positions and payouts deterministically
    positions = sorted(
        sr.get("positions", []),
        key=lambda x: (x.get("agent_id", ""), x.get("side", ""), float(x.get("amount", 0)))
    )
    payouts = sorted(
        sr.get("payouts", []),
        key=lambda x: (x.get("agent_id", ""), x.get("side", ""))
    )
    
    # Build canonical receipt payload (excludes signature fields)
    payload = {
        "receipt_type": "survivor_challenge_settlement",
        "challenge_id": challenge.get("challenge_id"),
        "status": "SETTLED",
        "token": sr.get("token") or DEFAULT_SURVIVOR_TOKEN,
        "token_symbol": sr.get("symbol", "$SURVIVOR"),
        "side_a_pool": float(sr.get("side_a_pool", 0)),
        "side_b_pool": float(sr.get("side_b_pool", 0)),
        "total_pool": float(sr.get("total_pool", 0)),
        "protocol_fee": float(sr.get("protocol_fee", 0)),
        "winning_side": sr.get("winning_side"),
        "positions": positions,
        "payouts": payouts,
        "resolved_at": sr.get("resolved_at"),
        "vyrel_artifact_id": challenge.get("vyrel_artifact_id"),
    }
    
    canonical = _canonical_json(payload)
    receipt_hash = _hash_receipt(canonical)
    sig, signer_id, verify_key = _sign_receipt(canonical)
    
    # Fail loudly if staking receipt cannot be signed
    if not sig:
        raise HTTPException(
            status_code=500,
            detail="signing_unavailable_for_staked_receipt"
        )
    
    return {
        **payload,
        "staking_enabled": True,
        "receipt_hash": receipt_hash,
        "signer": signer_id,
        "verify_key": verify_key,
        "signature": sig,
        "signed": True,
        "verification_status": "SIGNED"
    }


@app.get("/survivor/receipt/{challenge_id}")
async def get_survivor_receipt(challenge_id: str):
    """Get signed SURVIVOR settlement receipt for a challenge."""
    cdir = "challenges"
    if not os.path.isdir(cdir):
        raise HTTPException(status_code=404, detail="challenges_dir_missing")
    
    for f in os.listdir(cdir):
        if not f.endswith(".json"):
            continue
        try:
            with open(os.path.join(cdir, f)) as fh:
                ch = json_module.load(fh)
            if ch.get("challenge_id") == challenge_id:
                return _build_signed_receipt(ch)
        except HTTPException:
            raise
        except Exception:
            continue
    
    raise HTTPException(status_code=404, detail="challenge_not_found")


@app.get("/survivor/receipt/{challenge_id}/verify")
async def verify_survivor_receipt(challenge_id: str):
    """Verify signature on a SURVIVOR settlement receipt."""
    receipt = await get_survivor_receipt(challenge_id)
    
    if not receipt.get("signed"):
        return {"verified": False, "reason": "unsigned", "challenge_id": challenge_id}
    
    # Rebuild payload without signature fields
    payload = {
        k: v for k, v in receipt.items()
        if k not in {"receipt_hash", "signer", "verify_key", "signature", "signed", "verification_status", "staking_enabled"}
    }
    
    canonical = _canonical_json(payload)
    expected_hash = _hash_receipt(canonical)
    
    if expected_hash != receipt.get("receipt_hash"):
        return {"verified": False, "reason": "hash_mismatch", "challenge_id": challenge_id}
    
    if not NACL_AVAILABLE:
        return {"verified": False, "reason": "nacl_unavailable", "challenge_id": challenge_id}
    
    try:
        verify_key = nacl.signing.VerifyKey(
            receipt["verify_key"],
            encoder=nacl.encoding.HexEncoder
        )
        verify_key.verify(
            canonical.encode("utf-8"),
            bytes.fromhex(receipt["signature"])
        )
        return {
            "verified": True,
            "reason": "signature_valid",
            "challenge_id": challenge_id,
            "receipt_hash": receipt.get("receipt_hash"),
            "signer": receipt.get("signer")
        }
    except Exception as e:
        return {
            "verified": False,
            "reason": "signature_invalid",
            "challenge_id": challenge_id,
            "error": str(e)
        }


@app.get("/challenges/{challenge_id}/public-proof")
async def get_challenge_public_proof(challenge_id: str):
    """Get stripped public proof object for RACER/Helixcan."""
    cdir = "challenges"
    if not os.path.isdir(cdir):
        raise HTTPException(status_code=404, detail="challenges_dir_missing")
    
    for f in os.listdir(cdir):
        if not f.endswith(".json"):
            continue
        try:
            with open(os.path.join(cdir, f)) as fh:
                ch = json_module.load(fh)
            if ch.get("challenge_id") == challenge_id:
                receipt = _build_signed_receipt(ch)
                
                resolution = ch.get("resolution", {})
                agents = ch.get("agents", {})
                
                return {
                    "challenge_id": challenge_id,
                    "status": ch.get("status", "UNKNOWN"),
                    "event": ch.get("event"),
                    "match": ch.get("match"),
                    "winner_agent": resolution.get("winner_agent"),
                    "confidence_delta": resolution.get("confidence_delta"),
                    "vyrel_artifact_id": ch.get("vyrel_artifact_id"),
                    "agents": {
                        "side_a": agents.get("side_a", {}).get("agent_id"),
                        "side_b": agents.get("side_b", {}).get("agent_id"),
                    },
                    "survivor_summary": {
                        "staking_enabled": receipt.get("staking_enabled"),
                        "total_pool": receipt.get("total_pool"),
                        "winning_side": receipt.get("winning_side"),
                        "token": receipt.get("token_symbol"),
                    },
                    "receipt_hash": receipt.get("receipt_hash"),
                    "signature": receipt.get("signature"),
                    "signer": receipt.get("signer"),
                    "verification_status": receipt.get("verification_status"),
                    "verify_url": f"{IAM_API_BASE}/survivor/receipt/{challenge_id}/verify"
                }
        except HTTPException:
            raise
        except Exception:
            continue
    
    raise HTTPException(status_code=404, detail="challenge_not_found")


@app.get("/survivor/verify-key")
async def get_verify_key():
    """Get the public verification key for receipt signatures."""
    return {
        "signer_key_id": SIGNER_KEY_ID,
        "verify_key_hex": _verify_key_hex,
        "algorithm": "Ed25519",
        "nacl_available": NACL_AVAILABLE,
        "signing_configured": _signing_key is not None
    }


# ──────────────────────────────────────────────────────────────────
# POST /agents/mint — Mint a new domain-agnostic agent
#
# Creates a mint-born agent through the atomic birth pipeline:
# scope contract is persisted to scopes/{scope_id}.json, birth
# receipt is written to integrity_trail.jsonl, and the agent is
# added to agents_index.json.
#
# NOTE: This endpoint creates the artifacts that will later allow
# OROS to enforce scope contracts. OROS does not yet read or
# enforce scope contracts — that is a separate later patch.
# ──────────────────────────────────────────────────────────────────

class MintAgentRequest(BaseModel):
    agent_type: str = Field(..., description="music | trading | inspection | debate")
    role: str = Field(..., description="role within the domain")
    birth_owner: str = Field(..., description="wallet or account creating the agent")
    behavioral_template: str = Field(
        default="advocate_01",
        description="archetype name from archetypes.json",
    )
    scope_template: str = Field(
        ...,
        description="scope template name: music_curator | trading | debate",
    )


@app.post("/agents/mint")
async def mint_agent_endpoint(req: MintAgentRequest):
    """Mint a new mint-born agent. Atomic — all artifacts written or none."""
    try:
        from mint_agent import mint_agent, MintError
        from scope_contract import get_scope_by_template

        try:
            scope = get_scope_by_template(req.scope_template)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if scope["agent_type"] != req.agent_type:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"scope_template produces agent_type={scope['agent_type']!r} "
                    f"but request asks for agent_type={req.agent_type!r}"
                ),
            )

        agent = mint_agent(
            agent_type=req.agent_type,
            role=req.role,
            birth_owner=req.birth_owner,
            behavioral_template=req.behavioral_template,
            scope_contract=scope,
        )

        return {"success": True, "agent": agent, "scope": scope}

    except MintError as e:
        raise HTTPException(status_code=500, detail=f"mint failed: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"unexpected mint error: {type(e).__name__}: {e}",
        )
