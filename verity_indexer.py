import os, json, time, math
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

ARCHETYPES_PATH = os.getenv("ARCHETYPES_PATH", "archetypes.json")
TRAIL_PATH = os.getenv("TRAIL_PATH", "integrity_trail.jsonl")
OUT_INDEX = os.getenv("OUT_INDEX", "agents_index.json")
OUT_SEED = os.getenv("OUT_SEED", "agents_seed.json")
MIN_TURNS = int(os.getenv("INDEX_MIN_TURNS", "3"))
REQUIRE_ONCHAIN = os.getenv("INDEX_REQUIRE_ONCHAIN", "1") == "1"
PASSLIKE = {"PASS", "ADJUST", "ADJUST+TRANSITION"}

def now_ts(): return int(time.time())
def load_json(path): return json.loads(open(path).read())
def euclid(a, b):
    keys = ["certainty", "aggressiveness", "consistency"]
    return math.sqrt(sum((float(a.get(k,0.0))-float(b.get(k,0.0)))**2 for k in keys))

def parse_trail(path):
    if not os.path.exists(path): return []
    rows = []
    for line in open(path):
        line = line.strip()
        if not line: continue
        try:
            r = json.loads(line)
            if "paid" not in r: r["paid"] = False
            if "payment" not in r: r["payment"] = None
            rows.append(r)
        except: continue
    return rows

def build_seed_agents(archetypes):
    return [{
        "agent_id": a["agent_id"], "kind": "SEED",
        "archetype": a["archetype"], "role": a.get("role"),
        "identity_anchor": a["identity_anchor"],
        "elasticity": a.get("elasticity"),
        "risk_tolerance": a.get("risk_tolerance"),
        "stake_multiplier": a.get("stake_multiplier"),
        "escalation_sensitivity": a.get("escalation_sensitivity"),
        "iam_profile": a.get("iam_profile", {}),
        "indexed": True, "indexed_reason": "founding_seed",
        "updated_at": now_ts(),
    } for a in archetypes]

def summarize_agent_turns(rows):
    total = len(rows)
    passlike = sum(1 for r in rows if r.get("decision") in PASSLIKE)
    blocks = sum(1 for r in rows if r.get("decision") == "BLOCK")
    transitions = sum(1 for r in rows if r.get("decision") == "ADJUST+TRANSITION")
    avg_int = (sum(float(r.get("integrity_score",0)) for r in rows)/total) if total else 0.0
    avg_dev = (sum(float(r.get("deviation",0)) for r in rows)/total) if total else 0.0
    onchain = any(
        (isinstance(r.get("payment"),dict) and r["payment"].get("transaction")) or
        r.get("txHash") or r.get("relayTx") or r.get("onchain_tx")
        for r in rows
    )
    return {
        "total_turns": total, "passlike_turns": passlike,
        "block_turns": blocks, "transition_turns": transitions,
        "integrity_rate": round(passlike/total,4) if total else 0.0,
        "avg_integrity_score": round(avg_int,4),
        "avg_deviation": round(avg_dev,4),
        "has_onchain_proof": onchain,
        "last_seen_ts": max(int(r.get("ts",0)) for r in rows) if total else 0,
    }

def assign_archetype(identity_state, archetypes):
    if not isinstance(identity_state, dict): identity_state = {}
    best, dist = "Unknown", 10**9
    for a in archetypes:
        d = euclid(identity_state, a["identity_anchor"])
        if d < dist: dist, best = d, a["archetype"]
    return best, round(float(dist),4)

def build_index(trail_rows, archetypes):
    by_agent = {}
    for r in trail_rows:
        aid = r.get("agent_id")
        if aid: by_agent.setdefault(aid,[]).append(r)
    agents = []
    for agent_id, rows in by_agent.items():
        rows = sorted(rows, key=lambda x: float(x.get("ts",0)))
        s = summarize_agent_turns(rows)
        last_identity = rows[-1].get("identity_state") or rows[-1].get("identity") or {}
        arch, arch_dist = assign_archetype(last_identity, archetypes)
        q_turns = s["total_turns"] >= MIN_TURNS
        q_onchain = s["has_onchain_proof"] if REQUIRE_ONCHAIN else True
        qualifies = q_turns and q_onchain
        agents.append({
            "agent_id": agent_id, "kind": "EXTERNAL",
            "indexed": bool(qualifies),
            "indexed_reason": (
                "meets_thresholds" if qualifies else
                f"needs_turns({MIN_TURNS})" if not q_turns else
                "needs_onchain_proof"
            ),
            "archetype": arch, "archetype_distance": arch_dist,
            **s, "updated_at": now_ts(),
        })
    agents.sort(key=lambda a: (0 if a["indexed"] else 1, -float(a["integrity_rate"]), -int(a["last_seen_ts"])))
    return agents

def write_json(path, obj): open(path,"w").write(json.dumps(obj,indent=2)+"\n")

def main():
    archetypes = load_json(ARCHETYPES_PATH)["archetypes"]
    trail_rows = parse_trail(TRAIL_PATH)
    seeds = build_seed_agents(archetypes)
    index = build_index(trail_rows, archetypes)
    write_json(OUT_SEED, {"generated_at": now_ts(), "agents": seeds})
    write_json(OUT_INDEX, {"generated_at": now_ts(), "min_turns": MIN_TURNS, "require_onchain": REQUIRE_ONCHAIN, "agents": index})
    print(f"Seed agents: {len(seeds)}")
    print(f"External agents: {len(index)}")
    for a in index:
        print(f"  {a['agent_id'][:20]}... indexed={a['indexed']} turns={a['total_turns']} integrity={a['integrity_rate']}")

if __name__ == "__main__": main()
