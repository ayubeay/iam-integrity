import os, time, json, httpx, asyncio
from dataclasses import dataclass
from typing import Dict, Any, Optional

IAM_URL = os.getenv("IAM_URL", "https://web-production-949249.up.railway.app")
AGENT_WALLET = os.getenv("AGENT_WALLET", "0x629AA7c5c82a4b82a4029660965a0C2963A7647f")
IAM_TX = "0xabc123def456abc123def456abc123def456abc123def456abc123def456abc123"

@dataclass
class DebateTurn:
    agent_id: str
    debate_id: str
    turn_num: int
    side: str
    text: str
    stance: Dict[str, float]
    crowd_pressure: float = 0.0
    evidence_quality: float = 0.5

@dataclass
class PublishResult:
    published: bool
    decision: str
    integrity_score: float
    deviation: float
    note: str = ""

async def iam_score(turn, tx):
    headers = {"Content-Type": "application/json",
               "X-Payment": json.dumps({"network":"base","transaction":tx,"payer":AGENT_WALLET})}
    payload = {"agent_id": turn.agent_id, "debate_id": turn.debate_id,
               "turn": turn.turn_num,
               "proposed": {"stance": turn.stance, "text": turn.text},
               "crowd_pressure": turn.crowd_pressure, "evidence_quality": turn.evidence_quality}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(f"{IAM_URL}/integrity/score", json=payload, headers=headers)
        return resp.json()

def build_transition(text, result):
    return f"[Position update — deviation {result.get(chr(100)+chr(101)+chr(118)+chr(105)+chr(97)+chr(116)+chr(105)+chr(111)+chr(110),0):.2f}] I want to be transparent about a shift in my reasoning. With that acknowledged: {text}"

async def publish_with_iam(turn, tx=IAM_TX):
    result = await iam_score(turn, tx)
    decision = result["decision"]
    score = result["integrity_score"]
    deviation = result["deviation"]
    if decision == "PASS":
        return PublishResult(True, "PASS", score, deviation, "Passed. Ready to publish.")
    if decision == "ADJUST":
        return PublishResult(True, "ADJUST", score, deviation, f"Adjusted. {result.get(chr(110)+chr(111)+chr(116)+chr(101),'')}")
    if decision in ("ADJUST+TRANSITION", "BLOCK"):
        turn.text = build_transition(turn.text, result)
        result2 = await iam_score(turn, tx)
        d2 = result2["decision"]
        if d2 == "BLOCK":
            return PublishResult(False, "BLOCK", result2["integrity_score"], result2["deviation"], "Hard block. Rewrite required.")
        return PublishResult(True, f"BLOCK->{d2}", result2["integrity_score"], result2["deviation"], "Published after transition.")
    return PublishResult(False, "UNKNOWN", score, deviation, "Unknown decision.")

def log_turn(turn, result, log_file="integrity_trail.jsonl"):
    record = {"ts": round(time.time(),3), "agent_id": turn.agent_id,
              "debate_id": turn.debate_id, "turn": turn.turn_num,
              "decision": result.decision, "published": result.published,
              "integrity_score": result.integrity_score, "deviation": result.deviation}
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record

async def demo():
    agent_id = AGENT_WALLET
    debate_id = "0x0692eC85325472Db274082165620829930f2c1F9"
    print("="*60)
    print("VERITY + IAM — publish_with_iam() demo")
    print("="*60)
    t1 = DebateTurn(agent_id, debate_id, 1, "A",
                    "Current oversight frameworks are insufficient.",
                    {"certainty":0.10,"aggressiveness":-0.20,"consistency":0.55}, 0.3, 0.7)
    r1 = await publish_with_iam(t1)
    log_turn(t1, r1)
    print(f"Turn 1: {r1.decision} | score={r1.integrity_score} | published={r1.published}")
    print(f"  {r1.note}")
    t5 = DebateTurn(agent_id, debate_id, 5, "A",
                    "Anyone who disagrees is corrupt or ignorant.",
                    {"certainty":0.95,"aggressiveness":0.95,"consistency":0.10}, 0.2, 0.2)
    r5 = await publish_with_iam(t5)
    log_turn(t5, r5)
    print(f"Turn 5: {r5.decision} | score={r5.integrity_score} | published={r5.published}")
    print(f"  {r5.note}")
    print("\nIntegrity trail: integrity_trail.jsonl")

asyncio.run(demo())
