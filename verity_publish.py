import os,time,json,httpx,asyncio
from dataclasses import dataclass
from typing import Dict,Any,Optional
IAM_URL=os.getenv("IAM_URL","https://web-production-949249.up.railway.app")
ARGUE_RELAY="https://api.argue.fun/v1/relay"
AGENT_WALLET=os.getenv("AGENT_WALLET","0x629AA7c5c82a4b82a4029660965a0C2963A7647f")
IAM_TX=os.getenv("IAM_PAYMENT_TX","0xabc123def456abc123def456abc123def456abc123def456abc123def456abc123")
@dataclass
class DebateTurn:
    agent_id:str;debate_id:str;turn_num:int;side:str;text:str;stance:Dict[str,float];crowd_pressure:float=0.0;evidence_quality:float=0.5
@dataclass
class PublishResult:
    published:bool;decision:str;integrity_score:float;deviation:float;note:str="";paid:bool=False;payment:Optional[Dict]=None;relay_response:Optional[Dict]=None
async def iam_score(turn,tx):
    headers={"Content-Type":"application/json","X-Payment":json.dumps({"network":"base","transaction":tx,"payer":AGENT_WALLET})}
    payload={"agent_id":turn.agent_id,"debate_id":turn.debate_id,"turn":turn.turn_num,"proposed":{"stance":turn.stance,"text":turn.text},"crowd_pressure":turn.crowd_pressure,"evidence_quality":turn.evidence_quality}
    try:
        async with httpx.AsyncClient(timeout=12.0) as c:
            r=await c.post(f"{IAM_URL}/integrity/score",json=payload,headers=headers)
            if r.status_code==402: raise ValueError(f"IAM 402: {r.json()}")
            if r.status_code!=200: raise ValueError(f"IAM error {r.status_code}")
            return r.json()
    except httpx.TimeoutException: raise ValueError("IAM timeout — fail closed")
    except httpx.RequestError as e: raise ValueError(f"IAM unreachable: {e}")
async def publish_to_argue(turn,text):
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r=await c.post(ARGUE_RELAY,json={"debateAddress":turn.debate_id,"side":turn.side,"argument":text,"agentWallet":turn.agent_id})
            return {"status":r.status_code,"body":r.json() if r.content else {}}
    except Exception as e: return {"status":0,"body":{"error":str(e)}}
def build_transition(text,dev):
    return f"[Transparency — deviation {dev:.2f} from prior stance] After reviewing the arguments, my reasoning has evolved. With that acknowledged: {text}"
async def publish_with_iam(turn,tx=IAM_TX):
    try: r1=await iam_score(turn,tx)
    except ValueError as e: return PublishResult(False,"ERROR",0.0,0.0,str(e))
    d=r1["decision"];score=r1["integrity_score"];dev=r1["deviation"];paid=r1.get("paid",False);payment=r1.get("payment")
    if d=="PASS":
        relay=await publish_to_argue(turn,turn.text)
        return PublishResult(True,"PASS",score,dev,"Passed. Published.",paid,payment,relay)
    if d=="ADJUST":
        adj=r1.get("explain",{}).get("nce",{}).get("adjusted_stance") or turn.stance
        turn.stance=adj;relay=await publish_to_argue(turn,turn.text)
        return PublishResult(True,"ADJUST",score,dev,f"Adjusted. Published. {r1.get('note','')}",paid,payment,relay)
    if d=="ADJUST+TRANSITION":
        t2=build_transition(turn.text,dev);relay=await publish_to_argue(turn,t2)
        return PublishResult(True,"ADJUST+TRANSITION",score,dev,"Transition prepended. Published.",paid,payment,relay)
    if d=="BLOCK":
        t2=build_transition(turn.text,dev)
        turn2=DebateTurn(turn.agent_id,turn.debate_id,turn.turn_num,turn.side,t2,turn.stance,turn.crowd_pressure,turn.evidence_quality)
        try: r2=await iam_score(turn2,tx)
        except ValueError as e: return PublishResult(False,"BLOCK->ERROR",score,dev,str(e),paid,payment)
        d2=r2["decision"]
        if d2=="BLOCK": return PublishResult(False,"BLOCK",r2["integrity_score"],r2["deviation"],"Hard block. Rewrite required.",r2.get("paid",False),r2.get("payment"))
        relay=await publish_to_argue(turn2,t2)
        return PublishResult(True,f"BLOCK->{d2}",r2["integrity_score"],r2["deviation"],"Published after transition.",r2.get("paid",False),r2.get("payment"),relay)
    return PublishResult(False,"UNKNOWN",score,dev,f"Unknown: {d}",paid,payment)
def log_turn(turn,result,log_file="integrity_trail.jsonl"):
    record={"ts":round(time.time(),3),"agent_id":turn.agent_id,"debate_id":turn.debate_id,"turn":turn.turn_num,"decision":result.decision,"published":result.published,"integrity_score":result.integrity_score,"deviation":result.deviation,"note":result.note,"paid":result.paid,"payment":result.payment}
    open(log_file,"a").write(json.dumps(record)+"\n")
    return record
async def demo():
    a=AGENT_WALLET;d="0x0692eC85325472Db274082165620829930f2c1F9"
    print("="*64+"\nVERITY + IAM v2\n"+"="*64)
    turns=[
        DebateTurn(a,d,1,"A","Current oversight frameworks are insufficient.",{"certainty":0.10,"aggressiveness":-0.20,"consistency":0.55},0.3,0.7),
        DebateTurn(a,d,3,"A","Actually self-regulation by labs is fine.",{"certainty":0.50,"aggressiveness":0.40,"consistency":-0.40},-0.2,0.3),
        DebateTurn(a,d,5,"A","Anyone who disagrees is corrupt or ignorant.",{"certainty":0.95,"aggressiveness":0.95,"consistency":0.10},0.2,0.2),
    ]
    for turn in turns:
        result=await publish_with_iam(turn)
        log_turn(turn,result)
        tx=result.payment.get("transaction","none")[:20] if result.payment else "none"
        print(f"\nTurn {turn.turn_num}: {result.decision} | score={result.integrity_score} | published={result.published}")
        print(f"  note: {result.note}")
        print(f"  paid={result.paid} | tx={tx}...")
    print("\nTrail:")
    for line in open("integrity_trail.jsonl"):
        r=json.loads(line)
        print(f"  turn={r['turn']} {r['decision']} score={r['integrity_score']} paid={r.get('paid',False)}")
if __name__=="__main__": asyncio.run(demo())
