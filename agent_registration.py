#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def load_trades(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trades file not found: {path}")
    trades: list[dict[str, Any]] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                trades.append(json.loads(line))
    return trades


def split_trades(trades):
    closed = [t for t in trades if t.get("status") == "closed"]
    open_ = [t for t in trades if t.get("status") == "open"]
    allow_closed = [t for t in closed if t.get("decision") == "ALLOW"]
    throttle_closed = [t for t in closed if t.get("decision") == "THROTTLE"]
    allow_all = [t for t in trades if t.get("decision") == "ALLOW"]
    throttle_all = [t for t in trades if t.get("decision") == "THROTTLE"]
    return {"closed": closed, "open": open_, "allow_closed": allow_closed,
            "throttle_closed": throttle_closed, "allow_all": allow_all, "throttle_all": throttle_all}


def calc_trade_stats(trades):
    if not trades:
        return {"count": 0, "win_rate": 0.0, "avg_final_pnl": 0.0, "median_final_pnl": 0.0,
                "max_gain_pct": 0.0, "max_loss_pct": 0.0, "avg_max_gain_pct": 0.0, "avg_max_loss_pct": 0.0}
    finals = [float(t.get("final_pnl_pct") or 0.0) for t in trades]
    max_gains = [float(t.get("max_gain_pct") or 0.0) for t in trades]
    max_losses = [float(t.get("max_loss_pct") or 0.0) for t in trades]
    wins = sum(1 for x in finals if x > 0)
    return {"count": len(trades), "win_rate": round(wins / len(trades), 4),
            "avg_final_pnl": round(sum(finals) / len(finals), 4),
            "median_final_pnl": round(statistics.median(finals), 4),
            "max_gain_pct": round(max(max_gains), 4), "max_loss_pct": round(min(max_losses), 4),
            "avg_max_gain_pct": round(sum(max_gains) / len(max_gains), 4),
            "avg_max_loss_pct": round(sum(max_losses) / len(max_losses), 4)}


def dominant_regime(trades):
    if not trades: return "unknown"
    c = {}
    for t in trades: r = str(t.get("regime") or "unknown"); c[r] = c.get(r, 0) + 1
    return max(c, key=c.get)


def dominant_decision(trades):
    if not trades: return "unknown"
    c = {}
    for t in trades: d = str(t.get("decision") or "unknown"); c[d] = c.get(d, 0) + 1
    return max(c, key=c.get)


def determine_zodiac(regime, stats):
    vol = stats.get("max_gain_pct", 0) + abs(stats.get("max_loss_pct", 0))
    if regime == "microcap_ignition" or vol >= 20: return ("aries", "\u2648", "fire", "aggressive_initiative")
    if regime == "midcap_continuation": return ("leo", "\u264c", "fire", "confident_continuation")
    if regime == "lowcap_expansion": return ("sagittarius", "\u2650", "fire", "expansion_seeking")
    return ("aries", "\u2648", "fire", "aggressive_initiative")


def determine_archetype(regime, win_rate, median_pnl):
    if regime == "microcap_ignition": return "inner_truth"
    if regime == "midcap_continuation" and win_rate >= 0.45: return "maximizer"
    if regime == "lowcap_expansion" and median_pnl > 0: return "radical"
    return "inner_truth"


def calculate_verity(trades):
    groups = split_trades(trades)
    closed = groups["closed"]; throttle_closed = groups["throttle_closed"]
    if not trades: return 0.05
    if not closed: return 0.10
    stats = calc_trade_stats(closed)
    confidence = min(stats["count"] / 20.0, 1.0)
    pnl_c = max(min((stats["median_final_pnl"] + 10.0) / 20.0, 1.0), 0.0)
    payoff_c = 0.0
    mla = abs(stats["max_loss_pct"])
    if mla > 0: payoff_c = min(stats["max_gain_pct"] / mla / 3.0, 1.0)
    tr = len(throttle_closed) / max(stats["count"], 1)
    raw = 0.35 * stats["win_rate"] + 0.25 * pnl_c + 0.20 * payoff_c + 0.20 * confidence
    raw *= (1.0 - 0.20 * tr)
    return round(max(0.05, min(raw, 0.95)), 4)


def derive_life_stage(total, closed, allow_closed=0, verity=0.0):
    if closed < 3: return "seed"
    if closed < 10: return "adolescent"
    if allow_closed < 3 or verity < 0.30: return "adolescent"
    if closed < 30: return "prime"
    if verity < 0.50: return "prime"
    return "mature"


def determine_status(trades, verity):
    g = split_trades(trades); closed = g["closed"]; ac = g["allow_closed"]
    stats = calc_trade_stats(closed)
    if len(closed) < 3: return "seeded"
    if len(ac) >= 10 and stats["win_rate"] >= 0.40 and stats["median_final_pnl"] > 0 and verity >= 0.35: return "qualified"
    if verity >= 0.25: return "observed"
    return "seeded"


def listing_status(trades, verity):
    g = split_trades(trades); ac = g["allow_closed"]; closed = g["closed"]
    stats = calc_trade_stats(closed)
    if len(ac) >= 10 and verity >= 0.40 and stats["median_final_pnl"] > 0: return "listable"
    if len(ac) >= 3 and verity >= 0.30: return "watchlist"
    return "not_listable"


def is_eligible(trades):
    g = split_trades(trades); closed = g["closed"]; stats = calc_trade_stats(closed)
    if len(closed) >= 3 and stats["max_gain_pct"] >= 5.0:
        return True, f'closed={len(closed)} max_gain={stats["max_gain_pct"]:.1f}%'
    if len(trades) >= 25 and stats["max_gain_pct"] >= 10.0:
        return True, f'total={len(trades)} exploratory birth'
    return False, f'insufficient: closed={len(closed)} max_gain={stats["max_gain_pct"]:.1f}%'


def trade_history_digest(trades):
    return hashlib.sha256(json.dumps(trades, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def register_agent(trades, wallet):
    now = time.time(); g = split_trades(trades)
    cs = calc_trade_stats(g["closed"]); regime = dominant_regime(trades)
    zodiac, sym, elem, bias = determine_zodiac(regime, cs)
    arch = determine_archetype(regime, cs["win_rate"], cs["median_final_pnl"])
    verity = calculate_verity(trades); stage = derive_life_stage(len(trades), len(g["closed"]), len(g["allow_closed"]), verity)
    status = determine_status(trades, verity); digest = trade_history_digest(trades)
    aid = "agent_" + hashlib.sha256(f"{wallet}:{digest}".encode()).hexdigest()[:16]
    return {
        "agent_id": aid, "operator_id": "momentum_sniper_001", "source": "momentum_sniper",
        "created_at": now, "birth_timestamp": now, "status": status, "birth_reason": "paper_trade_history",
        "zodiac": zodiac, "zodiac_symbol": sym, "element": elem, "bias": bias, "archetype": arch,
        "life_stage": stage, "oversight": "high" if verity < 0.25 else ("moderate" if verity < 0.50 else "low"),
        "exec_limit_mult": 0.35 if status == "seeded" else (0.65 if status == "observed" else 1.0),
        "verity_score": verity, "wallet": wallet,
        "total_trades": len(trades), "closed_trades": len(g["closed"]), "open_trades": len(g["open"]),
        "allow_closed_trades": len(g["allow_closed"]), "throttle_closed_trades": len(g["throttle_closed"]),
        "dominant_regime": regime, "dominant_decision": dominant_decision(trades),
        "win_rate": cs["win_rate"], "avg_final_pnl_pct": cs["avg_final_pnl"],
        "median_final_pnl_pct": cs["median_final_pnl"],
        "max_gain_pct": cs["max_gain_pct"], "max_loss_pct": cs["max_loss_pct"],
        "listing_status": listing_status(trades, verity),
        "trade_history_digest": digest,
        "vloid_config": {"survivor_gate": True, "praetor_posture": True, "helix_execution": True},
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/paper_trades.jsonl"
    wallet = "HYsRqHRc8w2pMkFSJQH3X5utY8nef9iqUwccctuP7a97"
    trades = load_trades(path)
    print(f"=== ELIGIBILITY CHECK ({len(trades)} trades) ===")
    eligible, reason = is_eligible(trades)
    print(f"Eligible: {eligible} | Reason: {reason}")
    if not eligible: print("Agent not yet eligible"); sys.exit(0)
    profile = register_agent(trades=trades, wallet=wallet)
    print("\n=== AGENT PROFILE (v2) ===")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
