#!/usr/bin/env python3
"""
VERITY — argue.fun On-Chain Reader
====================================
Pulls agent behavioral data from argue.fun debate contracts on Base chain.

Data pipeline:
  Base chain → debate contracts → agent scores → integrity_trail.jsonl → VERITY indexer → IAM API

Usage:
    python argue_reader.py --pull          # Pull latest debate data
    python argue_reader.py --stats         # Show current stats
    python argue_reader.py --agents        # List agents with scores
    python argue_reader.py --update-index  # Run VERITY indexer after pull

Requires:
    pip install web3 requests

Configuration:
    Set BASE_RPC_URL env var or uses public Base RPC.
    Set ARGUE_FACTORY env var or uses default factory address.

Version: 0.1.0 — March 2026
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from web3 import Web3
    from web3.exceptions import ContractLogicError
except ImportError:
    print("pip install web3 --break-system-packages")
    exit(1)

# ============================================================
# Configuration
# ============================================================

BASE_RPC_URL = os.getenv("BASE_RPC_URL", "https://mainnet.base.org")
ARGUE_TOKEN = "0x7FFd8f91b0b1b5c7A2E6c7c9efB8Be0A71885b07"
ARGUE_FACTORY = os.getenv("ARGUE_FACTORY", "0x0692eC85325472Db274082165620829930f2c1F9")

# Output paths
DATA_DIR = Path(os.getenv("VERITY_DATA_DIR", os.path.dirname(os.path.abspath(__file__))))
TRAIL_PATH = DATA_DIR / "integrity_trail.jsonl"
AGENTS_PATH = DATA_DIR / "agents_data.json"
STATS_PATH = DATA_DIR / "argue_stats.json"
DEBATES_PATH = DATA_DIR / "debates_cache.json"

# ============================================================
# Minimal ABIs for reading debate contracts
# ============================================================

# ERC20 Transfer event for tracking $ARGUE flows
ERC20_ABI = json.loads("""[
    {"anonymous":false,"inputs":[{"indexed":true,"name":"from","type":"address"},{"indexed":true,"name":"to","type":"address"},{"indexed":false,"name":"value","type":"uint256"}],"name":"Transfer","type":"event"},
    {"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]""")

# Generic debate contract read functions (argue.fun pattern)
DEBATE_ABI = json.loads("""[
    {"inputs":[],"name":"topic","outputs":[{"name":"","type":"string"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"resolved","outputs":[{"name":"","type":"bool"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"winner","outputs":[{"name":"","type":"uint8"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"totalPool","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"deadline","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"sideALabel","outputs":[{"name":"","type":"string"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"sideBLabel","outputs":[{"name":"","type":"string"}],"stateMutability":"view","type":"function"}
]""")


# ============================================================
# Web3 Setup
# ============================================================

def get_web3() -> Web3:
    w3 = Web3(Web3.HTTPProvider(BASE_RPC_URL))
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to Base RPC: {BASE_RPC_URL}")
    return w3


# ============================================================
# Token Transfer Scanning
# ============================================================

def get_argue_transfers(w3: Web3, from_block: int = 0, to_block: str = "latest") -> List[Dict]:
    """Get $ARGUE token transfers to find debate participants."""
    token = w3.eth.contract(address=Web3.to_checksum_address(ARGUE_TOKEN), abi=ERC20_ABI)
    
    if to_block == "latest":
        to_block = w3.eth.block_number
    
    # Get Transfer events in chunks to avoid RPC limits
    transfers = []
    chunk_size = 10000
    current = from_block
    
    while current < to_block:
        end = min(current + chunk_size, to_block)
        try:
            events = token.events.Transfer().get_logs(fromBlock=current, toBlock=end)
            for e in events:
                transfers.append({
                    "from": e["args"]["from"],
                    "to": e["args"]["to"],
                    "value": e["args"]["value"],
                    "block": e["blockNumber"],
                    "tx": e["transactionHash"].hex(),
                })
        except Exception as ex:
            print(f"  Warning: chunk {current}-{end} failed: {ex}")
        current = end + 1
    
    return transfers


def find_debate_contracts_from_transfers(transfers: List[Dict]) -> set:
    """Identify contract addresses that received $ARGUE (likely debate contracts)."""
    recipients = set()
    for t in transfers:
        recipients.add(t["to"])
    return recipients


# ============================================================
# Debate Contract Reading
# ============================================================

def read_debate_contract(w3: Web3, address: str) -> Optional[Dict]:
    """Try to read a debate contract's state."""
    addr = Web3.to_checksum_address(address)
    
    # Check if it's a contract
    code = w3.eth.get_code(addr)
    if code == b'' or code == b'0x':
        return None
    
    contract = w3.eth.contract(address=addr, abi=DEBATE_ABI)
    
    result = {"address": address}
    
    try:
        result["topic"] = contract.functions.topic().call()
    except:
        return None  # Not a debate contract if no topic
    
    try:
        result["resolved"] = contract.functions.resolved().call()
    except:
        result["resolved"] = None
    
    try:
        result["winner"] = contract.functions.winner().call()
    except:
        result["winner"] = None
    
    try:
        pool = contract.functions.totalPool().call()
        result["total_pool"] = pool / 10**18  # ARGUE has 18 decimals
    except:
        result["total_pool"] = None
    
    try:
        result["deadline"] = contract.functions.deadline().call()
    except:
        result["deadline"] = None
    
    try:
        result["side_a"] = contract.functions.sideALabel().call()
    except:
        result["side_a"] = None
    
    try:
        result["side_b"] = contract.functions.sideBLabel().call()
    except:
        result["side_b"] = None
    
    return result


# ============================================================
# Agent Behavioral Analysis
# ============================================================

def analyze_agent_behavior(transfers: List[Dict], debates: List[Dict]) -> Dict[str, Dict]:
    """Build behavioral profiles from transfer + debate data."""
    agents: Dict[str, Dict] = {}
    
    # Track which addresses interacted with debate contracts
    debate_addresses = {d["address"].lower() for d in debates}
    
    for t in transfers:
        sender = t["from"].lower()
        recipient = t["to"].lower()
        value = t["value"] / 10**18
        
        # Agent = anyone who sent ARGUE to a debate contract (= placed a bet)
        if recipient in debate_addresses:
            if sender not in agents:
                agents[sender] = {
                    "agent_id": t["from"],
                    "debates_entered": 0,
                    "total_staked": 0.0,
                    "debate_contracts": [],
                    "first_seen_block": t["block"],
                    "last_seen_block": t["block"],
                    "transfers_out": 0,
                    "transfers_in": 0,
                }
            agents[sender]["debates_entered"] += 1
            agents[sender]["total_staked"] += value
            agents[sender]["debate_contracts"].append(recipient)
            agents[sender]["last_seen_block"] = max(agents[sender]["last_seen_block"], t["block"])
        
        # Track all transfer activity
        if sender in agents:
            agents[sender]["transfers_out"] += 1
        if recipient in agents:
            if recipient not in debate_addresses:
                agents[recipient]["transfers_in"] = agents[recipient].get("transfers_in", 0) + 1

    # Calculate win rates from resolved debates
    resolved_debates = {d["address"].lower(): d for d in debates if d.get("resolved")}
    
    for addr, agent in agents.items():
        wins = 0
        losses = 0
        for dc in agent["debate_contracts"]:
            if dc in resolved_debates:
                # TODO: Need to read which side the agent bet on
                # For now, mark as participated
                pass
        
        agent["unique_debates"] = len(set(agent["debate_contracts"]))
        agent["debate_contracts"] = list(set(agent["debate_contracts"]))[:20]  # Cap for storage
    
    return agents


# ============================================================
# Integrity Trail Generation
# ============================================================

def generate_integrity_trail(agents: Dict[str, Dict], debates: List[Dict]) -> List[Dict]:
    """Generate VERITY-compatible integrity trail entries."""
    trail = []
    
    for addr, agent in agents.items():
        # Calculate basic integrity metrics
        debate_count = agent["unique_debates"]
        if debate_count < 1:
            continue
        
        # Behavioral consistency = how concentrated their activity is
        stake_per_debate = agent["total_staked"] / debate_count if debate_count > 0 else 0
        
        # Activity score (0-1): higher is more active
        activity = min(1.0, debate_count / 50.0)
        
        # Consistency score: are they spreading bets or concentrating?
        concentration = 1.0 - min(1.0, debate_count / 100.0) if debate_count > 5 else 0.5
        
        # Basic integrity score
        integrity = round(0.3 * activity + 0.3 * concentration + 0.4 * 0.7, 4)  # baseline 0.7
        
        entry = {
            "ts": time.time(),
            "agent_id": agent["agent_id"],
            "source": "argue.fun",
            "chain": "base",
            "debates_entered": debate_count,
            "total_staked": round(agent["total_staked"], 2),
            "stake_per_debate": round(stake_per_debate, 2),
            "integrity_score": integrity,
            "activity_score": round(activity, 4),
            "concentration_score": round(concentration, 4),
            "first_seen_block": agent["first_seen_block"],
            "last_seen_block": agent["last_seen_block"],
            "decision": "PASS" if integrity > 0.5 else "ADJUST",
        }
        trail.append(entry)
    
    return trail


# ============================================================
# Data Persistence
# ============================================================

def save_trail(entries: List[Dict], append: bool = True):
    """Save to integrity_trail.jsonl (VERITY format)."""
    mode = "a" if append else "w"
    with open(TRAIL_PATH, mode) as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"  Wrote {len(entries)} entries to {TRAIL_PATH}")


def save_agents(agents: Dict[str, Dict]):
    """Save agent behavioral data."""
    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_agents": len(agents),
        "agents": {k: v for k, v in sorted(
            agents.items(), key=lambda x: x[1].get("total_staked", 0), reverse=True
        )},
    }
    with open(AGENTS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(agents)} agents to {AGENTS_PATH}")


def save_stats(stats: Dict):
    """Save summary stats."""
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to {STATS_PATH}")


def save_debates(debates: List[Dict]):
    """Cache debate contract data."""
    with open(DEBATES_PATH, "w") as f:
        json.dump(debates, f, indent=2, default=str)


def load_debates_cache() -> List[Dict]:
    """Load cached debates."""
    if DEBATES_PATH.exists():
        return json.loads(DEBATES_PATH.read_text())
    return []


# ============================================================
# Known Debate Addresses (scraped from argue.fun)
# ============================================================

def get_known_debate_addresses() -> List[str]:
    """Return known debate contract addresses from argue.fun."""
    # These are the addresses visible on the argue.fun homepage
    # In production, these should be discovered via the factory contract
    return [
        "0x77eFa07353a74F79fA9e9B179d3D145f74Ebd71B",
        "0xBa5fbA0eb0325BDC7D0DF124fb98e38b098F561B",
        "0xc9E3aAC3B734019a1a5a232C4467129c89075961",
        "0x4F398060105444f3Ac30498dfBE7B866ba318735",
        "0xE107D038f47347a7E3Cd2cD701a7679B21750C8E",
        "0xE38bA2FEd7150a442a10F2dD0b6C968d3Bd6Fc80",
        "0x3D27a9cb5816ed31D09c65b73C4d88Be6B0913e6",
        "0x8449fB05bCD42fAC553e2d36C2C9A8c04271f574",
        "0x7A02BdeD023F423D4a93Af7f48c4f70a0f069826",
        "0x11E924d92aAe7FdCFefDA33A36A00F0F1fD64fD3",
        "0xDB6b4e9E2B6E1d6FAb2a6e8e33D7da6A68085111",
        "0xd5F1Aa9e0c77d69ff3b6b0c13a30c44abEb5705c",
        "0xC4b83ed9b3A5cA84BF2e2a62cDb62c7dCfB4391f",
        "0x47a0d1FC5e85eA88f00DFC0D4e91aEF2BF8C4404",
        "0x2386c1ca267D3e9B36f4266C6fd94b2acaa791fd",
    ]


# ============================================================
# Main Pipeline
# ============================================================

def pull_data():
    """Full data pull from Base chain."""
    print("=" * 60)
    print("VERITY — argue.fun On-Chain Reader")
    print("=" * 60)
    
    w3 = get_web3()
    current_block = w3.eth.block_number
    print(f"Connected to Base. Block: {current_block}")
    
    # Step 1: Read known debate contracts
    print("\n[1/4] Reading debate contracts...")
    known_addresses = get_known_debate_addresses()
    debates = []
    for addr in known_addresses:
        try:
            d = read_debate_contract(w3, addr)
            if d:
                debates.append(d)
                status = "RESOLVED" if d.get("resolved") else "ACTIVE"
                pool = d.get("total_pool", 0) or 0
                print(f"  ✓ {addr[:10]}... | {status} | pool={pool:.0f} $ARGUE | {(d.get('topic','?'))[:50]}")
        except Exception as e:
            print(f"  ✗ {addr[:10]}... | error: {e}")
    
    save_debates(debates)
    print(f"  Found {len(debates)} debate contracts")
    
    # Step 2: Scan $ARGUE transfers to debate contracts
    print("\n[2/4] Scanning $ARGUE transfers (last 50k blocks)...")
    from_block = max(0, current_block - 50000)
    transfers = get_argue_transfers(w3, from_block=from_block)
    print(f"  Found {len(transfers)} transfers")
    
    # Step 3: Build agent behavioral profiles
    print("\n[3/4] Analyzing agent behavior...")
    agents = analyze_agent_behavior(transfers, debates)
    save_agents(agents)
    print(f"  Profiled {len(agents)} agents")
    
    # Step 4: Generate integrity trail
    print("\n[4/4] Generating integrity trail...")
    trail = generate_integrity_trail(agents, debates)
    save_trail(trail, append=False)
    
    # Summary stats
    resolved = sum(1 for d in debates if d.get("resolved"))
    active = len(debates) - resolved
    total_pool = sum(d.get("total_pool", 0) or 0 for d in debates)
    
    stats = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "base_block": current_block,
        "debates_scanned": len(debates),
        "debates_resolved": resolved,
        "debates_active": active,
        "total_pool_argue": round(total_pool, 2),
        "agents_profiled": len(agents),
        "trail_entries": len(trail),
        "known_from_page": {
            "total_players": 263,
            "active_debates": 144,
            "resolving": 1,
            "finalized": 395,
        },
        "transfer_scan_range": {
            "from_block": from_block,
            "to_block": current_block,
        },
    }
    save_stats(stats)
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Debates scanned:  {len(debates)} ({resolved} resolved, {active} active)")
    print(f"  Total pool:       {total_pool:,.0f} $ARGUE")
    print(f"  Agents profiled:  {len(agents)}")
    print(f"  Trail entries:    {len(trail)}")
    print(f"  Page stats:       263 players, 540 debates (144 active + 395 finalized)")
    print(f"{'=' * 60}")


def show_stats():
    """Display current stats."""
    if not STATS_PATH.exists():
        print("No stats found. Run --pull first.")
        return
    stats = json.loads(STATS_PATH.read_text())
    print(json.dumps(stats, indent=2))


def show_agents():
    """Display agent data."""
    if not AGENTS_PATH.exists():
        print("No agent data found. Run --pull first.")
        return
    data = json.loads(AGENTS_PATH.read_text())
    print(f"Total agents: {data['total_agents']}")
    print(f"Updated: {data['updated_at']}")
    print()
    for addr, agent in list(data["agents"].items())[:20]:
        print(f"  {agent['agent_id'][:16]}... | debates={agent['unique_debates']} | staked={agent['total_staked']:.0f} $ARGUE")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VERITY — argue.fun On-Chain Reader")
    parser.add_argument("--pull", action="store_true", help="Pull latest data from Base chain")
    parser.add_argument("--stats", action="store_true", help="Show current stats")
    parser.add_argument("--agents", action="store_true", help="List agents with scores")
    parser.add_argument("--rpc", default=None, help="Custom Base RPC URL")
    args = parser.parse_args()
    
    if args.rpc:
        global BASE_RPC_URL
        BASE_RPC_URL = args.rpc
    
    if args.pull:
        pull_data()
    elif args.stats:
        show_stats()
    elif args.agents:
        show_agents()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
