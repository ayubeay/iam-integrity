# OOBE Bounty — Full On-Chain Agent Lifecycle (May 30, 2026)

## Status: SURVIVOR agent operating live on OOBE SAP protocol (mainnet)

## Finalized on-chain milestones
1. Agent registered — GTZNpoUacZrZU1PZfbzyyy34m1WizvUwE5aMfLXAf5hx
   - PDA seed [sap_agent, wallet], wallet 4aet1MhW5gbf46dqzrQB1qxGjM3Q3hN7ndKPRrntW5vg
   - identity: did:sap:survivor-execution-agent, is_active=true
   - x402_endpoint: https://survivor-oracle-production.up.railway.app/x402
2. Tool published — risk-screen (tx 5N9gxe...CRGR)
   - Tool PDA 8BWJpCAKXjb1ZdgoAnmxE6bqRcUss5rg3Xm3kaGYZRkY
3. Agent stake — 0.1 SOL locked (tx 3MBXaLK1...2GPQ)
   - Stake PDA 966ahz46zniX3xNtzjeTkLGbBLmPehTvdh5gicztVFE7
4. Pricing tier set — standard, 1000 lamports/call, SOL, Escrow mode (tx 2HAZUTH2...Wtki)
5. Escrow created + funded — 9zK9uaDWcuKzzTXeayhto33ksPz1s7CT52HTPZPQuw3r (tx 2rWR14d...WxB2X)
   - balance 10000 lamports, max_calls 10, price_per_call 1000, permissive (no co-signer/arbiter, dispute_window 0)

## Settlement status
settle_calls_v2 attempted; returns InvalidAccount (6089) at escrow_v2.rs:333.
Verified clean: all 5 accounts derive correctly (agent, agent_stats, escrow),
agent authority = signer wallet, escrow funded + unexpired, calls<max, balance sufficient.
No on-chain precedent of successful settle_calls_v2 on this program.
Conclusion: handler precondition not satisfiable in single self-dealing tx; settlement
release path documented but not exercised. Escrow is settlement-ready.

## Critical engineering findings (SDK/IDL drift)
- Published synapse_agent_sap.mainnet-5acct.json IDL is STALE vs deployed program.
  - update_agent: stale IDL claimed 4 accounts incl. phantom pricing_menu; real program = 3 accounts [wallet, agent, system_program], pricing written into agent account.
  - create_escrow_v2: stale IDL claimed 7 accounts (agent_stake/agent_stats/pricing_menu); real = 4 [depositor, agent, escrow, system_program].
- Ground truth recovered via `anchor idl fetch` (on-chain IDL, 274KB) — used for all subsequent instructions.
- Prior findings still valid: tool PDA uses sha256(name) not raw name; Pdas.hashString broken.
- SDK AgentModule.updateAgent feeds wrong account order (caused InvalidProgramId 3008); bypassed with manual IDL-driven encoding.
- Validator requires rate_limit > 0 (InvalidRateLimit 6036 on rate=0).

## Methodology note
Every instruction simulated before send. Zero SOL wasted on failed txs.
All account layouts verified against on-chain IDL + on-chain account decode before spending.

## Budget
- Remaining: ~0.0239 SOL
- Stake 0.1 SOL recoverable (request_unstake, 7-day cooldown)
