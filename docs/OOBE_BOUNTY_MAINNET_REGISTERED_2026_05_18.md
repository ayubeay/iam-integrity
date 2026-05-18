# OOBE Bounty — Mainnet Agent Registered (May 18, 2026)

## Status: Agent live on Solana mainnet, bounty-eligible

## Milestone
SURVIVOR Execution Agent registered on Solana mainnet via SAP.

### Mainnet
- Tx: 5bd4s4J5ThujZ3TFfjTJ26SjApzetyrFD1KGKbyX3VMi8DNsPrCAc6g7qiAXYcpnzvNmjyTvswDhWnC5EVnGPeSe
- Agent PDA: GTZNpoUacZrZU1PZfbzyyy34m1WizvUwE5aMfLXAf5hx
- Wallet: 4aet1MhW5gbf46dqzrQB1qxGjM3Q3hN7ndKPRrntW5vg
- Cost: ~0.04 SOL
- Program: SAPpUhsWLJG1FfkGRcXagEDMrMsWGjbky7AyhGpFETZ (v0.25.0)
- SDK: synapse-sap-sdk v0.17.0
- Account shape: 5 accounts (wallet, agent, agent_stats, global_registry, system_program)

### Devnet (earlier May 18)
- Tx: 5EXKQwayd1KKeF4Dyyx7mrWpSTQPZVhi9Sz6E8G8jQ3KW3qikPYL8KNPx62zLxC6BNW4iYNbgFm7q3Y76mRb2MZt
- Agent PDA: GTZNpoUacZrZU1PZfbzyyy34m1WizvUwE5aMfLXAf5hx
- Program: older devnet version (pre-v0.25.0)
- SDK: synapse-sap-sdk v0.16.1
- Account shape: 6 accounts (wallet, agent, agent_stats, pricing_menu, global_registry, system_program)

## Permanent metadata committed
- Name: SURVIVOR Execution Agent
- Description: SAP-native execution admissibility agent. Signed risk screening, escrow evaluation, proof adjudication.
- agentId: did:sap:survivor-execution-agent
- agentUri: https://survivor-oracle-production.up.railway.app
- x402Endpoint: https://survivor-oracle-production.up.railway.app/x402
- Capabilities: survivor:risk-screen, survivor:escrow-evaluate, survivor:proof-adjudicate, solana:token-analysis
- Protocols: survivor, solana, sap

## Critical findings — IDL drift between mainnet vs devnet
The SAP program on mainnet was upgraded to v0.25.0 (removed pricing_menu); devnet runs older version (still requires pricing_menu).
SDK v0.17.0 source code (instructions/agent.ts) correctly omits pricing_menu, but the bundled IDL file still includes it.
For mainnet, the IDL must be patched locally to remove pricing_menu from register_agent.

Two IDLs preserved in ~/oobe-survivor-agent:
- synapse_agent_sap.devnet-6acct.backup.json
- synapse_agent_sap.mainnet-5acct.json

## Resources spent (across May 13-18)
- ~0.07 SOL devnet (free, faucet)
- ~0.04 SOL mainnet (real, ~$4)
- $5 Telegram Stars (DM to OOBE dev for IDL bug confirmation)
- ~15 hours total debugging across 5-6 sessions

## Bounty workflow status
- [x] Step 1: Agent registered on devnet + mainnet
- [ ] Step 2: Publish tool descriptor (risk-screen)
- [ ] Step 3: Client creates x402 escrow + funds it
- [ ] Step 4: Client calls agent with x402 headers
- [ ] Step 5: Agent validates, runs risk logic
- [ ] Step 6: Agent settles on-chain
- [ ] Step 7: Agent writes signed receipt to ledger
- [ ] Step 8: Demo proves end-to-end autonomy
- [ ] Step 9: Demo video + X thread tagging @OOBEonSol and @AceDataCloud
- [ ] Step 10: Submit to Superteam Earn (deadline ~June 3)

## Budget for remaining workflow
- Available: 0.06 SOL (~$6)
- Estimated need: ~0.05 SOL
- Buffer: razor-thin — plan to top up if any retries needed

## Recipe: Mainnet registration reproducibility
For anyone hitting the same IDL/SDK drift issue:
1. Clone OOBE-PROTOCOL/synapse-sap-sdk
2. Checkout v0.17.0 (`git checkout v0.17.0`)
3. Build cleanly (`rm -rf dist node_modules; npm install; npm run build`)
4. Copy IDL: `cp src/idl/synapse_agent_sap.json <your-project>/`
5. Patch IDL: remove `pricing_menu` from `register_agent` accounts
6. Use direct Anchor invocation against the patched IDL with 5 accounts:
   wallet, agent, agent_stats, global_registry, system_program
7. Simulate first (`.simulate()` instead of `.rpc()`)
8. Submit real tx only after simulation succeeds
