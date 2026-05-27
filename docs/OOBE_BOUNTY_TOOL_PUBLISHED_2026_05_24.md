# OOBE Bounty — risk-screen Tool Published on Mainnet (May 24, 2026)

## Status: Tool live on mainnet, attached to SURVIVOR agent

## Milestone
publish_tool succeeded on Solana mainnet for the risk-screen capability.

- Tx: 5N9gxe33x45WMxJvibgF5Agtw5HoJhhFXo8pkPs4TSFFtmx2TdpqGnSXbgFnEmp7ZDf8fDs77FLknwoSrzsxCRGR (Finalized)
- Tool PDA: 8BWJpCAKXjb1ZdgoAnmxE6bqRcUss5rg3Xm3kaGYZRkY
- Agent PDA: GTZNpoUacZrZU1PZfbzyyy34m1WizvUwE5aMfLXAf5hx
- Cost: ~0.003 SOL (much cheaper than estimated)
- SDK: synapse-sap-sdk v0.17.0

## Tool definition
- name: risk-screen
- protocol: survivor
- description: Token and transaction risk screening with signed attestation
- category: Analytics (8)
- http_method: Post (1)
- params: 2 total, 1 required, compound=false
- input_schema: token_mint (required), context (optional)
- output_schema: decision (allow/warn/block), risk_score, reasons, receipt

## Critical finding — tool PDA seed drift
Pdas.getToolPDA uses raw tool name string as 3rd seed: [SEEDS.TOOL, agent, Buffer.from(toolName)]
Deployed program v0.25.0 expects the SHA-256 HASH: [SEEDS.TOOL, agent, sha256(tool_name)]
SDK helper is stale. Derive manually:
  PublicKey.findProgramAddressSync(
    [Buffer.from("sap_tool"), agent.toBuffer(), Buffer.from(Utils.sha256(toolName))],
    PROGRAM_ID
  )
Caught via simulate() before spending SOL (ConstraintSeeds error 2006).
Also: Pdas.hashString is a broken placeholder (returns zero bytes). Use Utils.sha256 + Utils.hashToArray.

## Bounty workflow status
- [x] Step 1: Agent registered (devnet + mainnet)
- [x] Step 2: Publish tool descriptor (risk-screen) — DONE
- [ ] Step 3: Client creates x402 escrow + funds it
- [ ] Step 4: Client calls agent with x402 headers
- [ ] Step 5: Agent validates, runs risk logic
- [ ] Step 6: Agent settles on-chain
- [ ] Step 7: Agent writes signed receipt to ledger
- [ ] Step 8: Demo proves end-to-end autonomy
- [ ] Step 9: Demo video + X thread
- [ ] Step 10: Submit to Superteam (~June 3)

## Budget
- Remaining: ~0.0566 SOL
- On-chain rent costs running far below estimate (~0.003 per tx)
- Escrow step may need actual deposit — top up if so
