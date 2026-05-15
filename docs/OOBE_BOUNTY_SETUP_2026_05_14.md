# OOBE Bounty — Setup Complete (May 14, 2026)

## Toolchain
- SDK: ~/synapse-sap-sdk (cloned, built, npm-linked)
- CLI: synapse-sap v0.15.0 (globally available)
- Keypair: ~/.config/solana/oobe-agent.json
- Pubkey: 4aet1MhW5gbf46dqzrQB1qxGjM3Q3hN7ndKPRrntW5vg
- Mainnet balance: 0.10002 SOL (funded from rentreclaim wallet)

## Registration plan (locked)
- Name: SURVIVOR Execution Agent
- Capabilities:
  - survivor:risk-screen
  - survivor:escrow-evaluate
  - survivor:proof-adjudicate
  - solana:token-analysis
- Frame: SAP-native execution admissibility agent
- Differentiator vs other submissions: signed execution receipts + auditable settlement memory, NOT raw payment volume

## Devnet RPC verified
synapse-sap --cluster devnet agent list → empty result, no errors
