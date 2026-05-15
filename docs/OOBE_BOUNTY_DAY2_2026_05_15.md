# OOBE Bounty — Day 2 (May 15, 2026)

## Status: Blocked on IDL/SDK mismatch, awaiting OOBE dev reply

## Today's progress
- Devnet wallet funded via Solana faucet (1 SOL)
- Discovered CLI v0.15.0 contains broken pricingMenu placeholder in register flow
- Bypassed CLI via direct Anchor invocation (~/oobe-survivor-agent/register-anchor.ts)
- All diagnostics confirmed clean:
  - PROGRAM_ID matches deployed program (devnet + mainnet, byte-identical)
  - GlobalRegistry PDA exists on devnet, owned by SAP program
  - register_agent IDL discriminator matches on-chain
  - GlobalRegistry discriminator [100,213,140,104,66,152,15,238] matches IDL
  - Derived PDAs match expected addresses
- Registration via Anchor methods.registerAgent() fails with AnchorError 3007 AccountOwnedByWrongProgram (origin: global_registry)
- Cloned synapse-sap-explorer; confirmed their builder.register() uses identical 5-account shape (no pricingMenu)
- DM sent to @ethercode_0xKpt (OOBE dev) via paid Telegram channel ($5 / 25 Stars)

## Blocking issue
SDK src/instructions/agent.ts passes pricingMenu account
IDL register_agent does not include pricingMenu
Direct Anchor call with IDL-only accounts (5 accounts) still fails on global_registry ownership check
Root cause unknown without IDL clarification from OOBE team

## Resources spent
- ~0.07 SOL on devnet (no successful registration)
- $5 Telegram Stars to DM OOBE dev
- ~5 hours of debugging

## Tomorrow's decision tree
- Reply received → resume Day 2 build with fix
- No reply by EOD May 16 → drop OOBE, redirect to LeadScan revival + SURVIVOR outreach + Solana infra cold DMs
