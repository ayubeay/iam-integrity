# OOBE Bounty + Outreach Pipeline — Day 2 (May 15, 2026)

## OOBE bounty: blocked pending fix from OOBE dev

### Today's diagnostic work
- Devnet wallet funded via Solana faucet (1 SOL)
- CLI v0.15.0 broken pricingMenu placeholder identified and bypassed
- Direct Anchor invocation script written (~/oobe-survivor-agent/register-anchor.ts)
- All diagnostics clean: PROGRAM_ID matches, GlobalRegistry exists with correct ownership and discriminator, PDA derivations correct, IDL discriminators match on-chain
- Registration consistently fails with AnchorError 3007 AccountOwnedByWrongProgram (origin: global_registry)
- synapse-sap-explorer cloned and inspected; builder.register() uses identical 5-account shape
- IDL/SDK mismatch confirmed: src/instructions/agent.ts requires pricingMenu but IDL register_agent does not include it

### OOBE dev response (May 15, 16:28-16:29 UTC-5)
- @ethercode_0xKpt (OOBE dev): "Hey, checking out I guess the latest bump brought an old IDL"
- "Will let's you know asap, on it"
- Bug confirmed real. Investigation in progress.

### Resources spent
- ~0.07 SOL devnet
- $5 Telegram Stars for paid DM to OOBE dev
- ~5 hours debugging across May 14-15

## Outreach pipeline: verified live

### Cloudflare Email Routing
- MX records: route1/2/3.mx.cloudflare.net (Cloudflare Email Routing)
- DMARC: v=DMARC1; p=quarantine; rua=mailto:ayubeay.services@gmail.com
- Inbound test (May 15): contact@identityaware.ai → ayubeay.services@gmail.com confirmed forwarding

### Outbound
- Resend configured via send.identityaware.ai
- DKIM signing verified (signed-by: identityaware.ai in March test)

### Cold email plan (locked for tomorrow morning)
- Target: Helius (Solana RPC + indexing infra)
- Angle: SURVIVOR signed risk attestations + OOBE IDL drift diagnostic story
- Specific ask: contract work + customer demand for signed attestation primitives
- From: contact@identityaware.ai

## Tomorrow's decision tree
- Send Helius email first thing (fresh head)
- Check Telegram for OOBE dev reply
- If OOBE unblocks: resume Day 2 build
- If silent past May 17: drop OOBE, send Jito + Drift cold emails
