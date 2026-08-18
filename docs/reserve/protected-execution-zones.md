# RESERVE - Protected Execution Zones / Governed Software Change

**Status:** Reserved. Do not scaffold modules to satisfy this.

## Why it exists
    CAPABILITY IS NOT AUTHORITY

An agent being technically able to modify code, deploy, place a trade, move money, change
infrastructure or actuate a physical system does not mean it has authority to do so.

The finance work demonstrated the operational version: a governance component can exist and
pass its isolated tests while the real path bypasses it. On 2026-08-15 the mandate check was
correct in verifyAuthorization and silently dropped at the firewall - revocation would have
worked in tests and done nothing in production.

    A CONTROL THAT EXISTS BUT IS NOT REACHED IS NOT A CONTROL

## Zone classes, illustrative not closed
    CODE_ZONE      auth, authorization, billing, IAM, production config, migrations
    CAPITAL_ZONE   trades, withdrawals, transfers, leverage, settlement
    IDENTITY_ZONE  credentials, signing keys, permissions, delegation
    DATA_ZONE      destructive deletion, sensitive datasets, exports
    INFRA_ZONE     production compute, networking, DNS, databases, deployment
    PHYSICAL_ZONE  robotics, machinery, infrastructure actuation

## The same chain, a different transport
    human mandate -> agent proposes -> evidence -> policy judgment
    -> execution authorization -> protected-zone firewall -> transport -> receipt

For software, an authorization might bind patch hash, repository and base revision, affected
zones, test evidence, target environment, deployment artifact, mandate id and expiry.

## The governor's question
It should not fundamentally care whether the final operation is git merge, deploy_production,
place_equity_order, transfer_usdc, rotate_signing_key, delete_dataset or open_valve.

    DOES THIS ACTION CROSS AN AUTHORITY BOUNDARY?

If yes: what human authority permits it, what bounds apply, what evidence is required, is
the authority still active, is this exact action what was authorized, has state drifted,
what independent controls exist downstream, and what actually happened?

## Do not generalise yet
The current Connector Declaration -> Mandate -> Policy -> Authorization -> Firewall ->
Transport -> Receipt chain stays an empirical reference implementation. Robinhood and
Crypto.com discovery keeps producing evidence about how execution authority behaves across
real venues. **Promote only after repeated surfaces demonstrate the abstraction holds.**

This prevents architecture-by-analogy.

## Doctrines
    CAPABILITY IS NOT AUTHORITY
    POLICY ALLOW IS A JUDGMENT, NOT AUTHORIZATION
    AUTHORITY MUST SURVIVE THE ENTIRE EXECUTION PATH
    DEFAULT CLOSED AT UNDECLARED RISK-BEARING DIMENSIONS
    TEST THE PATH, NOT ONLY THE COMPONENT
    A CONTROL THAT IS NEVER REACHED IS NOT A CONTROL
    PROTECTED CODE IS ONE INSTANCE OF A PROTECTED EXECUTION ZONE
    AUTOMATION MUST NOT BECOME RESPONSIBILITY LAUNDERING
