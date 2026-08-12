# RESERVE - Private Agent Trading Infrastructure (PATI)

**Status:** Reserved. Do not activate until explicitly authorized.
**Relationship to MomentumSniper:** SEPARATE. **To RACER:** complementary, not identical.

## Thesis
A platform for privately controlled autonomous trading agents, where capital remains owned
and controlled by the trader. We supply intelligence infrastructure, execution
orchestration, governance, safety controls, connectivity, observability and receipts.

    YOUR CAPITAL. YOUR AGENT. YOUR MANDATE. GOVERNED EXECUTION INFRASTRUCTURE.

Infrastructure for autonomous trading, not a shared fund. No pooled customer capital.

## Critical separation
MomentumSniper stays PRIVATE proprietary technology. Do not expose its source, thresholds,
signals, models, regime logic or alpha; do not turn it into a SaaS product; do not couple
the platform so tightly that the private strategy must be disclosed.

It may serve as an internal proving environment, stress test and reference implementation.
The commercial platform is the governed agent infrastructure, not the strategy.

## Capital control
Prefer non-custodial architecture. Separate:

    INTELLIGENCE AUTHORITY  the system believes this trade should occur
    CAPITAL AUTHORITY       this agent may execute this action against this capital

A strategy decision must not equal execution permission. Where possible: withdrawals
disabled, transfers separately authorized, trade-size and daily-loss ceilings, venue and
asset allowlists, chain restrictions, emergency revocation, customer kill switch, minimal
credential scope.

## Strategy layer
Strategies propose. They do not hold final authority.

    strategy -> proposed action -> vLOID and risk policy
      -> ALLOW execute | THROTTLE reduce | DEFER wait | DENY nothing

Sources may include customer-authored, platform-approved, third-party, licensed, private,
deterministic rule systems, or governed AI strategies.

## RACER relationship
RACER is a public strategy marketplace and evaluation environment. PATI is a private
execution environment. A strategy being visible or highly ranked in RACER must NEVER grant
execution authority.

    DISCOVERY != AUTHORIZATION

## Security
Assume autonomous financial agents become attack targets. Credential isolation, least
privilege, strategy sandboxing, malicious-strategy detection, prompt and data manipulation
resistance, transaction simulation, contract and token verification, abnormal behaviour
detection, exposure ceilings, emergency revocation, tamper-evident receipts, customer-
visible history. No strategy may bypass the execution governor.

## Business model - reserved, not decided
Subscription, infrastructure usage, API usage, execution usage, premium data, additional
chains or venues, advanced governance, private deployments, marketplace fees.

Do NOT assume performance fees. Anything involving performance compensation, discretionary
management, custody, pooled funds, investment recommendations, securities or managed
accounts needs legal and regulatory analysis first. **Non-custodial architecture does not
by itself establish regulatory exemption.**

## Long-term possibility
Personal autonomous capital infrastructure - portfolio, trading, arbitrage, treasury, yield
and commerce agents, each under identity, mandate, capital boundary, governance, execution
and receipts. Long-term only; do not expand scope.

## What is actually being reserved
Not another trading bot. The possibility of turning the vLOID execution stack into the
governed control plane for privately owned autonomous capital agents.

The moat need not be "our bot predicts markets better." The thesis is that autonomous
capital should not execute merely because an AI or a strategy wants it to.
