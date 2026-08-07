# RESERVE - Governed Swap Execution Adapter (0x on Solana)

**Status:** Reserved with a concrete build plan. Weekend-scale, not a product.
**Urgency:** MEDIUM. The demonstration is stronger than the integration.

## The premise
0x launched its Swap API on Solana in open beta - the first non-EVM chain for an
EVM-only-until-now aggregator. SPL, native SOL and Token-2022 across 10+ liquidity sources.

The architecturally important detail: swap-instructions returns a quote, a route plan and
instructions rather than a finished transaction. 0x never submits anything or sets priority
fees. Signing, submission and compute budget stay with the integrator, and transaction bytes
are reserved for the integrator's own instructions.

That makes it unusually compatible with a governance layer.

## The demonstration
Not "we integrated the 0x API." The question worth asking publicly:

    what happens when a DEX aggregator chooses the route,
    but an independent governance system decides whether that route may execute?

    0x -> proposed route -> adapter -> SURVIVOR (token and program risk)
       -> VERITY (evidence) -> vLOID (admissibility) -> DENY | ALLOW
       -> OROS (governed execution) -> Solana -> signed receipt

## Build plan
Day 1 - adapter. token_in, token_out, amount, taker, slippage. Capture zid, expected output,
instructions, ALTs and every leg of route_plan. Normalise into an intent the architecture
understands: provider, chain, tokens, amounts, route, slippage_bps, provider_request_id.

Day 2 - governance. Before transaction construction: token check, program check, route
check, slippage check, recipient check, instruction check, simulation. ALLOW constructs the
versioned transaction; DENY means nothing is signed. That distinction is the product.

Day 3 - receipts and demo. Provider, network, intent, route, each check's result, the vLOID
decision, execution status, transaction signature, governance latency, receipt signature.

## Show a rejection
A demo containing only successful transactions does not demonstrate governance. Deliberately
include a second case where 0x returns a valid route, policy detects a violation, vLOID
denies, and the transaction is never signed.

That is the moment the thesis becomes legible.

## Token-2022 as a test case
0x routes PYUSD and USDG. Those carry the transfer controls SURVIVOR already classifies -
permanent delegate, freeze authority, latent hooks - which makes them far more interesting
policy subjects than ordinary SPL tokens. The transfer-fee execution cost work applies
directly.

## Reserved for later
0x reserves transaction bytes for integrator instructions. That opens composing verification
or receipt instructions INTO the transaction rather than governing around it. Not this
weekend.

Integrator fee controls exist on Solana. Do not monetise before proving governed execution.

## Related but separate: Alpenglow
Anza's Alpenglow bug bounty runs Aug 5-19, up to 50,000 SOL. Findings must not be publicly
disclosed - public disclosure disqualifies them. If pursued at all it is a private security
track, never a build-in-public one. Studying the architecture publicly is fine; publishing
suspected vulnerabilities or PoCs is not.

Quantumglow - Anza's post-quantum Alpenglow - is research relevant to the Zircon
cryptographic-evolution program (c), not a weekend build.

## Grading
The strongest thing here is that it uses infrastructure that already exists and runs, rather
than starting something disconnected. The narrative is honest: an experiment, not a product.
Whether it is worth a weekend depends on whether a public demonstration of governed
execution serves a current goal - and right now the concrete deadline is Tuesday.
