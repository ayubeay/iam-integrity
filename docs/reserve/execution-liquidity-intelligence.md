# RESERVE - Execution Liquidity Intelligence (ELI)

**Status:** Reserved. Not an active build.
**Parent:** Universal Money Router / vLOID execution architecture.
**Urgency:** LOW until multi-venue routing creates real need.

## Thesis
ELI is not an exchange, market maker, liquidity provider, order-book dashboard or a
commitment to operate proprietary liquidity. It is a venue-neutral intelligence layer
answering: where, how, and under what constraints can this transaction actually execute
right now?

    ADVERTISED LIQUIDITY  !=  EXECUTABLE LIQUIDITY  !=  SUCCESSFUL EXECUTION

Optimise for the last.

## The problem
A venue can display substantial liquidity and still execute poorly: insufficient depth at
size, spread, slippage, latency, stale quotes, volatility, partial fills, failures,
outages, rate limits, settlement uncertainty, counterparty risk, fragmentation, regime
change.

"Provider X has liquidity" is not evidence for routing a transaction.

## Position
    intent -> IAM -> VERITY -> vLOID admissibility -> ELI -> money router
    -> OROS -> venue -> settlement -> receipt -> learning loop

ELI informs execution. It does not bypass governance.

## Realized executable liquidity
Liquidity is not binary. It is conditional on size x time x venue x regime x latency x
settlement x risk. Displayed depth of $5M may execute $10k excellently and $2M
unacceptably.

## Execution frontier
For a given transaction, candidate strategies include a single venue, splits across
venues, incremental execution, or deferral. The best path is not the one advertising the
deepest book - it is the best governed expected outcome.

Outputs: ROUTE_SINGLE, ROUTE_SPLIT, THROTTLE, DEFER, ESCROW, SIMULATE, DENY. ELI
recommends; OROS and vLOID decide admissibility.

## Predicted versus realized
Every execution records what was predicted - price, slippage, latency, fees, fill,
settlement - and what actually happened, then computes PREDICTION ERROR.

A provider that repeatedly advertises excellent conditions and delivers poor realized
execution should decline in routing confidence. That is the defence against quote games,
disappearing depth, systematic rejection and latency manipulation.

## Contextual competence, not a universal score
A provider may be excellent for small transactions, poor for large, good in volatile
regimes and degrading in latency. Learn the profile rather than assigning one number.

## VERITY is a different question
    VERITY  how much should we trust this provider?
    ELI     how suitable is this provider for THIS execution?

A trustworthy provider can be the wrong venue. A liquid venue can have unacceptable trust
characteristics.

## Connection to the quota incident
NOMINAL AVAILABILITY != USABLE CAPACITY. A dependency can be technically available while
latency, quota exhaustion or degraded infrastructure make it unsuitable for a
time-sensitive execution. Execution quality is end-to-end, so ELI should eventually
consider infrastructure health alongside financial liquidity.

## Moat
Not access to a provider - competitors integrate the same ones. The compounding asset is
execution history, provider history, regime history, prediction accuracy, routing
knowledge, governance and receipts.

## Scope discipline
Do not build an exchange, become a market maker, purchase liquidity, integrate providers
without demand, build another order-book dashboard, or tokenise anything.

## Activation gate
Only when a real execution system has multiple viable routes, materially different
outcomes, enough volume to generate evidence, measurable routing costs, and a reason static
routing is no longer sufficient.

## Governing principle
Liquidity is not something a provider claims. For execution infrastructure it is something
demonstrated through successful execution.

    quote -> predict -> govern -> route -> execute -> settle -> receipt -> compare
    -> learn -> route better
