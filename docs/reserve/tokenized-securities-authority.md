# RESERVE - Tokenized Securities / Multi-Representation Asset Governance

**Status:** Reserved architecture. Do not implement.
**Activation:** when a production-accessible tokenized securities venue can be characterised
from evidence rather than headlines.

## Origin
Characterising Robinhood and Crypto.com on 2026-08-15 established that connector capability
is not agent authority, that mandates must be narrower than venue capability, and that
risk-bearing dimensions default closed. Robinhood exposed the need for durable mandates;
Crypto.com exposed the need for instrument-type and leverage bounds.

Tokenised securities add a third dimension: several instruments can carry substantially the
same economic exposure while conferring different legal, custody, settlement, transfer and
counterparty properties.

## The reserved principle
    ECONOMIC EQUIVALENCE DOES NOT IMPLY AUTHORITY EQUIVALENCE

An agent authorised to trade one representation of an asset is not automatically authorised
to trade another because both reference the same underlying.

    AAPL conventional equity
    AAPL issuer-sponsored tokenised security
    AAPL custodial or beneficial-interest token
    AAPL third-party tokenised representation
    AAPL synthetic exposure
    AAPL perpetual derivative

These must not collapse into one AAPL authority domain.

## Future connector class
TOKENIZED_SECURITIES_CONNECTOR - a class, not a commitment to any chain, broker, exchange,
custodian or tokenisation provider. No venue adapter until a real surface can be observed.

## Mandate dimensions to investigate at activation
underlying_asset, instrument_type, representation_type, issuer, custody_model,
settlement_model, settlement_network, transferability, redemption_model,
trading_hours_regime, leverage, counterparty, jurisdiction.

Every risk-bearing dimension defaults closed where omission would widen authority.
Illustrative only - do not freeze names into a schema before discovery.

## VERITY implication
Evaluate the REPRESENTATION, not the ticker. The question is not "is this AAPL" but: what
does this instrument represent, who stands behind that representation, what rights does
possession confer, and was the agent authorised for THIS representation?

## Receipt implication
A receipt should distinguish underlying asset, representation traded, venue, custody path,
settlement path, network, mandate bounds applied, venue-enforced controls, client-enforced
controls, and the evidence establishing representation identity.

Otherwise materially different instruments launder behind a common ticker.

## Invariant
    same ticker            != same instrument
    same price exposure    != same legal claim
    same venue capability  != same agent authority

Authorisation attaches to the governed representation and execution path, not the symbol.
