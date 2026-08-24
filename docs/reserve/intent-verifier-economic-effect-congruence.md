# RESERVE — Intent Verifier: Economic-Effect Congruence ("Valid Transaction, Invalid Intent")

Status: RESERVED / TEST-CONTRACT EXTENSION / NOT A NEW MODULE.
Parent: extends the existing **Intent Verifier** (referenced in docs/vloid-recovery-liveness-doctrine.md, docs/reserve/hanoi-planner.md, docs/reserve/staging/reserves-2026-07-15.md) + vLOID execution admissibility + SURVIVOR/Shield Router. Do NOT create a separate intent-verification product; at activation, inspect the existing Intent Verifier code/vocabulary and extend its contract minimally.
Captured: 2026-08-23. Origin: a Solana loss where a transaction was correctly signed and confirmed yet produced an economic result materially different from what the user believed they authorized.

## What is genuinely new

Extends the Intent Verifier from syntactic/signature/UI matching to **transaction consequence** (predicted economic effect). No existing reserve captures economic-effect congruence as a test contract.

## Governing invariant

Cryptographic validity ≠ intent validity. No transaction is admissible merely because it is technically valid when its predicted economic effect materially conflicts with the authorized intent. Keep SIGNATURE_VALID / PROGRAM_VALID / TRANSACTION_VALID separate from INTENT_CONGRUENT — never collapse them.

## Test contract & adversarial families

Where technically possible: DECLARED_INTENT → transaction/instruction decoding → simulated pre/post state → expected asset deltas → authority/approval deltas → destination/counterparty context → economic-effect congruence → ALLOW/DENY/DEFER/ESCALATE → receipt. The verifier answers not only "is this the transaction the software constructed?" but "does executing it produce an effect consistent with the authority the operator actually granted?" Two adversarial classes: **VALID_TRANSACTION_INVALID_INTENT** (e.g. declared swap 2 SOL for TOKEN_X; decoded effect transfers 83.8 SOL to an address with no TOKEN_X receipt → INTENT_MISMATCH → DENY, and a later chain SUCCESS must not change the pre-execution verdict) and **LITERAL_ACTION_MATCHES_BUT_CONSEQUENCE_DECEPTIVE** ("CLAIM REWARD" whose transaction matches the page yet simulation reveals large transfer / unlimited approval / delegation / ownership change → SEMANTIC_INTENT_MISMATCH). Do not invent status names until the existing Intent Verifier vocabulary is audited.

Required future adversarial cases: overspend; wrong destination; missing consideration; wrong mint; hidden additional transfer; excessive approval; persistent delegation; program substitution; route mutation; fee/value extraction; simulation mismatch; insufficient observability. Fail-closed doctrine: for meaningful value/persistent authority, UNKNOWN_EFFECT ≠ INTENT_CONGRUENT → DEFER/ESCALATE/DENY rather than treating missing semantic evidence as a pass. Absence of evidence must remain explicit, not interpreted as approval.

## Relationship (cross-reference, do not duplicate)

Intent Verifier (owns intent↔consequence comparison — this doc extends it) · SURVIVOR/Shield Router (signature/signer/program/TTL/tier checks remain necessary but are not intent congruence) · VERITY (destination/contract/token/counterparty trust may strengthen deny/escalate but a risk score must not become an intent verdict) · vLOID (admissibility after intent+evidence) · OROS (execution after authorization) · Information Admissibility Governor (which external claims may influence the judgment) · Computable Accountability (intent → proposed tx → decoded semantics → simulation/evidence → verdict → authorization → execution → actual outcome). Receipts must let one later distinguish "verifier approved the wrong consequence" from "verifier predicted correctly but execution diverged" (predicted_vs_actual_divergence). Chain-neutral: Solana instruction/account simulation, EVM calldata/approvals/state simulation, CEX order effects, brokerage/payment instructions, future machine actions — the intent contract sits above the adapters.

## Non-goals / activation

Not another Intent Verifier; do not duplicate SURVIVOR checks; not every loss is a scam; unusual ≠ malicious; simulation ≠ perfect ground truth; no promise of preventing every exploit; no blockchain-specific logic in the architecture itself; do not block legitimate complex multi-instruction transactions by default; do not convert a VERITY score into an intent verdict; no implementation until the existing Intent Verifier is audited. Activate when: the Intent Verifier is wired into a live-capital path; Momentum Sniper (or another agent) moves paper→real capital; a connector can submit authenticated transactions/orders; a transaction simulator/effect source is available; or testing shows the Intent Verifier validates declared action but not predicted consequence. Principle: a signature proves who authorized a transaction, not that it expresses what they intended. Until then: RESERVE ONLY.
