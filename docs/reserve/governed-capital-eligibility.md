# RESERVE — Governed Capital Eligibility (GCE)

Status: Future architecture reserve. Activate only after VERITY, SURVIVOR,
vLOID, Universal Money Router, and Shield Router mature. No execution
authority granted here.
Canonical home: iam-integrity/docs/reserve/governed-capital-eligibility.md
Captured: 2026-07-25 (consolidated packet)

## Purpose

Replace static reputation-based credit with continuously governed,
execution-based capital allocation. Where traditional lending evaluates the
past, GCE evaluates whether capital should support a **specific execution
now**. Historical performance is evidence, not authorization: capital is
released only when an execution satisfies current governance requirements,
depending on both accumulated trust and present admissibility.

## Non-goals

Not a credit score, not a wallet-history lender, and not a single-metric
underwriting model (no reliance on P&L, wallet age, volume, win rate, or a
reputation number in isolation). It does not decide whether a project is
admissible for funding at all (that is Capital Admissibility) and does not
match founders to evaluators (that is DACI). It governs the release of
capital against an already-admissible execution.

## Relationship to existing stack

GCE is the execution-financing facet of the capital trio:

    Project admissibility (Capital Admissibility Framework)
        -> Capital intelligence (Domain-Aware Capital Intelligence)
        -> Execution financing (Governed Capital Eligibility)

Just as vLOID decides whether an execution is admissible, GCE decides
whether that admissible execution should receive financial support —
separating the right to execute from the right to be financed. Pipeline:
IAM -> VERITY -> SURVIVOR -> DRIFT -> vLOID -> Shield Router -> Governed
Capital Eligibility -> Universal Money Router -> OROS -> Execution ->
Receipts.

## Activation condition

Do not build until the named upstream layers (VERITY, SURVIVOR, vLOID,
Universal Money Router, Shield Router) exist and there is a real capital-
backed execution stream to govern. Reserve is not build.

## Eligibility factors

Computed from multiple independent signals, no single one dominating:
identity (verified continuity, account integrity — IAM); trust (execution
quality, repayment history, policy compliance, receipts — VERITY);
execution (intent consistency, admissibility, safety, authorization —
vLOID); environment (regime, volatility, liquidity, counterparty quality —
DRIFT); safety (fraud, execution anomalies, abnormal routing, exposure
concentration — Shield Router); endurance (discipline, recovery after
failures, governance adherence — SURVIVOR).

## Continuous credit

Eligibility evolves after every governed execution — each updates repayment,
execution, governance, and behavioral quality. There is no fixed
monthly/quarterly review; every execution is a review. Each financed
execution emits identity, execution, risk, capital-allocation, policy,
settlement, governance, and audit receipts.

## Beyond trading

Governs any capital-backed execution: trading, business lending, invoice
financing, marketplace settlements, supply-chain finance, procurement,
milestone payments, AI-agent execution budgets, compute credit, API-usage
credit, and commercial financing. It governs capital allocation, not merely
loans.

## Doctrine

Reputation opens the conversation; governance authorizes the capital.
Capital should follow verified execution confidence under current
conditions, not historical success alone.

## Cross references

Capital Admissibility Framework (project-side sibling) · Domain-Aware
Capital Intelligence (evaluator-side sibling) · Future Rights Exchange
(claims that mature over time) · Ownership Proofs vs Execution Rights
(possession vs authorization) · Meta-Architecture: Observation to Strategic
Moat (GCE sits at the Admissibility/Execution boundary).
