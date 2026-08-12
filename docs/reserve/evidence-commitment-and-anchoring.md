# RESERVE - Evidence Commitment & Independent Anchoring

**Status:** Reserved. No blockchain integration or receipt migration authorized.
**Scope:** vLOID, OROS, VERITY, LITMUS, HelixAtlas, execution receipts.

## Principle
A receipt should not claim something is true because the receipt exists or is anchored. It
should make later claims about what was observed, evaluated, decided and executed
**independently testable**.

External anchoring strengthens integrity and existence claims. It does not establish truth.

## Four concepts that must stay separate
    EVIDENCE          what was observed or supplied. Not automatically truth.
    INTERPRETATION    what the system concluded. May be wrong even on genuine evidence.
    AUTHORITY         what the system is permitted to do. Separate from evidence.
    EXECUTION OUTCOME what actually occurred. Recorded apart from the decision.

Do not collapse the chain into one ambiguous "proof" object.

## What a commitment proves
A hash proves the committed bytes correspond to that hash. It supports claims that a
package existed in that form, has not been altered, was referenced by a receipt, or was
included in a batch.

It does NOT prove the source was truthful, the observation complete, the interpretation
correct, the policy wise, the execution legitimate, or that the world matched the evidence.

## Graduated durability - do not anchor everything
    LEVEL 1  internal receipt        signed, timestamped, append-only, replayable
    LEVEL 2  cryptographic commitment where alteration detection matters
    LEVEL 3  independent anchor      only where an external trust boundary adds real value

Blockchain is one implementation of level 3, not the doctrine. Trusted timestamping,
transparency logs and independent archival are alternatives. Selection depends on
permanence, cost, verification accessibility, privacy, throughput, reliability,
jurisdiction and longevity.

## Batching
Do not assume one transaction per receipt. If anchoring becomes useful, prefer Merkle
batching so many receipts share one commitment while preserving inclusion proofs. Not until
volume justifies it.

## Privacy
Public anchoring must not become public data leakage. Commit hashes, not raw evidence.
Never anchor credentials, private account data, personal information, confidential
documents, proprietary prompts, customer data or internal secrets.

## The closed loop
    evidence -> VERITY -> LITMUS -> OROS -> authorization -> execution -> receipt
    -> becomes future evidence

Today's receipt is tomorrow's evidence. That is an auditable execution history rather than a
pile of unrelated logs.

## Worked example - the August 2026 quota incident
A mature chain would record: provider observation, quota and degradation evidence, DRIFT
detection, routing decision, API Connect result, OROS decision, downstream result, receipt,
commitment.

Months later a reviewer could verify what the system knew, what it decided, why it acted,
and whether those records were later changed. The commitment does not claim the decision was
optimal - it establishes the integrity of the trail.

## Core doctrine
Evidence is not truth. A hash is not truth. A blockchain transaction is not truth. An anchor
is not truth. No single mechanism substitutes for the others.

## Non-goals
No blockchain product for receipts. Not everything on-chain. "On-chain" is not "true."
Anchoring does not replace VERITY or governance. Evidence receipts are not execution
authority. Do not call an anchored record legal proof without legal basis.
