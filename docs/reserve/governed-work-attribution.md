# RESERVE — Governed Work Attribution

Status: RESERVED — organizational execution primitive. NOT an active build.
Captured: 2026-08-27.
Scope: deliberately cross-cutting. Consumed by HELIX Builders, WIRE, ShiftTrust,
agent teams and any governed multi-contributor system. Not owned by any one of them.

Consolidates four mechanisms that arrived separately and belong together:
Stage-Gated Work Attribution · Outcome Attribution Graph · Opportunity Readiness Gate ·
Persistent Contribution Attribution. Plus the Bounded Responsibility doctrine.

## 1. Stage-Gated Work Attribution (SGWA)

Complex work decomposes into bounded stages with admissibility criteria and receipts.

    source / opportunity → contributor assignment → stage prerequisites
    → contributor action → evidence → stage completion decision → handoff
    → downstream stage → outcome → attribution graph
    → compensation / reputation / learning receipt

A stage is not complete because a contributor says so. Completion may require required
inputs present, task criteria satisfied, evidence attached, validation performed,
unresolved issues disclosed, dependencies identified, next responsible party identified,
handoff context available, completion receipt issued.

## 2. Outcome Attribution Graph (OAG)

    Opportunity → discovered by A → validated by B → diagnosed by C
    → implemented by D → reviewed by E → accepted by customer → settled

Prevents the final actor from automatically receiving all attribution, and prevents
earlier contributors from claiming outcomes their actions do not support.

**Critical distinction — work completion vs downstream outcome.** A contributor can
successfully perform their stage even if a later stage fails. TASK_RECEIPT and
OUTCOME_RECEIPT are separate artifacts. This prevents responsibility laundering and
incorrect compensation attribution.

## 3. Opportunity Readiness Gate (ORG)

Before assigning a specialized contributor, agent, capital allocation or other expensive
capability, determine whether prerequisite conditions are satisfied.

    raw opportunity → prerequisite discovery → evidence collection → qualification
    → readiness assessment → READY / DEFER / REJECT / ESCALATE
    → specialist assignment → execution

**Readiness is distinct from trust.** An opportunity can be entirely legitimate and not
ready. VERITY may find the customer and evidence trustworthy while the issue still lacks
reproduction steps, logs, scope, environment, authorization, budget or acceptance criteria.

**Readiness is distinct from matching.** The system may know exactly which expert could
solve something and still decline to assign them yet.

**Governing efficiency doctrine:**
*Do not consume expensive capability to discover cheap missing prerequisites.*

Readiness rigour is proportional to execution cost, risk, irreversibility, scarcity of
specialist capacity and uncertainty. Cheap reversible actions may require almost none.

Generalizes well beyond sales: software (bug → evidence → reproduction → engineer),
security (signal → severity/context → investigator), WIRE (capability → environment →
success criteria → contributor), ShiftTrust (request → eligibility → safety → worker),
capital (transaction → bottleneck → repayment source → decision), agent execution
(intent → context → permissions → required evidence → authorization).

## 4. Persistent Contribution Attribution (PCA)

When a contribution occurs once but helps create economic value that persists,
attribution need not terminate when the contributor's immediate task ends.

    contribution → attribution receipt → qualified handoff → downstream execution
    → outcome → persistent value → attribution remains active
    → recurring settlement / recognition → termination condition → final receipt

**PROVENANCE ≠ ENTITLEMENT.** The graph records what happened; policy determines what
that history means economically. A system may permanently record that A originated an
opportunity without granting A permanent economic rights.

Persistence conditions must be explicit (customer active, revenue continuing, contract
valid, contribution still materially connected). Termination events must be explicit
(churn, expiration, buyout, maximum period, revenue threshold, agreement termination,
misattribution finding, material transformation, policy sunset).
**Never imply perpetual rights merely because provenance is permanent.**

Attribution weight models to research, none chosen: FIXED · DECAYING · MILESTONE ·
CAPPED · PERMANENT PROVENANCE WITH TEMPORARY ECONOMIC RIGHT.

## 5. Bounded Responsibility

A contributor should know explicitly what they control, what they must verify, what
constitutes completion, what must be escalated, who receives the work next, and
**which downstream outcomes they do not control**.

Cleaner accountability than assigning vague responsibility for an entire business
outcome to every participant.

## Full loop

    RAW OPPORTUNITY → QUALIFY → READINESS → MATCH → ASSIGN → EXECUTE → VERIFY
    → HANDOFF → OUTCOME → ATTRIBUTE → PERSISTENCE CHECK → SETTLE → RECEIPT

## Gaming surface

Fake referrals, self-referrals, duplicate sourcing claims, last-touch appropriation,
collusion, fake retention, sybil contributors, recycled opportunities, discovery claimed
after evidence already existed. VERITY evaluates attribution evidence. Disputes carry
first-class states: UNCONTESTED / CONTESTED / UNDER_REVIEW / SUPERSEDED /
SPLIT_ATTRIBUTION / INVALIDATED. Corrections create superseding receipts; history is
never silently rewritten.

## Doctrine

**VALUE MAY OUTLIVE LABOUR.** A contributor's immediate work can end while the value it
created continues. Governed work systems should distinguish time worked from value
created from value persistence from economic entitlement — without implying that every
persistent contribution deserves recurring compensation.

## Relationship to existing canonical reserves

- `computable-accountability.md` — receipt and responsibility-chain substrate.
- `flow-economics-engine.md` — economic attribution layer.
- `universal-execution-timeline.md` — execution memory.
- `helix-builders.md` — a consumer, not the owner.
- `contributor-continuity-handoff-gate.md` — continuity of responsibility on departure.
- `helix-workspace-youth-distributed-capability-learning.md` / ShiftTrust — bounded
  responsibility for inexperienced workers remains subordinate to those safeguards.

## Anti-patterns

Equating activity with contribution; equating contribution with final outcome; giving
the final actor automatic ownership of prior work; punishing upstream contributors for
downstream failure; attribution without evidence; handoffs that erase responsibility;
compensation dependent on opaque attribution; economic rights surviving beyond
contractual authorization; structures that create securities, employment, tax or
licensing obligations without legal review.

## Activation

Revisit when several independent contributors participate in one outcome, work
frequently changes hands, compensation or reputation depends on contribution, disputes
about responsibility become possible, or contributor continuity becomes operationally
important.

RESERVED. DO NOT BUILD.
