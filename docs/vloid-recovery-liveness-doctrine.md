# vLOID — Deterministic Recovery & Liveness Doctrine
**Status:** RESERVED · Core execution-governance doctrine · **Reserved:** July 21, 2026
**Parent:** vLOID · Primary runtime: OROS · Supporting: IAM, VERITY, LITMUS, DRIFT, Shield Router, VYRE
**Source pattern:** Failure-resistant, time-bound, escrowed execution flows observed in distributed and on-chain systems.

## 1. Purpose
vLOID determines admissibility, coordinates execution, verifies identity, monitors drift, produces receipts. This reserve adds the missing question: what must happen when an approved execution is duplicated, interrupted, delayed, partially completed, abandoned by its initiating service, or challenged after completion? The system must not merely prevent bad execution — it must recover from incomplete execution deterministically and auditably.
Lifecycle: Intent > Identity > Admissibility > Simulation > Authorization > Execution > Receipt > Settlement > Recovery/Challenge.
Recovery is not an exception outside the architecture. Recovery is a first-class execution state.

## 2. Core doctrine
### 2.1 Never let the actor define reality
A client, operator, agent, wallet, browser, or model must not independently declare: current time, completion, fund movement, approval, lock expiry, receipt validity, retry permission. Authoritative state comes from approved sources: signed system clock, blockchain clock, TEE, settlement provider, verified database state, signed receipt chain, independently observed external evidence.
Mapping: VERITY — is the evidence source trusted; LITMUS — does the claimed state satisfy doctrine; OROS — what happens next.
### 2.2 Every execution must be idempotent
Requests may arrive repeatedly (network retries, duplicated messages, agent loops, refreshes, provider callbacks, failover, queue redelivery, malicious replay, human double-submission). Repeated execution must not create repeated consequences unless duplication is explicitly allowed. Every governed action carries idempotency identity: execution_id, intent_hash, actor_id, policy_version, operation_type, resource_id, execution_epoch.
Behavior: first valid request > execute · exact duplicate > return existing result · conflicting duplicate > deny or challenge · incomplete earlier request > resume or compensate per state. OROS never blindly executes the same intent twice merely because it received two requests.

## 3. Canonical execution-state machine
PROPOSED > SIMULATED > ADMISSIBLE > AUTHORIZED > IN_PROGRESS > SETTLED.
Branches: PROPOSED>DENIED · SIMULATED>MUTATED · AUTHORIZED>EXPIRED · IN_PROGRESS>PARTIAL · IN_PROGRESS>FAILED · PARTIAL>COMPENSATING · PARTIAL>ESCROWED · FAILED>RETRYABLE · FAILED>TERMINAL · SETTLED>CHALLENGED · CHALLENGED>UPHELD · CHALLENGED>REVERSED.
No vague success/failure: state must explain where execution stopped and what remains permitted.

## 4. Simulation-before-authorization
Before authorization, show expected consequences: assets moved, accounts affected, permissions granted, fees, counterparties, lock period, settlement route, external calls, policy decisions, possible failure states, recovery conditions, expiration. Authorization binds to simulation: simulation_hash, intent_hash, policy_version, route_hash, expected_effects_hash, authorization_signature. If execution differs materially from the approved simulation, the authorization is invalid — preventing approval mismatch, counterparty substitution, amount/route mutation, permission escalation, hidden downstream actions. Direct extension of the Intent Verifier.

## 5. Custody-minimized execution
Execution rights controlled by policy, not by possession of a secret: program-derived accounts, multisig, policy-controlled escrow, hardware-backed service identities, threshold signatures, short-lived delegated credentials, bounded execution tokens, smart-contract vaults, provider-side restricted accounts. No operator bypasses doctrine via a master credential. IAM identifies; LITMUS defines permitted authority; OROS enforces; Shield Router verifies the approved path.

## 6. Permissionless or substitutable completion
An execution should not depend on the original backend/operator/agent staying online. Where safe, another authorized executor may complete an already-approved action: releasing an expired lock, finalizing settlement, publishing a receipt, completing a queued transfer, closing a completed workflow, triggering timeout compensation, verifying an artifact. Distinguish: who may INITIATE, AUTHORIZE, EXECUTE, FINALIZE, CHALLENGE — not always the same party. This creates liveness resilience.

## 7. Absolute-time doctrine
Store absolute deadlines at obligation creation: unlock_at = trusted_time + approved_duration, then persist. Never recompute deadlines from changing context. Record per time-bound execution: created_at, authorized_at, expires_at, execute_after, execute_before, settled_at, challenge_until, time_source, time_source_confidence. Avoid exact-time equality; use ranges (now >= execute_after AND now < execute_before). DRIFT flags clock-source divergence, settlement delay, stale authorization, long durations, repeated expiration, abnormal retry timing.

## 8. Pull-based settlement where appropriate
Prefer, where appropriate: eligible participant requests settlement > system verifies entitlement > executes or releases the claim — over fragile global push loops. Obligations remain representable independently, settled individually. Each entitlement: beneficiary, amount or computation rule, eligibility evidence, settlement status, claim identifier, expiration, source pool/obligation, receipt reference. OROS may coordinate either model; obligations stay inspectable and independently recoverable.

## 9. Atomicity and partial execution
Multiple changes in one logical execution either complete atomically or model partial completion + compensation explicitly. Never an undocumented half-state. If steps cannot be atomic, OROS tracks each: STEP_N_COMPLETE / FAILED / COMPENSATION_REQUIRED. Recovery decisions: retry remaining steps, reverse completed steps, escrow value, pause for human review, settle via alternate route, issue terminal failure receipt.

## 10. Replay and duplicate protection
Every execution boundary validates: actor identity, signer authority, resource ownership, expected program/service, policy version, nonce/sequence, expiration, execution identity, previous settlement state, route identity. Replays resolve deterministically: already settled > return prior receipt · still processing > return current state · expired > deny · same identity different payload > challenge · unknown execution identity > reject or re-adjudicate. Shield Router treats replay protection as a core gate.

## 11. Recovery classes
R0 SAFE RETRY — no irreversible effect (failed fetch, provider timeout, unsigned request, unavailable route pre-settlement); retry automatically.
R1 RESUME — some steps complete, remainder deterministic; OROS resumes from last verified checkpoint.
R2 COMPENSATE — irreversible/externally visible step completed but workflow failed; compensating action required (refund, reversal, escrow release, permission restore, downstream cancellation).
R3 ESCROW AND REVIEW — ambiguous state or conflicting evidence; pause value/authority while VERITY and LITMUS reassess.
R4 TERMINAL FAILURE — recovery impossible or prohibited; end with complete failure receipt, remaining exposure, responsible execution path, required human/legal follow-up. Properly receipted failure is a valid final system output.

## 12. DRIFT integration
DRIFT exists already; not a new module. DRIFT_EXEC — abnormal mutation within one execution: payload/route/signer/amount changed after approval, retry count exceeded, state-sequence violated, settlement != simulation, unauthorized recovery attempted. DRIFT_SYS — recurring systemic pressure: elevated timeouts, provider degradation, repeated compensation, clock divergence, growing unsettled obligations, unusual replay activity, rising recovery queues, rising finalization latency. DRIFT provides deviation evidence to OROS and LITMUS; it does not decide the remedy.

## 13. Required recovery receipt
Every recovery action produces a structured receipt: execution_id, original_intent_hash, original_authorization_id, failure_class (e.g. R2_COMPENSATE), failed_at_state, completed_steps, failed_step, recovery_action, recovery_actor, recovery_authorization (policy_bound), policy_version, verity_evidence, drift_flags, started_at, resolved_at, final_state, signature. Recovery receipts CONNECT to the original execution receipt; history stays append-only: intent > authorization > execution > failure > recovery > final settlement receipts.

## 14. HELIX and Solana relevance
For future Solana programs: PDA-controlled custody where appropriate; canonical account ownership checks; signer and program validation; on-chain time source; absolute unlock timestamps; duplicate participation rules; idempotent claim instructions; pull-based claims at scale; permissionless finalization where safe; deterministic account closure; explicit rent-return destination; safe partial claims; replay-resistant seeds and identifiers; transaction simulation before wallet approval; human-readable execution consequences.
Applies to: staking/participation programs, escrow, token distribution, rewards, vesting, governed settlement, RACER economic workflows, future $SURVIVOR accountability functions, Universal Money Router settlement flows, agent-controlled spending, RWA execution.

## 15. What not to build now
No generic staking product because this came from a staking discussion. Do not create: a new token-lockup application, a standalone recovery product, a second execution coordinator, a duplicate DRIFT module, a separate "liveness engine", an unnecessary blockchain dependency. This is doctrine and runtime behavior inside vLOID and OROS.

## 16. Activation gate
Activates when a live workflow has at least one of: funds under conditional control, delayed execution, multi-step settlement, cross-provider retries, expirable authorization, recoverable external side effects, asynchronous callbacks, permission delegation, human-agent joint approval, legally/financially significant failure.
First implementation stays narrow: one workflow, one state machine, one idempotency key, one deterministic retry path, one compensation path, one recovery receipt. Generalize only after that.

## Final placement
vLOID: VERITY (is evidence/actor trustworthy) · LITMUS (is the action doctrinally admissible) · IAM (who may continue acting) · OROS (what executes, settles, or recovers) · DRIFT (did execution or system behavior deviate) · Shield Router (did the request travel a valid path) · Deterministic Recovery & Liveness Doctrine (idempotency, simulation binding, time authority, atomicity, retry, compensation, escrow, challenge, recovery receipts).
This strengthens vLOID at the point where many systems fail: not at authorization, but after execution has already begun and reality becomes messy.
