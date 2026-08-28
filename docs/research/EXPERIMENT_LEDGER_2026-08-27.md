# Experiment Ledger — 2026-08-27

**Purpose.** This ledger records experimentally learned truth across the portfolio. It is distinct from implementation status and from future reserves.

**Status semantics.** `VALIDATED` / `REJECTED` / `INCONCLUSIVE` / `ACTIVE` / `PROPOSED` / `BLOCKED` may apply at the individual-claim level rather than to an entire experiment. One run can validate one claim while rejecting another.

**Research doctrine.** Negative results are first-class evidence. A rejected hypothesis is retained when it narrows the design space or changes an architectural assumption.

**Build boundary.** Inclusion in this ledger is **not** implementation authorization. Zircon remains reserve-only.

**Provenance convention.** Every entry carries a footer of the form
`source artifact → experiment date → implementation commit → evidence boundary → conclusion date`.
The evidence boundary is the scope within which the conclusion holds. It exists because three separate results in this ledger — E1, E8/E9 and E12 — each failed in a different dimension of provenance: world provenance, hardware scope, and temporal validity. A conclusion without its boundary is not preserved, only remembered.

---

## Method rules that produced this document

- An empty result is evidence only if the instrument is proven capable of producing a non-empty one.
- Observation ≠ system state until source scope and provenance are established.
- Commit count is a proxy, not a state change.
- Co-location is not dependency. Dependency is not ownership.
- A reserve is not a completed experiment. A unit test is not an experiment. A JSONL file is not proof that a hypothesis was tested.

---

## Doctrine extracted from measured results

| Source | Finding |
|---|---|
| E3 | **Any denominator policy is a different way of dividing by a hole.** Missing evidence is an acquisition/coverage problem before it is a scoring problem. |
| E4 | **Evidence quality and claim scope must match.** Structural evidence cannot silently become a claim of overall safety. |
| E6 | **Pre-registration worked.** A criterion fixed before observation let an unfavourable result be accepted without moving the goalposts. |
| E8/E9 | **A regime boundary beats a universal answer.** Fair Compute found where the transition sits, not who wins. |
| E1 | **Ledger truth ≠ world truth.** A perfectly reconciled receipt can faithfully preserve a decision made from contaminated evidence. |
| E11 | **Prediction → production measurement → confirmation.** The expected operating rate existed before the measurement was taken. |

---

# Completed and active experiments

## E1 · Evidence Epoch 1 — HELIX-JANUS

| Claim | Status |
|---|---|
| Receipt integrity | **VALIDATED** |
| Governance behaviour | **VALIDATED** |
| Market-source provenance | **FAILED** |
| P&L as strategy evidence | **REJECTED — inadmissible** |

**Hypothesis.** A receipts-before-execution loop can produce a complete, internally consistent, non-repudiable record of every adjudication.

**Mechanism.** OBSERVE → ADJUDICATE → EXECUTE → SETTLE → RECEIPT → LEARN. Paper-locked under a triple lock (`LIVE_MODE` + `!PAPER_MODE` + `CONFIRM_LIVE_EXECUTION`); live adapter is a deliberate stub; no broker credentials exist.

**Evidence.** Sealed by SHA-256 plus line count:

    data/receipts.jsonl  lines=3861  sha256=1029461fb77262db4ddba7cb3073cce34f794793e34e6e343d15041bf6e80136
    data/trades.jsonl    lines=137   sha256=cafdde182cba5b4081aa3935f172b627b4bbaad85ec009638a0109172c8046b0
    data/state.json      lines=74    sha256=db8972aab75bd40cc2737a39e9982b050092ed845561062204df902653d08c21

Reconciliation is exact: 1,862 adjudications × 2 writes + 137 settlements = 3,861 lines. Receipt ids strictly monotonic; zero duplicate lines; `doctrine_violations 0`. Decisions ALLOW 138 / DENY 1,661 / ESCALATE 57. Outcomes 137 closed, 70 wins, 67 losses, gross P&L −387.67.

**Result.** The ledger reconciles perfectly and is the strongest component in the system. But an unknown subset of decisions ran on synthetic random-walk candles and nothing in the record distinguishes them.

**Learned.** A record can be perfectly honest about what a system decided and worthless as evidence about the world. Internal consistency is orthogonal to external validity.

**Withdrawn finding.** A "duplicate ledger" defect was reported during reconnaissance and is **withdrawn** — it compared a line count to a receipt count, two different things. Recorded so the false version is not rediscovered.

**Unknown.** Which specific receipts were mock-contaminated. Epoch 1 may be partially reconstructable; do not assume write-off.

**Capital** none · **Runtime risk** none · **Reusable** the sealed corpus; the line-count-vs-counter reconciliation method.

*Provenance:* `helixjanus docs/EXECUTION_HANDOFF.md` → 2026-06-14 … 2026-08-21 → sealed at `912e68a` → **boundary:** one machine, paper mode only, yfinance-or-synthetic sources with no per-receipt source attribution; conclusions apply to receipt mechanics, **not** to market behaviour → concluded 2026-08-21.

---

## E2 · Evidence Epoch 2 — HELIX-JANUS · **ACTIVE**

**Hypothesis.** If every observation carries its own provenance and a failed fetch defers instead of substituting, the Epoch 1 contamination class becomes impossible rather than merely unlikely.

**Mechanism.** `Observation` dataclass (`requested_source`, `source`, `fallback_reason`, `fetched_at`); `is_usable` gate evaluated *before* `candles[-1]`; one `DATA_UNAVAILABLE` deferral receipt per symbol per cycle; `evidence_epoch: 2` stamped on every receipt with settlement inheritance; DRIFT vocabulary renamed to advisory (`STABLE / ELEVATED / DEGRADED / RETRAIN_SIGNALLED`) so an observation cannot wear an instruction's name.

**Evidence so far.** 16 receipts since the seal. `RCP_20260821_002007` — `evidence_epoch: 2`, `data_source: YFINANCE`, `fallback_reason: None`. 31 tests.

**Unknown.** No real fetch failure has occurred. **The deferral path is unit-tested but not production-exercised.**

**Capital** none · **Runtime risk** none (paper triple-lock) · **Next** run to a stated N and require at least one genuine deferral.

*Provenance:* `helixjanus docs/EXECUTION_HANDOFF.md` → opened 2026-08-21 → `88ff0f6`, `b7fbafa`, `8306baa` → **boundary:** one machine, paper mode, two instruments (QQQ 15m, USO 4h), yfinance only; the happy path is exercised, the deferral path is not → **open**.

---

## E3 · Shadow denominator — SURVIVOR Oracle · **REJECTED**

**Hypothesis.** Recomputing the evidence score over only measured-and-valid signals produces a better score.

**Mechanism.** `shadow = Σ(subscore × weight over included) / Σ(weight over included)`. Excluded `lpLocked` (measures burned LP, a launch convention) and `devWalletActivity` (never collected). Implemented as observation; live scoring unchanged.

**Evidence.** 20 tokens at 50–65% coverage.

**Result — rejected on three grounds.**

1. **It rewards ignorance.** JUP and RAY score a perfect 100 *because* their holder query failed, leaving only the signals they happen to pass. A token known less well outranks every token known better. This failure was predicted before implementation and appeared exactly where predicted.
2. **It does not discriminate.** Within the fully-measured group the delta ranges +16 to +24 — near a constant offset. Ranking is nearly unchanged from live.
3. **It compresses the top.** Five distinct tokens tie at 91 where the live model separated them.

**Learned.** The denominator is not the problem to solve. Any denominator policy is a different way of dividing by a hole. The problem is upstream: two of seven signals are globally unmeasured.

**Kept.** `shadow_denominator` retained in scorer output as `enforced: false` observation — ongoing evidence about what a reduced denominator would do as coverage changes. Costs nothing.

**Capital** none · **Reusable** the 20-token panel; the predicted-failure-confirmed pattern.

*Provenance:* `survivor-oracle docs/SHADOW_DENOMINATOR_RESULT.md` → 2026-08-02 → `measured-evidence-v0.5.1-shadow` → **boundary:** 20 Solana tokens at 50–65% signal coverage, seven-signal scorer; conclusion concerns denominator policy under partial coverage, not scoring in general → concluded 2026-08-02.

---

## E4 · Validated Five — SURVIVOR Oracle

| Claim | Status |
|---|---|
| Honesty (removal of fake/unmeasured signals) | **VALIDATED** |
| Discrimination improvement | **REJECTED** |
| Semantic contract | **REFRAMED** |

**Hypothesis.** A fixed five-signal model — not per-token renormalization — beats a model carrying a silent penalty (`lpLocked` null, its weight still in the denominator) and a constant (`devWalletActivity: 50` contributing 7.5 points to every token while coverage simultaneously reports NOT_COLLECTED).

**Mechanism.** Weights fixed in advance: mintAuthority 31 · freezeAuthority (transfer control) 15 · holderConcentration 23 · tokenAge 15 · liquidityDepth 16. If any required signal fails to resolve, `score_status: INCOMPLETE`. **The denominator is never reduced** — this is the explicit correction to E3.

**Evidence.** 16 tokens. Mean delta +20.6; all 16 crossed bands; 10 gates loosened; **ordering nearly unchanged**; every delta between +11 and +24. The five tokens tied at 67 remain tied at 91.

**Learned.** Removing defects that affect almost every token similarly shifts the scale without changing what the model knows. Recalibrating bands to fit would reproduce the current tier assignments with different numbers on them.

The genuine finding is semantic: the score carries **two incompatible claims** — *structural evidence* (how favourable are the on-chain properties evaluated?) versus *token safety* (how likely is this token to be safe or durable?). SLERF at 79 is defensible under the first and misleading under the second. The five signals do not measure abandonment, project continuity, treasury behaviour, development, governance or market survival.

**Decision taken.** Do not recalibrate LOW/MEDIUM/HIGH — those labels imply a broader verdict than five structural signals support. Change the semantic contract first: `score_name: structural_evidence_score`, with explicit does-not-claim language.

**Capital** none · **Reusable** the fixed-weight-with-INCOMPLETE design; the two-claims distinction, which generalizes well beyond tokens.

*Provenance:* `survivor-oracle docs/VALIDATED_FIVE.md` → 2026-08-04 → `validated-five-v0.6.0-shadow`, production remained 0.5.3 → **boundary:** 16 Solana tokens, five structural on-chain signals; establishes nothing about project viability, returns, developer continuity or market durability → concluded 2026-08-04.

---

## E5 · LP signal applicability, three variants — SURVIVOR Oracle · **REJECTED**

LP capped at 85; LP removed with weight redistributed; LP as a bounded bonus. Each rejected on population-level measurement.

*Provenance:* cited in `survivor-oracle docs/SHADOW_DENOMINATOR_RESULT.md`; primary source `survivor-oracle docs/LP_SIGNAL_APPLICABILITY_RESEARCH.md` **not read during this audit** → **boundary:** status carried on citation, not verification. Re-verify before relying on it.

---

## E6 · OHLCV shadow evaluation — Momentum Sniper · **REJECTED (integration); doctrine unchanged**

**Mechanism.** Evaluation protocol and promotion gates **locked before observation**. Shadow window opened, then closed at a fixed sample.

**Evidence.** 205 distinct mints.

**Result.** Coverage FAIL → integration rejected. Root cause diagnosed as **GeckoTerminal indexing lag, not pool absence** — 18/18 resolved mints lagged.

**Learned.** A criterion fixed in advance allowed a negative result to be accepted cleanly rather than renegotiated. This is the portfolio's clearest instance of pre-registration working.

**Capital** none.

*Provenance:* `momentum-sniper-doctrine` → criterion locked `5465c01` (2026-07-16) → window closed `b85a07d` (2026-07-21) → root cause `269f4c8` (2026-07-21) → **boundary:** 205 distinct mints observed through one OHLCV provider; establishes a coverage failure for that provider on that population, not that OHLCV integration is unachievable → concluded 2026-07-21.

---

## E7 · Performance criterion and reset — Momentum Sniper · **INCONCLUSIVE; observation continues**

**Evidence.** n = 6,695 cumulative paper trades at the gate review. Performance gate failed on **median −8.03%**. Criterion RESET; observation continues under OUTCOME B.

**Open question.** Persistent negative median with positive mean.

**Unknown.** Not verified during the 2026-08-27 audit — the remote runtime was not probed.

**Capital** none (paper only).

*Provenance:* `momentum-sniper-doctrine` (doctrine/observation record) → gate review 2026-07-13 → criterion `42dd50f`, doctrine `b1181f9` → **boundary:** paper-only observation on one venue population; no live capital, no live fills, so the record establishes signal behaviour under simulated execution and not realized performance → **open**.

---

## E8 · Fair Compute Milestone A (browser vs native) — iam-integrity · **VALIDATED, narrowly**

**Hypothesis.** Browser WebAssembly and native Rust execute a deterministic dependent-memory workload at comparable speed.

**Mechanism.** A single `implementation_hash` is enforced across runtimes; the report generator refuses to emit a comparison when it differs, so ratios are between the same source compiled two ways.

    implementation_hash = 50f713afa874d4456c399eb592d14f4260fa544c37c30f116a811b293c1110c6

**Result.** **0.98× browser/native** on the stable DRAM-latency-bound sizes (128 MiB, 256 MiB), on `min`. Smaller sizes excluded from the headline as too noisy to quote — 8 MiB showed 192% native spread.

**Learned.** Runtime parity holds once memory latency dominates execution. Equally important: the noise-flagging discipline kept unstable measurements out of the headline instead of averaging them in.

**Explicitly not claimed.** Hardware fairness.

*Provenance:* `iam-integrity bench/fair-compute-bench/results/report/comparison.md` → generated 2026-07-21 → single implementation hash above → **boundary:** one deterministic dependent-memory workload, one machine, headline drawn only from sizes with ≤15% relative spread in both runtimes → concluded 2026-07-21.

---

## E9 · Fair Compute Phase 2 (GPU vs CPU) — iam-integrity

| Claim | Status |
|---|---|
| Parameter-sensitive hardware result | **VALIDATED** |
| General fairness thesis | **INCONCLUSIVE** |

**Question.** Can a massively-parallel GPU outperform a deliberately latency-bound, dependency-chained workload while preserving identical execution semantics?

**Mechanism.** Determinism gate on every row — the Metal kernel reproduced the CPU/native digest, so both devices ran the same workload. Workers scale as `budget / per-worker size`.

**Result — non-monotonic.** Peak GPU advantage **2.66× at 4 MiB/worker** (256 workers). At 256 MiB/worker (4 workers) the **CPU wins at 0.38×**.

**Learned.** There is a **vulnerability window** at intermediate scratchpad sizes, where the CPU has fallen off its cache cliff but enough parallel chains still fit to keep the GPU busy. Concrete design implication: a fair-compute workload must size per-worker scratchpad above roughly 32 MiB on this hardware, or a GPU farm reclaims a real edge. This turns scratchpad size from an arbitrary knob into a studied variable.

**Unknown.** Where the transition sits on other hardware.

**Blocked by capital.** Phase 2B requires a data-center GPU (CUDA).

*Provenance:* `iam-integrity bench/fair-compute-bench-gpu/results/report/phase2.md` → generated 2026-07-21 → determinism gate passed on every row → **boundary:** Apple M1 integrated GPU vs Apple M1 CPU (8 threads), one aggregate memory budget, one implementation, one machine; tests parallel throughput, not single-chain latency (Phase 1) and not adversarial Sybil replication (Phase 3). **Not established:** general GPU resistance → concluded 2026-07-21.

---

## E10 · EHFC engine vs doctrine — Backpack Engine

| Claim | Status |
|---|---|
| Doctrine compliance audit | **VALIDATED** |
| Simulation results | **INCONCLUSIVE** (mock data only) |
| Divergence queue | **DESIGNED_NOT_RUN** |
| System | **BLOCKED by policy** |

**Result.** Six doctrine points verified compliant in code: one-trade-per-event, no mid-price fills, no manual review or live capital, HOT-only entries (stricter than spec), exit thesis with all experiments disabled by default, required logs. Nine divergences catalogued.

**Highest-priority gap.** The Flow/ADV admission gate — the spec's central admission gate — is **unimplemented**, and `MISSING_FLOW_DATA` actually fires on index membership, a proxy label. Also: HOT trigger is 3-of-3 where spec says 2-of-3; micro caps tradeable in code but untradeable in spec; slippage numbers more optimistic than spec; `TIME_STOP` unimplemented, so a position that never sees liquidity normalize holds forever.

**Learned.** The code predates the reconstructed spec. Tests encode *current* behaviour so any future change is deliberate and visible, and test names flag the doctrine gaps.

**Blocked.** Activation gate is Momentum Sniper proving live cashflow — currently unmet.

*Provenance:* `backpack-engine docs/RECONCILIATION.md` → reconciliation 2026-07-15, engine last run 2026-01-14 → recovered baseline `a2cceb8`, schema v1.2 → **boundary:** mock data only, no broker, no capital; the compliance audit is about code-vs-spec conformance and says nothing about strategy validity → concluded 2026-07-15.

---

## E11 · Mediation layer under production load — API Connect · **VALIDATED**

**Prediction, recorded before measurement.** Approximately 2 upstream calls per minute under the cache-first design.

**Evidence.** 66-hour window: 46,300+ requests, zero exhausted. `swap_metrics` 34,215 served at 78% cache hit with 7,554 upstream calls at 99.95% success — **1.9 upstream calls/min**. Negative cache and stale-serving both exercised in production. Consumer side: poller 429 rate **99.6% → 2.7%** across 30,256 fetches, with 826 fail-open fallbacks absorbed by preserved backoff.

**Learned.** A quantitative design prediction, stated in advance, held under production load. This is the strongest engineering evidence in the portfolio precisely because the expected rate existed before the measurement.

**Capital** none · **Reusable** `verify-ops.sh`; the operational-receipt table (primary/fallback/serving/failovers per capability).

*Provenance:* `api-connect docs/NEXT-SESSION.md` and registry entry → 66h window ending 2026-07-21 → production `e958c32` → **boundary:** one provider mix, one chain (Solana), three data types, one consumer; the confirmed prediction concerns upstream call rate under cache-first mediation, not provider correctness → concluded 2026-07-21.

---

## E12 · Temporal admissibility — VERITY / Litmus · **PROPOSED**

Naturally occurring instance observed 2026-08-26. Authoritative production VERITY refreshes on an internal 6-hour timer and reported a last-refresh age of ~2h at observation. The local checkout stopped advancing 2026-03-13. A downstream component retains a code path that consumes the local artifact. Filesystem existence alone could not distinguish current evidence from stale evidence, and the condition persisted undetected for approximately five months.

The obsolete local refresh mechanism was retired on 2026-08-27; the freshness question it exposed was not resolved by that retirement.

*Provenance:* observed during portfolio audit → 2026-08-26/27 → local checkout `5af40f2` (2026-03-07), local data artifacts dated 2026-03-13 → **boundary:** one consumer, one upstream, one naturally occurring instance; a single observed case is not a tested hypothesis → **open, see Top 5 #1**.

---

## E13 · Evidence-source attribution — Litmus · **PROPOSED**

`litmus-firewall/pow_fetcher.py` runs the scorer in the local checkout and reads the local leaderboard artifact, while setting the emitted receipt's `url` field to the VERITY **production** health endpoint. The receipt therefore cites an authoritative source it never read — and that endpoint would show a healthy recent refresh, appearing to corroborate figures it played no part in producing. Structurally valid, internally consistent, and wrong about its own origin.

*Provenance:* `litmus-firewall/pow_fetcher.py` lines 126 / 128 / 138 → observed 2026-08-26 → repository established `0503ccd` (2026-08-21) → **boundary:** one code path in one component, read statically; not yet executed under observation → **open, see Top 5 #1 Claim B**.

---

# TOP 5 — pre-Zircon candidates

Selection constraints: existing machinery only, no new architecture, zero capital, no runtime risk, no activation of blocked systems, and **no interference with third-party infrastructure**.

## 1 · Stale-but-valid artifact — does governance detect it? · **BLOCKING candidate**

Two independent claims, recorded separately so one run cannot confound them.

**Claim A — temporal admissibility.** *Hypothesis:* the consumer cannot distinguish a stale artifact from a current one. **Validated = defect confirmed.**
**Claim B — evidence-source attribution.** *Hypothesis:* the receipt's cited source is not the source the data came from. Independently true or false regardless of Claim A.

Each concludes as `VALIDATED` / `REJECTED` / `INCONCLUSIVE` on its own line.

**Why now.** Zircon inherits an evidence/receipt model. If freshness and source attribution are not part of admissibility, every receipt inherits the hole. This was found by accident; it would not have been found by design.

**Machinery reused.** The VERITY local/production pair, already divergent by five months and both readable. `litmus-firewall/pow_fetcher.py` as negative control. HELIX-JANUS `Observation` as **positive control** — a system in the same portfolio that already defers correctly.

**Minimum experiment.** Run the Litmus VERITY path against the current stale local artifact; capture the emitted receipt; test whether anything in it distinguishes a months-old artifact from a current one (Claim A) and whether its cited source corresponds to the data actually read (Claim B). Then run the HELIX-JANUS fetch path against an unavailable source and compare what each records.

**Accept / reject.** Claim A rejected if the path independently flags staleness or degrades confidence. Claim B rejected if the receipt's cited source matches its actual source, or if it records both.

**Window.** Under an hour. **Decision unlocked.** Whether freshness bounds and source attribution must be first-class receipt fields before Zircon defines a receipt schema.

## 2 · Do consumers honour `INCOMPLETE`? · **BLOCKING candidate**

**Question.** The validated-five design emits `score_status: INCOMPLETE` when a required signal fails to resolve. Do downstream consumers gate on it, or consume it silently?

**Why now.** Dependency and finality semantics, inherited directly by Zircon. E3 already proved that missing data can *inflate* a score. If `INCOMPLETE` is not enforced downstream, the guarantee exists only at the producer and is unenforceable.

**Machinery reused.** survivor-oracle scorer, survivor-shield-sdk, agentguard, poi-engine consumer path — all present, none modified.

**Minimum experiment.** Static trace of `score_status` from producer to every consumer, then one probe with a deliberately unresolvable signal.

**Accept / reject.** Defect confirmed if any consumer proceeds on `INCOMPLETE` without degrading or deferring.

**Window.** 2–3 hours. **Decision unlocked.** Whether admissibility must be enforced at the boundary or can be delegated to producers.

## 3 · Does the replay harness still reproduce its own baselines?

**Why now.** The Zircon gate names a replay model explicitly, and replay apparatus already exists: a replay runner with ten baseline/result JSON files across four versions. **It is not under version control**, so nothing guarantees the baselines still correspond to the runner.

**Minimum experiment.** Run the runner against the recorded baseline; diff against the recorded results.

**Accept / reject.** Reproduces → working replay apparatus is available for every later experiment. Does not → a dated determinism finding, which is itself the answer.

**Window.** Under an hour. **Decision unlocked.** Whether replay is available machinery or must be rebuilt — this gates the cost of repeating experiments 1 and 2.

## 4 · Do two independent providers agree?

**Question.** Two providers both serve `ohlcv_5m` and `ohlcv_4h`. Asked the same question, do they give the same answer?

**Why now.** Trust model. Failover assumes providers are interchangeable; that has never been measured. E6 already found one provider lagging in a way that looked like absence.

**Machinery reused.** Both adapters exist; capability telemetry records serving provider and failover count. Measured by a read-only harness **outside** api-connect — no code change, reserves untouched, parked state preserved.

**Accept / reject.** Divergence beyond a stated tolerance means *available* and *correct* are different capabilities, and readiness cannot conflate them.

**Window.** A day of sampling. **Decision unlocked.** Whether redundancy is fallback or quorum.

## 5 · Complete Epoch 2 to a stated N, with at least one real deferral

**Why now.** Already running, costs nothing, and is the only controlled execution experiment with clean provenance. Its weakness is precise: the deferral path has never fired in production.

**Accept / reject.** N cycles with zero unattributable receipts **and** at least one genuine `DATA_UNAVAILABLE` receipt.

**Constraint on provocation.** A deferral must come from a controlled local or test-path failure, or from a naturally occurring fetch failure. **Manufacturing an outage at an external provider is out of scope** — it would interfere with third-party infrastructure, and it would make the deferral synthetic, testing the code path rather than the condition.

**Window.** Continuous. **Decision unlocked.** Whether provenance-carrying observation is sufficient to close the Epoch 1 contamination class.

---

## Ranking

**1 and 2 are the blocking candidates** — both test admissibility rules Zircon would otherwise inherit unexamined.
**3 is the cheapest and gates the cost of everything after it.**
**4 and 5 are informative but non-blocking.**

None requires capital, new architecture, Backpack activation, Litmus modification, or reserve commitment.

## Execution discipline

One bounded experiment at a time. Hypothesis and acceptance conditions frozen in writing before observation, per E6. Negative results are first-class and recorded with the same weight as confirmations. Every conclusion is recorded with its evidence boundary.

**Zircon remains reserved, not activated.**
