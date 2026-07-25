# IMPLEMENTATION STATUS — Canonical Registry

This file records **implementation truth, not roadmap ambition**. It exists
because the July 15 audit found two prototypes (HELIX-JANUS, Backpack/EHFC)
whose real implementation state had disconnected from the portfolio record —
one running un-versioned on a laptop, one forgotten on a server.

Maturity scale:
- 🟢 Production — deployed, verified runtime, receipts/telemetry
- 🟡 Local prototype — implemented and runnable (possibly running) locally
- 🟠 Dormant/partial — functional implementation, not currently operated
- 🔴 Reserve only — architecture/doctrine exists, no implementation

Last full audit: **2026-07-15** (this file's baseline).

---

## 🟢 api-connect — production

- **Canonical repo:** github.com/ayubeay/api-connect · commit `e307d3c`
- **Runtime:** Railway (EU West), https://api-connect-production-b1a7.up.railway.app
- **Deployment mode:** auto-deploy on push to main; persistent volume
  `api-connect-data` mounted at /data
- **Operational state:** live; canary consumer (poi-engine) passed 2-day
  validation (2,260 requests, 71% cache hit rate, 0 consumer failures)
- **Implemented:** cache-first provider router, helius/jupiter-v3 provider
  (token_price), GeckoTerminal provider (normalized crypto OHLCV, 5m/4h,
  pool-resolved), Birdeye OHLCV fallback (gated on BIRDEYE_API_KEY),
  HeliusSwap provider (swap_metrics), sqlite cache on volume, execution
  receipts, Prometheus telemetry at /v1/metrics; THREE distinct ops
  endpoints (design rule, docs/NEXT-SESSION.md): /v1/admin/health
  (providers), /readyz (capability readiness, 503 only on critical-no-path),
  /v1/admin/capabilities (how each capability is fulfilled — route, failover
  count, last success — added 2026-07-24 Phase A); circuit breaker, negative
  cache, graceful shutdown, 52 tests
- **Known gaps:** intervals deliberately narrow (5m/4h only); no external
  alerting wired yet; equities/ETF asset class not supported (deliberate).
  RESOLVED 2026-07-22: multi-provider health observability (/v1/admin/health
  + gauge report all providers); capability-aware /readyz (503 only when a
  critical capability has no healthy path); OHLCV fallback provider (Birdeye,
  gated on BIRDEYE_API_KEY — geckoterminal primary, birdeye fallback, so a
  GT outage keeps ohlcv available instead of flipping /readyz to 503).
  RESOLVED 2026-07-25: BIRDEYE_API_KEY now SET in Railway — Birdeye OHLCV
  fallback ACTIVE + LIVE-VERIFIED via scripts/verify-ops.sh (/v1/admin/health
  lists birdeye supporting ohlcv_5m/4h; capabilities receipt shows ohlcv_5m
  and ohlcv_4h providers [geckoterminal, birdeye], serving geckoterminal,
  failover_total 0, all four circuits closed, /readyz 200 ok). OHLCV is no
  longer a single-provider path. Also 2026-07-25: /readyz is now the Railway
  deploy healthcheck, codified as config-as-code in railway.json
  (deploy.healthcheckPath=/readyz, commit 91d783a; deploy 20ead19d went Active
  on it, confirming a boot-healthy 200); verify-ops.sh + operational-receipt
  table (primary/fallback/serving/failovers per capability) added (commits
  91d783a, e307d3c).
  swap_metrics capability DEPLOYED + LIVE-VERIFIED (2026-07-18 evening:
  dedicated HELIUS_API_KEY set after initial misconfigured-key incident
  was diagnosed via own telemetry in one probe; SOL served 83 swaps/81
  wallets with truthful truncation via oldest_event_ms; BONK showed
  correctly differentiated m1/m5/m15 windows; cache hit 0ms at 16.7s age)
  — Phase B consumer (poi-engine fail-open shims) is the next build
- **Data/logs:** /data/cache.db, /data/provider_call_receipts.jsonl (Railway
  volume); Railway deploy/HTTP logs
- **Consumers:** poi-engine (production). Future: Momentum Sniper (OHLCV),
  HELIX-JANUS (only after deliberate equities extension), Backpack (dormant)
- **Next milestone:** first OHLCV consumer integration (poi-engine
  synthetic-candle replacement, then Momentum Sniper evaluation)
- **Verified:** 2026-07-21 (66h three-datatype evidence: 46,300+ requests,
  ZERO exhausted; swap_metrics 34,215 served at 78% cache hit with 7,554
  upstream calls at 99.95% success — 1.9 upstream/min matching the design's
  ~2/min prediction; negative cache and stale-serving both exercised in
  production; prior verification 2026-07-17 stands). 2026-07-25: Birdeye
  OHLCV fallback + /readyz Railway healthcheck live-verified (verify-ops.sh
  operational receipt). Only remaining known gap: no external alerting wired.

## 🟢 poi-engine — production

- **Canonical repo:** github.com/ayubeay/poi-engine · commit `c8c05e5`
- **Runtime:** Railway (EU West), https://poi-engine-production.up.railway.app
- **Deployment mode:** auto-deploy on push to main
- **Operational state:** live; poller healthy (20k+ successes, 0 failures at
  audit); adapter healthcheck reports honest "ok" (dead Jupiter v6 probe
  replaced with api-connect /health probe on 2026-07-15)
- **Implemented:** POI/liquidity engine, HeliusOnlyAdapter with api-connect
  first-hop price source (fail-open, env-gated), Phase 2 consumer telemetry
  (api_connect_success/fallback{reason}/latency in /v1/metrics), SSE UI,
  swap poller with backoff, 70 tests
- **Known gaps:** capital_flow / liquidity_stress signals still degraded
  (no liquidation/spot-burst source — unrelated to Helius mediation).
  RESOLVED 2026-07-18: OHLCV via api-connect (synthetic candles retired);
  swap metrics via api-connect (poller 429 rate 99.6% -> ZERO at first
  post-deploy read — 8/8 poller ticks and adapter path served first-hop,
  ~2 upstream calls/min replacing 7.5 failing direct calls/min)
- **Data/logs:** Railway logs; watchlist.json in repo
- **Provider relationships:** consumes api-connect for ALL THREE data
  types (price, OHLCV 5m/4h, swap metrics — d108852); CoinGecko and
  direct Helius retained as fail-open fallbacks only
- **Next milestone:** routine operation. Phase C CLOSED 2026-07-21 (66h
  read): poller 429 rate 99.6% -> 2.7% (30,256 fetches, 97.3% via
  api-connect; 826 fail-open fallbacks absorbed cleanly by preserved
  backoff). Optional Phase D tuning identified with evidence:
  API_CONNECT_SWAP_TIMEOUT_MS 1500->2500 (helius-swap upstream median ~1s,
  tail past 1.5s caused the 823 shim timeouts)
- **Verified:** 2026-07-17 (19h post-OHLCV evidence: 796/796/796
  price/5m/4h api-connect successes, ZERO fallbacks, CoinGecko and
  synthetic candles never invoked; rode out a GeckoTerminal 429 incident
  invisibly on cache)

## 🟢 Momentum Sniper — production (paper doctrine)

- **Canonical repo:** github.com/ayubeay/momentum-sniper-doctrine · commit
  `1eaf3df` (doctrine/observation record; engine code snapshot from initial
  commit era)
- **Runtime:** DigitalOcean droplet 143.110.228.143, /opt/momentum,
  systemd `momentum-sniper.service` (active since 2026-06-01)
- **Deployment mode:** manual on server; doctrine repo tracks observations
- **Operational state:** running continuously; paper-only observation
  infrastructure per doctrine b1181f9. At last gate review (2026-07-13):
  n=6,695 cumulative, criterion 42dd50f RESET (performance gate failed on
  median −8.03%); observation continues under OUTCOME B
- **Implemented:** full scan→score→gate→paper-execute→receipt loop, OROS
  governance, risk layer, Telegram ingestion, POI gate, divergence
  reporting, watchdog
- **Known gaps:** persistent negative median with positive mean (open
  research question); two known error TODO classes (bonding-curve pre-check,
  logger NoneType format); doctrine repo does not mirror current server code
- **Data/logs:** /opt/momentum/momentum_sniper.log (rotated),
  /opt/momentum/data/paper_trades.jsonl
- **Provider relationships:** Helius, DexScreener, Telegram (all direct);
  likely first OHLCV consumer of api-connect
- **Next milestone:** next observation at post-reset criterion terms;
  api-connect OHLCV consumption evaluation
- **Verified:** 2026-07-13 (SSH gate review)

## 🟡 HELIX-JANUS — running local prototype

- **Canonical repo:** github.com/ayubeay/helixjanus · baseline `14c4f14`
  (first version control 2026-07-15)
- **Runtime:** MacBook Air, ~/helixjanus, launchd
  `com.helixjanus.paper.plist` (paper cycles)
- **Deployment mode:** local only; PAPER hard-locked (triple lock:
  LIVE_MODE + !PAPER_MODE + CONFIRM_LIVE_EXECUTION="I_UNDERSTAND"; live
  adapter is a deliberate stub; zero broker credentials exist)
- **Operational state:** actively cycling; at audit 815 receipts, 64 paper
  trades, 1 open USO position; doctrine enforcement observed live (DENY on
  max-positions)
- **Implemented:** full OBSERVE→ADJUDICATE→EXECUTE→SETTLE→RECEIPT→LEARN loop;
  JANUS regime intelligence + DRIFT detector; ALLOW/DENY/ESCALATE
  admissibility; receipts-before-execution; ATR stops; kill switch; risk
  rails (1 position, notional cap, daily loss limit); QQQ 15m mean-reversion
  + USO 4h trend-follow, long-only, yfinance data
- **Known gaps:** no tests; no remote monitoring; single-machine runtime;
  data source is yfinance (api-connect consumption requires deliberate
  equities extension — NOT automatic)
- **Data/logs:** ~/helixjanus/data/ (receipts.jsonl, trades.jsonl,
  state.json, launchd logs) — local only, git-excluded by design
- **Naming note:** this is the TRADING-EXPRESSION of JANUS. It is not
  JANUS-ORA (see below).
- **Next milestone:** test coverage; periodic ops review of paper record
- **Verified:** 2026-07-15 (live state inspection)

## 🟠 Backpack Engine / EHFC Strategy 001 — dormant functional research

- **Canonical repo:** github.com/ayubeay/backpack-engine · recovered
  baseline `a2cceb8`
- **Runtime:** none scheduled. Historical runs: droplet
  /root/backpack-engine (manual, 2026-01-13/14). Original server artifacts
  preserved in place as archive (see docs/PROVENANCE.md)
- **Deployment mode:** manual simulation runs only; Phase Zero research;
  no broker, no capital, mock data only
- **Operational state:** dormant since January; functional (16/16 tests
  pass against recovered engine)
- **Implemented:** EHFC state machine (COLD/WARM/HOT/ENTERED/EXITED),
  eligibility gates, slippage tiers, one-trade-per-event discipline,
  LIQ_NORMALIZE exit, MAE/MFE tracking, seeded batch runner, CSV/JSON
  exports, disabled exit experiments (profit-lock, trailing, extend-hold)
- **Known gaps (doctrine divergences, docs/RECONCILIATION.md):** Flow/ADV
  gate unimplemented; micro caps tradeable vs spec untradeable; optimistic
  slippage numbers; TIME_STOP missing; 3-of-3 vs 2-of-3 HOT trigger;
  no doctrine score
- **Data/logs:** sample_outputs/ in repo; full raw history on server
- **Activation gate:** reserved until MomentumSniper proves live cashflow
  (per momentum RESERVED.md item 9) — currently unmet
- **Next milestone:** none until activation; divergence queue is the first
  work when research resumes
- **Verified:** 2026-07-15 (recovery + tests)

## 🟡 IAM / VERITY stack (iam-integrity) — local prototype

- **Canonical repo:** github.com/ayubeay/iam-integrity (this repo)
- **Runtime:** local + partial (Sonic/SoundKeep pipeline components;
  SURVIVOR agent registered live on SAP mainnet per OOBE docs)
- **Operational state:** mixed — agent minting, scope/ORA contracts, VERITY
  scoring implemented; OROS scope/ORA ENFORCEMENT deliberately reserved
  (docs/RESERVED.md); see docs/VALIDATION_MATRIX.md for claim-by-claim state
- **Known gaps:** enforcement layers unbuilt by design; status of each claim
  tracked in VALIDATION_MATRIX rather than here
- **Next milestone:** per VALIDATION_MATRIX activation conditions
- **Verified:** 2026-07-15 (repo inspection only — runtime claims not
  re-verified this audit)

## 🔴 JANUS-ORA dual-read validator — reserve only

- **Status:** UNCLAIMED. "JANUS validator does not exist"
  (docs/VALIDATION_MATRIX.md). ora_default_v2 hook reserved in ORA schema.
- **Do NOT confuse with HELIX-JANUS** (running trading prototype above).
  Shared name, shared dual-perspective philosophy, zero shared code.
- **Activation:** first adversarial behavior detected (per momentum
  RESERVED.md "Janus #2")
- **Verified:** 2026-07-15

## 🔴 Reserve canon batch — 2026-07-25 (architecture/doctrine, no implementation)

- **Status:** reserve only. 13 new canonical docs written to
  docs/reserve/ this session plus 2 module RESERVE.md updates. No code, no
  runtime, no execution authority — routing/doctrine only.
- **Canonical (iam-integrity/docs/reserve/):** flow-economics-engine,
  survivor-doctrine, opportunity-intelligence-evaluation-engine,
  vyre-vyrel-evolution, domain-aware-capital-intelligence,
  governed-capital-eligibility, helix-universal-execution-lifecycle
  (folds Context Classification + Canonical Execution Doctrine Preservation),
  helix-exchange-layer, helixshield-execution-governance,
  execution-assurance-layer, universal-execution-timeline,
  universal-timeline-semantic-index-engine, ownership-proofs-vs-execution-rights,
  ai-era-moat-doctrine, organizational-separation-doctrine,
  operational-workflow-discovery-engine, meta-architecture-observation-to-moat.
- **Module reserves:** api-connect/RESERVE.md §11 (Enterprise Knowledge
  Integrity); backpack-engine/RESERVE.md §1 (Research Operating System).
- **Promotions:** ai-era-moat-doctrine promotes 2026-07-24 staging #1;
  survivor-doctrine absorbs 2026-07-24 staging #4.
- **Provenance:** docs/reserve/staging/reserves-2026-07-25.md (session ledger).
- **Quarantined, NOT in this repo:** ~11 SoundKeep reserves (separate
  ecosystem) → transport hand-off artifact in session outputs.
- **Verified:** 2026-07-25.

## ⚪ Not yet audited (per doctrine rule 3, cannot be classified)

KONIGO Connect, SoundKeep runtime state, SURVIVOR/OOBE agent operational
state, EventPulse, SportGPT, Earthwise, RACER, HELIXCAN, and other named
systems have NOT had the machines/servers/repos/schedulers check this
registry requires. They remain unclassified until audited. Do not label
them reserve-only or production without that check.

---

## Update doctrine

1. Every milestone commit that changes maturity, deployment, runtime
   location, or integration status MUST update this file in the same
   session.
2. No system may be called **production** without a verified runtime or
   deployment receipt (live probe, log, or receipt cited with date).
3. No system may be called **reserve-only** until local machines, servers,
   repositories, and scheduled processes (launchd, cron, systemd) have been
   checked. The July 15 audit exists because this rule was violated by
   memory.
4. Distinct implementations sharing a name MUST carry explicit qualifiers
   (HELIX-JANUS vs JANUS-ORA is the canonical example).
5. This file records implementation truth, not roadmap ambition. Roadmaps
   live in reserves; truth lives here.
