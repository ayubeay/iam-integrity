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

- **Canonical repo:** github.com/ayubeay/api-connect · commit `f313d15`
- **Runtime:** Railway (EU West), https://api-connect-production-b1a7.up.railway.app
- **Deployment mode:** auto-deploy on push to main; persistent volume
  `api-connect-data` mounted at /data
- **Operational state:** live; canary consumer (poi-engine) passed 2-day
  validation (2,260 requests, 71% cache hit rate, 0 consumer failures)
- **Implemented:** cache-first provider router, helius/jupiter-v3 provider
  (token_price), GeckoTerminal provider (normalized crypto OHLCV, 5m/4h,
  pool-resolved so arbitrary mints work — added 2026-07-16), sqlite cache on
  volume, execution receipts (JSONL on volume), Prometheus telemetry at
  /v1/metrics (per-provider latency, classified error distributions, 429
  pressure, circuit state), provider health circuit breaker, graceful
  shutdown, admin endpoints, 15 tests
- **Known gaps:** one provider per data type (no OHLCV fallback provider
  yet); intervals deliberately narrow (5m/4h only); no alerting on metrics;
  equities/ETF asset class not supported (deliberate — see HELIX-JANUS note).
  swap_metrics capability BUILT (Phase A, 2026-07-18: HeliusSwapProvider
  with windows+events_60s, per-key negative cache honoring Retry-After,
  TTL 30s) — awaiting deploy + HELIUS_API_KEY env + Phase B consumer
- **Data/logs:** /data/cache.db, /data/provider_call_receipts.jsonl (Railway
  volume); Railway deploy/HTTP logs
- **Consumers:** poi-engine (production). Future: Momentum Sniper (OHLCV),
  HELIX-JANUS (only after deliberate equities extension), Backpack (dormant)
- **Next milestone:** first OHLCV consumer integration (poi-engine
  synthetic-candle replacement, then Momentum Sniper evaluation)
- **Verified:** 2026-07-17 (19h production evidence: 2,400+ requests across
  3 data types, cache hit rates 71-92%; survived a real GeckoTerminal
  429 incident — circuit breaker opened, 46 skips, 56 exhausted served —
  while consumer saw zero fallbacks. Failed Railway build d82e2d59 was the
  docs-only BCMS commit during the GitHub outage; production runs b2da2f1)

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
- **Known gaps:** OHLCV still direct CoinGecko (429s force synthetic
  candles — primary motivation for api-connect OHLCV); capital_flow /
  liquidity_stress signals degraded by Helius plan quota
- **Data/logs:** Railway logs; watchlist.json in repo
- **Provider relationships:** consumes api-connect (price); CoinGecko
  (OHLCV, direct); Helius (swap metrics, direct)
- **Next milestone:** swap-metrics mediation via api-connect (poller Helius
  429 rate measured at 99.6% — 515/517 fetches — on 2026-07-17)
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
