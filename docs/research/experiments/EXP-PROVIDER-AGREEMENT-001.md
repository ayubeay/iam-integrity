# EXP-PROVIDER-AGREEMENT-001 — Do Two Independent Providers Agree?

Status: `PROPOSED`
Registered: 2026-08-29
Migrated from: `docs/research/EXPERIMENT_LEDGER_2026-08-27.md` · Top-5 candidate 4
Related evidence: E6 (OHLCV shadow evaluation) found one provider lagging in a way that
looked like absence rather than delay.

## Hypothesis

Two providers both serving `ohlcv_5m` and `ohlcv_4h`, asked the same question, give the
same answer.

This is a fact about the providers, not about our code — which is why it is an experiment.
Failover assumes providers are interchangeable. That assumption has never been measured.

## Procedure

A **read-only harness outside `api-connect`.** No code change, reserves untouched, parked
state preserved. Capability telemetry already records serving provider and failover count.

Sample both providers for the same instrument and interval over a stated window. Record
divergence in value, in timing, and in availability — three different failure shapes that
a single "agreement" number would blur.

## Accept / reject

**Rejected (providers agree)** if divergence stays within a tolerance stated *before*
sampling.

**Validated (providers diverge)** beyond that tolerance, which would establish that
*available* and *correct* are different capabilities and that readiness cannot conflate
them. Redundancy would then be quorum, not fallback — a materially different architecture.

The tolerance must be fixed before the first sample. Choosing it afterward converts this
into a description of whatever was observed.

## Evidence boundary

One provider pair, one instrument set, one sampling window. Says nothing about other
providers, and nothing about behaviour during the market conditions not present in the
window — which are the conditions where divergence is most likely.

**No interference with third-party infrastructure.** Sampling only; no load generation, no
induced failure.

## Provenance

    source artifact:       EXPERIMENT_LEDGER_2026-08-27.md, Top-5 candidate 4
    registered:            2026-08-29
    implementation commit: none — read-only harness, no api-connect change
    evidence boundary:     one provider pair, one window
    conclusion date:       pending
