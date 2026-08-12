# RESERVE - Evidence-Governed Runtime Doctrine

**Status:** Reserved engineering doctrine. No migration or rewrite commitment.

## Principle
Runtime and language choices must be evidence-governed rather than ideology-driven. Do not
migrate a service from Node, Python or anything else because Go or Rust benchmarks better in
the abstract.

A runtime change needs measured production evidence that the current runtime is materially
responsible for unacceptable latency, throughput limits, memory, concurrency constraints,
infrastructure cost, reliability problems or operational complexity.

## Sequence
    correctness -> instrumentation -> production measurement -> bottleneck attribution
    -> targeted optimization -> selective migration only when justified

Telemetry must separate runtime overhead from network latency, database access, RPC
behaviour, third-party APIs, sequential dependency chains, serialisation, inference, scoring
computation, retries and queueing.

The mint-path investigation is the live example: 4.9 seconds inside fetchTokenData is a
sequencing question, not a language question, and no rewrite would have found that.

## Mixed runtimes are acceptable
Python for research, scoring and data-heavy work. TypeScript and Node where developer speed
and ecosystem dominate. Go or Rust for high-concurrency, latency-sensitive, network-heavy or
memory-sensitive paths.

Do not require whole-system rewrites. Migrate the smallest constrained surface that
materially improves the measured bottleneck.

## Migration receipt
    service -> observed bottleneck -> measurement period -> attributable runtime cost
    -> alternatives tested -> expected improvement -> migration scope -> before and after
    -> rollback path

A rewrite is justified only when evidence shows changing the runtime is likely to outperform
simpler architectural or operational improvements.

## Core principle
Do not ask which language is fastest. Ask what is actually limiting the system, prove it,
and change only what the evidence says is limiting it.
