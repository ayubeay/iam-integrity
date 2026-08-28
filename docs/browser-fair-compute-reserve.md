# BROWSER FAIR COMPUTE — Experimental R&D Reserve

**Status:** RESERVED — research and benchmarking only  
**Reserved:** July 19, 2026  
**Working thesis:** Explore whether useful computational work can be designed so ordinary consumer devices remain competitive with specialized GPU farms.

## Origin

BrowserCoin’s Sandglass proposal raised a broader systems question:

> Can an algorithm be designed around hardware characteristics that ordinary users already possess, rather than automatically rewarding whoever owns the largest specialized compute farm?

The immediate interest is not launching or mining a token. The interest is independently testing the engineering principle and determining whether it generalizes beyond cryptocurrency.

## Core research question

Can browser-based WebAssembly execution remain meaningfully competitive with:

- optimized native CPU implementations,
- consumer GPUs,
- data-center GPUs,
- large parallel compute farms,

when the workload is deliberately constrained by dependent memory latency rather than raw parallel throughput or memory bandwidth?

## Phase 0 — Rules of engagement

This begins as an isolated benchmark project.

It must not:

- launch a token,
- solicit investment,
- use visitors’ hardware without explicit informed consent,
- hide computation inside another product,
- make profitability claims,
- consume user electricity without visible controls,
- connect to the vLOID production environment,
- distract from active revenue and product priorities.

Every experimental run must be deliberate, local, measurable, and reversible.

## Phase 1 — Independent reproduction

Reproduce comparable implementations of one deterministic workload in:

1. Browser JavaScript/WebAssembly
2. Native Rust
3. Native C where practical
4. GPU implementation only if hardware and time permit

Measure:

- operations or hashes per second,
- median and tail latency,
- CPU and GPU utilization,
- memory footprint,
- power draw where observable,
- thermal throttling,
- performance per watt,
- output determinism across implementations.

The goal is not to confirm the original project’s claims. The goal is to try to falsify them.

## Phase 2 — Hardware fairness analysis

Test across a small but diverse device set:

- older consumer laptop,
- modern consumer laptop,
- desktop CPU,
- mobile browser where supported,
- consumer GPU,
- rented data-center GPU only if economically justified.

Produce a fairness profile showing:

- absolute performance,
- performance per dollar,
- performance per watt,
- GPU advantage ratio,
- browser-to-native efficiency ratio,
- concentration risk if one participant runs many parallel workers.

## Phase 3 — Adversarial evaluation

Attempt to break the fairness claim.

Test:

- many parallel browser workers,
- many native CPU threads,
- cache-size advantages,
- large-core-count servers,
- GPU occupancy optimizations,
- FPGA or ASIC plausibility at a conceptual level,
- botnets and hidden-browser abuse,
- thermal and battery effects on ordinary devices,
- Sybil behavior from one operator pretending to be many users.

A workload is not “fair” merely because one browser tab performs well. The system must also resist industrial replication of thousands of equivalent workers.

## Phase 4 — Generalization beyond mining

Only after benchmarking, evaluate whether the design pattern applies to useful work such as:

- decentralized verification,
- proof generation,
- integrity challenges,
- volunteer scientific computing,
- distributed testing,
- AI-output verification,
- execution-receipt validation,
- small deterministic agent tasks.

The preferred direction is useful verified computation, not speculative mining.

## Potential vLOID relationship

This experiment may eventually inform a compute-verification layer for autonomous systems.

Possible future mapping:

- **IAM** — identify the device or worker performing the task
- **VERITY** — score reliability from completed challenges
- **LITMUS** — define acceptable use and user-consent doctrine
- **OROS** — assign, adjudicate, and finalize computation
- **DRIFT** — detect runtime or implementation divergence
- **VYRE** — sign challenge inputs and execution receipts
- **Shield Router** — reject invalid, duplicated, or policy-violating work

This is a possible future relationship, not a current integration commitment.

## Minimum experiment artifact

The first useful deliverable is a public-neutral benchmark report containing:

- workload specification,
- implementation hashes,
- hardware specifications,
- repeatable commands,
- raw benchmark results,
- methodology,
- failed experiments,
- limitations,
- conclusions that clearly separate evidence from inference.

## Success criteria

The research is worth continuing only if:

1. Browser performance is consistently close to native CPU performance.
2. Specialized GPUs have a limited real-world advantage after power and cost are considered.
3. Results remain deterministic across implementations.
4. The system cannot be trivially dominated through parallel replication.
5. A useful non-speculative workload can be identified.
6. The work teaches transferable systems knowledge even if the fairness thesis fails.

## Failure is a valid result

If GPUs, many-core servers, custom hardware, or parallel farms regain a decisive advantage, document it and stop.

A negative result still creates value by showing where browser-fair compute fails and preventing a larger misallocation of time.

## Initial build queue

1. Write a minimal deterministic dependent-memory workload in Rust.
2. Compile the same implementation to WebAssembly.
3. Create a browser benchmark page with explicit Start and Stop controls.
4. Export JSON results containing hardware, runtime, latency, throughput, and implementation hash.
5. Run the first same-machine browser-versus-native comparison.
6. Decide whether additional hardware testing is justified.

## Product doctrine

Ordinary people must never become invisible infrastructure.

Any future system based on this research must provide:

- informed opt-in,
- visible resource usage,
- immediate stop controls,
- battery and thermal limits,
- transparent reward calculation,
- verifiable receipts,
- no hidden mining,
- no background execution without consent.

## Final framing

This reserve is not “build another cryptocurrency.”

It is:

> Investigate whether computational markets can be engineered so ordinary devices remain credible participants, and whether that principle can support useful, consent-based, verifiable work.


---

## Scope extension — 2026-08-27: Fair Useful Compute

This reserve's scope widens from the original browser-vs-CPU fairness question to the
broader **Fair Useful Compute** research programme. The original framing remains valid and
is not withdrawn; it becomes Phase 1 of a larger question.

### Evolved research question

Can a distributed compute protocol be designed so that ordinary consumer hardware remains
**economically meaningful** — not merely technically permitted to participate — while
preserving correctness, security, useful throughput and verifiable execution?

The target is not to prove CPUs always beat GPUs or that specialized hardware can be
eliminated. It is to determine whether workload and protocol design can bound the degree
to which capital-intensive parallel hardware translates into disproportionate reward or
network control.

Do not frame this primarily as "CPU mining." Bitcoin was the motivating observation; the
more valuable question covers AI inference, scientific computation, rendering, simulation,
agent workloads and verification jobs.

    ordinary hardware → joins compute network → receives useful/verifiable work
    → executes → result + execution evidence → verified → contribution measured
    → reward allocated → concentration/fairness measured

### Eligibility vs economic participation

A network can support laptops and ordinary CPUs while still centralizing economically.
Measure separately:

- **Technical accessibility** — can ordinary hardware execute valid work?
- **Economic relevance** — does it earn a meaningful reward share after throughput,
  energy, hardware cost, uptime, networking, memory and utilization?
- **Control concentration** — can a few capital-rich operators dominate despite
  permissionless enrollment?

**A protocol is not "democratized" merely because anyone can install its client.**

### Phase status

- **Phase 1 — COMPLETE / BANKED.** Deterministic dependent-memory workload across native
  Rust, Python reference, JavaScript and browser WASM; known-answer testing;
  implementation hashing; executed-step accounting; comparison tooling. Evidence:
  `bench/fair-compute-bench/results/report/comparison.md`; ledger entry E8.
- **Phase 2A — COMPLETE / BANKED.** Metal implementation passing the determinism gate
  before performance comparison; non-monotonic GPU/CPU throughput across per-worker memory
  with a vulnerability window at intermediate sizes and reversal at large scratchpads.
  Evidence: `bench/fair-compute-bench-gpu/results/report/phase2.md`; ledger entry E9.
  Correct interpretation: *evidence consistent with the hypothesis on the tested Apple
  Silicon configuration; it does not establish universal CPU/GPU fairness.*
- **Phase 2B — RESERVED.** Cross-architecture accelerator test (CUDA, consumer and
  datacenter GPUs, AMD, newer Apple Silicon, different host CPUs). Does a comparable
  transition region appear on materially different hardware? Blocked on capital.
- **Phase 2C — RESERVED.** Parameter and memory-budget surface — map a hardware-advantage
  surface rather than benchmarking one arbitrary configuration.
- **Phase 3 — RESERVED.** Participant economics and Sybil simulation. Heterogeneous
  participants; reward distribution, cost, ROI, concentration, Gini, top-1/top-10 share,
  participation survival, attack cost. *Can one operator convert capital into many
  identities or much hardware and recover a disproportionate share?*
- **Phase 4 — RESERVED.** Distributed validation across real heterogeneous nodes:
  assignment, verification, retries, latency, uptime, malicious-result handling, receipts.
- **Phase 5 — RESERVED.** Useful-compute pilot. Do not create a token or blockchain to
  demonstrate the concept; first prove useful work can be assigned, verified, rewarded and
  kept reasonably decentralized.

### External-network comparison track

Treat decentralized-inference networks as external research cases, not as evidence that
our hypothesis is validated. Inspect job assignment, work concentration, verification,
reward calculation, latency/uptime and batching advantages, hardware differences, Sybil
resistance, spoofed-compute prevention, disagreement handling, failed or malicious work,
bandwidth effects, contribution measurement, and whether ordinary devices remain
economically relevant after optimized operators enter.

### Fairness metrics

Never reduce fairness to throughput. Hardware advantage ratio · reward concentration ·
Gini · top-X% share · cost- and energy-adjusted contribution · minimum viable participant
hardware · specialization advantage · Sybil amplification · cloud-rental advantage ·
verification cost · useful-work efficiency · participant retention.

Candidate metric — **Consumer Participation Ratio**: the share of useful network work or
reward sustainably contributed by commodity devices under realistic costs. Its definition
and threshold must be experimentally justified, never chosen to make the protocol look
fair.

### ASIC-resistance doctrine

Avoid claiming permanent ASIC resistance. The better objective is **bounded specialization
advantage**: how much economic advantage can specialized hardware obtain, and does it
become large enough to centralize participation? Specialized hardware existing is not
failure; an overwhelming multiplier may be.

### Doctrine

**Permissionless participation is not the same as meaningful participation.**

And the methodological lesson from Phases 1 and 2A:
**correctness first, measurement second, economics third, claims last.** Do not strengthen
a claim because the result is commercially attractive. Every phase must be capable of
falsifying the fairness hypothesis.

### Activation

Resume when an external protocol provides something concrete to test; CUDA or other
hardware access becomes inexpensive; a hackathon or research opportunity makes it
strategically useful; one of our products needs distributed compute; a use case needs fair
heterogeneous scheduling; the evidence could support a publication, partnership or grant;
or capacity exists without displacing higher-priority execution.

Experiment definitions: `docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md`.

**Phases 2B–5 RESERVED. No new implementation without an explicit activation decision.**
