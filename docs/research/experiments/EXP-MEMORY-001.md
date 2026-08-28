# EXP-MEMORY-001 — Memory-Governance Attack Surface

Status: `PROPOSED`
Registered: 2026-08-28
Canonical relationship: Context Integrity · VERITY ·
`docs/reserve/evidence-lifecycle-state-provenance-envelope.md` ·
`docs/reserve/ownership-proofs-vs-execution-rights.md` (Extension 2026-08-28) ·
`docs/reserve/agent-metacognition-calibration-layer.md`

## Hypothesis

**An autonomous system can be compromised upstream of execution by manipulating what it
observes, remembers, correlates, retrieves or believes — even while its model, credentials,
permissions and execution controls remain uncompromised.**

The lifecycle under test:

    WORLD → OBSERVE → RECORD → PROVENANCE → STORE → RETRIEVE → CORRELATE
    → INFER → BELIEVE → AUTHORIZE → EXECUTE → RECEIPT

The portfolio's existing architecture is strongest toward the right of that chain. The
research question is whether the left side requires equivalent admissibility guarantees.

## Design constraint — what must not be touched

**Attack the agent without touching its weights or its executor.** No model modification, no
credential theft, no permission escalation, no execution-governor bypass. If the attack
requires any of those, it is a different experiment and this one has not been run.

Manipulate only: retention · deletion · summarization · provenance annotation · retrieval
ordering · evidence independence · historical context assembly.

Then measure whether **execution changes**.

## Why this is worth running rather than assumed

Published 2026 agent-security work already treats persistent memory, retrieval, context
assembly and provenance as attack surfaces rather than passive inputs, and reports
persistent behavioural compromise without weight modification — poisoned memories later
steering an agent when retrieved, and severe accuracy degradation after poisoning a small
fraction of stored memory. That literature establishes the surface exists. **It does not
establish how this portfolio's governance behaves under it**, which is the only question
this experiment answers.

External findings are cited as prior art, not as results. They are not evidence about our
systems.

## Measurements

Fraction of stored memory that must be manipulated before execution changes · whether the
receipt chain reveals the manipulation after the fact or only the changed outcome · whether
the metacognition/calibration layer registers reduced confidence · whether provenance
weighting alone is sufficient defence · false-suppression rate of legitimate low-trust
evidence under that defence · time from manipulation to first divergent authorization.

## Acceptance / rejection criteria

**Validated** if execution changes materially under memory-state manipulation alone while
every right-hand-side control reports normal.

**Rejected** if existing provenance, evidence-lifecycle and admissibility primitives detect
or neutralize the manipulation without new machinery — in which case those reserves are
strengthened with the evidence and nothing new is created.

**Inconclusive** if the attack succeeds only under manipulation intensities that a real
deployment's retention and provenance controls would already refuse.

## Boundary with the existing canon

If this and `EXP-GENEALOGY-001` both collapse into capabilities already owned by Context
Integrity, VERITY, Evidence Lifecycle, Computable Accountability and Ownership-vs-Execution
Rights, that is the preferred result: those reserves gain evidence and no new canonical
architecture is created. **Only what survives that decomposition earns a reserve.**

## Evidence boundary

Conclusions hold for the memory implementation, retention policy, summarizer and retrieval
ranking used in the run. Nothing here generalizes to a different memory architecture, and no
external study's numbers transfer to this system.

## Provenance

    source artifact:       Information-to-Execution Integrity research family, 2026-08-28;
                           published 2026 agent-memory security literature (prior art only)
    registered:            2026-08-28
    implementation commit: none — no implementation authorized
    evidence boundary:     one memory/retrieval configuration, synthetic corpus
    conclusion date:       pending
