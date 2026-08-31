# RESERVE - Invariant Precomputation & Verified Reuse Doctrine

**Status:** Cross-system optimisation doctrine. Not a product.
**Urgency:** LOW until profiling justifies it. The DISCIPLINE is free.

## Principle
Before repeatedly evaluating a large search, simulation, context or decision space, identify
expensive computations whose inputs remain invariant across evaluations. Compute those once,
version and hash them, cache the reusable representation, and recompute only state whose
dependencies changed.

Origin: a quantum-simulation workload that rebuilt and re-diagonalised Hamiltonians for every
point of an 870-point parameter sweep. Precomputing the invariant structures and reusing the
eigendecompositions took it from roughly two hours to 47.6 seconds.

The lesson is not CUDA. It is: **do not accelerate repeated work before asking why the work
is being repeated.**

## Where it applies here
SURVIVOR evaluating many proposed actions should not recompute identity, permissions, policy
bundle, program classification, counterparty verification, strategy configuration and market
metadata from zero every time. Those are slow-changing. Price, position, exposure, regime and
the requested action are not.

Context Integrity has the same shape - context engineering should not mean rebuilding an
enormous window every turn. Stable authoritative knowledge becomes versioned retrievable
representations; volatile information refreshes.

## State classes
    invariant          rarely or never changes
    slow-changing      changes on policy or configuration updates
    freshness-critical must be current or the decision is invalid
    per-execution      computed every time

## Correctness constraint
Reuse must never silently sacrifice freshness. Every reusable artifact carries dependency
information, version and provenance, creation time, invalidation conditions, and enough
identity to determine exactly which computation was reused.

**Execution receipts must be able to reference reused artifacts**, so an evaluation stays
reproducible. Otherwise the optimisation destroys the auditability it was meant to preserve.

Invalidate on: upstream dependency change, authoritative source change, policy or config
version change, security state change, freshness boundary crossed.

## Do not
Introduce GPU or CUDA complexity by default. Profile first; apply vectorisation, batching and
caching where sufficient; reach for custom kernels only where measured workloads justify it.
Our systems should not become GPU projects because CUDA produced an impressive result
somewhere else.

## Future
HelixAtlas could visualise reusable computation nodes and their invalidation relationships,
making it visible what was recomputed versus safely reused.

---

## Extension 2026-08-29 — Cognitive Artifacts as a Reuse Class

Status: RESERVED — DO NOT BUILD. Architectural refinement of this reserve.

### Why this belongs here and not in its own file

A proposal arrived for a separate "cognitive object store." This reserve already owns
verified reuse: the state classes, the correctness constraint that *reuse must never
silently sacrifice freshness*, invalidation conditions, and the requirement that execution
receipts reference reused artifacts. It already names context engineering as the same
shape. A separate store would duplicate that machinery for one artifact type, so what
follows is a state-class refinement rather than a new mechanism.

### Cognitive artifacts

Artifacts an agent produces by reasoning rather than by computation: derived plans,
intermediate conclusions, summaries of retrieved material, judged evidence, tool-output
interpretations, task decompositions, and assessments of another agent's output.

### The distinction that matters

This reserve's origin is numerical — eigendecompositions reused across a parameter sweep,
where reuse is **exact** and invalidation is **observable** from dependency change.

**Reusing a cognitive artifact reuses a judgment, not a computation.** A judgment carries
the assumptions, uncertainty, context window and evidence state present when it was made,
and none of those appear in a dependency graph. A cached conclusion can be stale in a way
a cached matrix cannot: its inputs are unchanged while its *warrant* has expired.

    invariant artifact    same inputs → same output, and staleness is detectable
    cognitive artifact    same inputs → same output, and staleness may not be

### Additional invalidation conditions

Beyond the triggers already listed, a cognitive artifact should invalidate on: objective
or mission change · admissibility or policy change affecting what the judgment may
influence · arrival of contradicting evidence · expiry of the evidence the judgment rested
on · model or method change that would alter the judgment · and the artifact's own stated
confidence falling below the threshold its consumer requires.

Where warrant cannot be established, the correct state is **recompute**, not reuse. The
existing constraint governs: reuse must never silently sacrifice freshness — and for a
judgment, freshness means the reasoning would still be made the same way, not that the
bytes are current.

### Provenance requirement

A reused cognitive artifact carries, in addition to this reserve's existing fields: what
question it answered · what evidence it rested on · what uncertainty it carried · what it
is admissible to influence. Without the last, a conclusion reached for one purpose is
silently reused to authorize another — which is the failure
`ownership-proofs-vs-execution-rights.md` names for knowledge, applied to cached
reasoning.

RESERVED — DO NOT BUILD.
