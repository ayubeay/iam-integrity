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
