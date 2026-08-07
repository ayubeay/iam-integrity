# Reserve: Rights Continuity & Royalty Recovery Engine

**Status:** Reserved. No implementation.
**Origin:** Pain points reported directly by working musicians.

## Purpose
Help creators reconstruct, verify and recover missing rights registrations and royalty
pathways across publishers, distributors, PROs, neighbouring-rights organisations, labels
and historical catalogues.

Not a replacement for distributors or royalty organisations. The evidence and continuity
layer that reconnects creators with revenue they are disconnected from.

## Problem
Ownership information fragments over time:
labels shut down, publishers change hands, songwriter or performer registrations are
missing, splits are wrong, payment destinations go stale, catalogues are abandoned,
identifiers are absent, ISRC and ISWC relationships break, PRO registrations never happen,
collaborators dispute.

Streaming continues while payment pathways disconnect. The missing component is identity
continuity, not distribution.

## Rights graph
A persistent relationship graph, each edge carrying provenance and confidence.

    COMPOSITION   songwriters, publishers, ownership splits, ISWC, IPI/CAE
    RECORDING     performers, producer, engineer, label, ISRC, distributor, master owner
    ADMIN         PRS, MCPS, PPL, ASCAP, BMI, SESAC, SoundExchange, international societies

## Receipts
Registration, Ownership, Split Verification, Distribution, Publishing, Performance Rights,
Neighbouring Rights, Catalogue Transfer, Rights Recovery.

Each records evidence source, confidence, timestamps, linked identifiers, unresolved
conflicts and execution history.

## Missing-revenue detection
Streaming with no matching songwriter registration. Performer credited but not registered
for neighbouring rights. ISRC with no linked composition. Composition registered but
recording missing. Royalties accruing to an unknown destination. Publisher dissolved.
Label inactive. Ownership conflicts. Duplicate registrations.

## Workflow
    discover work -> collect evidence -> build rights graph -> identify missing links
    -> estimate recoverable paths -> generate recovery plan -> prepare documentation
    -> recovery receipt

## vLOID integration
VERITY evidence verification and conflict scoring. IAM creator identity continuity and
contributor reconciliation. OROS recovery orchestration. Shield Router guards against
fraudulent ownership claims. DRIFT monitors ownership changes and catalogue transfers.
KONIGO Connect maintains linkage after company closures.

## Guiding principle
Music keeps earning long after the people who made it lose contact with the systems that
pay them. SoundKeep preserves the chain of evidence, restores continuity and generates
verifiable recovery receipts rather than replacing royalty organisations.
