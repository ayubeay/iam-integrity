# RESERVE - VERITY Provenance Resilience / Evidence-Bundle Architecture

**Status:** Reserved. Implement when an activation condition below is met.

## Core principle
**Provenance must not depend on the continued presence of a removable marker. A marker is
evidence; it is not the provenance record.**

Watermarks, metadata, C2PA credentials, statistical text markers, signatures, hashes and
embedded tags are evidence MECHANISMS contributing to an assessment - not the record itself.

Preserve provenance externally through signed receipts and causal records, so that stripping
metadata, re-encoding, screenshotting, paraphrasing or transcoding does not retroactively
erase the evidence chain.

## Target
    artifact created -> actor identity -> source evidence -> hash or fingerprint
    -> model, provider, version -> transformation history -> VERITY assessment
    -> authorization context -> signed external receipt -> receipt lineage

## The semantic rule that matters most
**Absence of evidence is not evidence of the opposite proposition.**

    watermark absent   != human-created
    C2PA missing       != unauthentic
    watermark present  != unquestionably genuine
    metadata intact    != provenance established

Distinguish missing, contradicted, degraded, unverifiable and positively refuted.

Provenance-state vocabulary, indicative: INTACT, TRANSFORMED_BUT_TRACEABLE,
PARTIALLY_ATTESTED, PROVENANCE_DEGRADED, UNRESOLVED, CONTRADICTORY.

## Not a watermark detector
Build a provenance evidence GOVERNOR ingesting multiple evidence types. One removable or
forgeable signal must not become a single point of truth. Adapters - C2PA, EXIF, signatures,
provider attestations, perceptual hashes, timestamp authorities, repository provenance -
stay replaceable through API Connect rather than coupling VERITY to a vendor.

## Doctrine
A provenance signal that disappears has degraded the available evidence; it has not
rewritten history. Provenance should survive conceptually even when the artifact no longer
carries the mechanism that expressed it.

## Activation
When VERITY ingests multimodal artifacts as consequential evidence; when vLOID governs
actions based on AI-generated or transformed media; when receipts need cross-artifact
lineage; or when a use case requires defensible AI-content accountability.
