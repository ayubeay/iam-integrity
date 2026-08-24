# RESERVE — vLOID Anomaly Handling Protocol (AHP)

Status: RESERVED / NOT ACTIVE BUILD.
Parent: vLOID (cross-stack governance primitive).
Captured: 2026-08-23. Origin: procedural abstraction from SCP-style containment discipline — the transferable idea is procedural, not paranormal: when you encounter something you don't understand, classification and action must not outrun evidence.

## What is genuinely new

A governed state for "something happened, we have evidence it happened, but we do not yet know what it means." No existing reserve owns this. It prevents an unexplained observation from becoming a fact merely because an agent, classifier, human, or model assigned a label.

## Governing invariant

An anomaly must never become a fact merely because the system assigned it a label. Preserve explicit epistemic separation — **OBSERVED / INFERRED / HYPOTHESIZED / REPRODUCED / VERIFIED / FALSIFIED / UNRESOLVED** — and never let these silently collapse. **UNKNOWN and UNRESOLVED are legitimate, valid governed states.**

## Lifecycle

    SIGNAL → OBSERVATION → INFORMATION ADMISSIBILITY → KNOWN CLASS?
      ├ YES → normal governed path
      └ NO → ANOMALY_DETECTED → UNKNOWN_CAUSE → CAPABILITY/EXECUTION RESTRICTION
              → EVIDENCE COLLECTION → CONTROLLED TEST/FALSIFICATION → HUMAN+AGENT REVIEW
              → RECLASSIFICATION {VERIFIED | FALSIFIED | KNOWN_CLASS | UNRESOLVED}

"Containment" = **capability restriction**, not shutdown: restrict what an uncertain phenomenon can influence while preserving the ability to investigate it (ALLOW_OBSERVATION / ISOLATE / THROTTLE / SANDBOX / DEFER / ESCALATE / REQUIRE_HUMAN_REVIEW / DENY_EXECUTION / AUTHORIZE_CONTROLLED_TEST — least destructive appropriate to evidence). Scientific discipline: test mundane explanations (instrumentation fault, corrupted data, bug, adversarial input, environmental effect, coincidence, model/human error, ordinary-but-unmodeled behavior) before extraordinary ones; extraordinary interpretations get no privileged treatment. Preserve contradictory evidence rather than deleting it when the working hypothesis changes.

## Relationship (cross-reference, do not duplicate)

Information Admissibility Governor ("should this evidence influence reasoning?") · VERITY (source/evidence trust) · DRIFT (divergence from baseline) · vLOID (what execution stays admissible while unresolved) · OROS (coordinates authorized investigation only) · IAM (who may inspect/test/override) · Shield Router/SURVIVOR (boundaries) · Computable Accountability (observation→evidence→inference→decision→authorization→execution chain, incl. epistemic evolution) · HelixAtlas (anomaly/confidence visualization) · Signal Drift (gameplay reuse of the same machinery). AHP does not swallow VERITY/DRIFT/vLOID — it adds the "admissible evidence, genuinely unusual, cause unknown — what may the system safely do while finding out?" boundary.

## Non-goals

Not an "SCP product"; no fictional/paranormal assumptions; anomalous ≠ dangerous; model confidence ≠ verification; do not auto-escalate every unknown; an AI-generated hypothesis must not overwrite primary evidence; do not duplicate VERITY/DRIFT/vLOID/OROS/IAM/Computable Accountability.

## Activation

Revisit when vLOID hits real states outside its ontology; DRIFT detects unexplained cross-system behavior needing governed investigation; WIRE/robotics introduces sensor/behavior anomalies needing isolation + evidence preservation; scientific/experimental agents require controlled hypothesis testing; or **multiple modules independently begin inventing their own UNKNOWN/quarantine/investigation/reclassification machinery** (strong tripwire that AHP has become a real cross-stack primitive). At activation define the minimal anomaly/evidence state machine first; do not build a large anomaly-management product/UI. Until then: RESERVE ONLY.
