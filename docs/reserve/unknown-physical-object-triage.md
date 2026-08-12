# RESERVE - Unknown Physical Object Triage (UPOT)

**Status:** Reserved future capability. Not an active build.
**Parent:** vLOID physical-world execution intelligence.

## Purpose
When a human, robot, drone, camera or agent encounters an object whose identity,
provenance, ownership, condition or hazard level is uncertain, reduce the dangerous
interval between encountering it and knowing what to do.

    unknown object -> observation -> evidence -> preliminary classification
    -> safety posture -> specialist escalation -> verified disposition

The purpose is NOT to declare from an image that an object is safe, radioactive, explosive
or toxic.

## Core principle
**Unknown must remain a legitimate classification.** The system should prefer "I do not
have sufficient evidence to determine this" over false certainty. Visual AI may generate
hypotheses; it must never substitute for physical measurement, authoritative records or
qualified personnel.

## Evidence states, kept distinct
    OBSERVED    what sensors or humans actually detected
    INFERRED    what models or rules concluded
    LOOKED_UP   what external records report
    MEASURED    what appropriate instrumentation detected
    VERIFIED    what authoritative evidence or qualified personnel established
    UNKNOWN     what remains unresolved

A model inference must never silently become a verified fact.

## Triage postures
SAFE_TO_CONTINUE, INSUFFICIENT_EVIDENCE, DO_NOT_HANDLE, ISOLATE_AREA, REQUEST_SPECIALIST,
REQUEST_INSTRUMENT, ESCALATE_TO_AUTHORITY, KNOWN_ASSET, POTENTIAL_HAZARD, CONFIRMED_HAZARD.
Terminology to be designed with domain experts.

## Non-goals
Not an AI bomb detector, radiation detector, HazMat replacement, emergency-services
replacement, or permission for an agent to manipulate an unknown object. Never declares
safety from visual recognition alone.

## Why it belongs here
Most computer vision asks "what is this?" As agents operate in the physical world they also
need a governed answer to "I don't reliably know what this is, so what actions are
admissible now?"

That is execution governance, not object recognition:

    perception -> evidence -> uncertainty -> verification -> admissibility -> safe action
    -> escalation -> receipt

## Build posture
Reserve only. When a credible deployment appears, first decide whether UPOT is a vLOID
capability, an API Connect capability family, a service consumed by other products, or a
domain-specific implementation over a common protocol. Reuse before creating.
