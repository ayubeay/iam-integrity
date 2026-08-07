# RESERVE - Execution Adapter Framework (EAF)

**Status:** Reserved. Deferred until ecosystem products show sustained traction.
**Urgency:** LOW as a framework. MEDIUM as a discipline applied per-integration.
**Classification:** Core infrastructure reserve.

## Summary
A provider-independent integration layer standardising how the ecosystem talks to external
systems. Rather than applications integrating directly with vendors, every external
platform connects through a governed adapter implementing a common execution contract.

Applications express intent. Adapters translate intent into provider-specific execution.
Applications depend on stable contracts; adapters depend on vendors.

    application -> governance -> execution contract -> adapter -> provider

## Why it deserves its own reserve
The pattern recurs across nearly every project:

    Momentum Sniper                    broker adapters
    Enterprise Execution Control Plane AI model adapters
    Universal Money Router             payment rail adapters
    ShiftTrust                         HR, scheduling, payroll adapters
    WIRE                               robotics platform adapters
    KONIGO Connect                     network and infrastructure adapters
    Commerce Sniper                    marketplace and supplier adapters
    SURVIVOR / API Connect             RPC and archive providers

Without the doctrine stated, each product invents its own coupling and each one has to be
rewritten when a vendor changes.

## Adapter categories
AI, payments, identity, communications, storage, workforce, commerce, trading, robotics.
Intentionally extensible.

## Standard contract
    request -> validate -> translate -> execute -> normalise response -> receipt -> return

Applications never need provider-specific logic.

## Adapter responsibilities
Protocol translation, authentication, authorisation, rate limiting, retries, failure
handling, timeouts, observability, telemetry, receipt generation, version compatibility,
capability discovery.

## Governance
Execution stays governed by Stewardship, PRAETOR, vLOID, VERITY, IAM, DRIFT, OROS, Shield
Router and SURVIVOR. Adapters execute only approved requests and never bypass governance.

## Receipts
Normalised regardless of provider: provider, adapter version, execution time, latency,
request ID, result, status, evidence, timestamp, signature.

## Non-goals
Not an API gateway, not an ESB, not a workflow engine, not a replacement for governance.
It is the translation layer between governed execution and external systems.

## Grading and what to do now
Building the framework before there are several live integrations would be architecture
ahead of evidence - the same error the SURVIVOR scoring work kept finding.

But the DISCIPLINE is cheap and should apply from the next integration onward:

    keep provider-specific code behind one module per provider
    never let a vendor's response shape leak into business logic
    normalise responses at the boundary
    emit a receipt at the boundary, not inside the caller

Two current cases already show the cost of not doing this: the Base x402 rail hardcodes a
facilitator URL in application code, and the holder query embeds RPC-specific error strings
in the fetcher. Neither is a crisis; both would have been cleaner behind an adapter.

Extract the framework when a third provider appears in the same category. Not before.
