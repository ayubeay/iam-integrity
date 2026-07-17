# RESERVE — AI Internet Protocol (AIP)

Status: Reserve Only — Do Not Build Yet
Classification: Long-term Research Direction (protocol layer)
Captured: 2026-07-16

## Vision

Reserve the concept of an AI Internet Protocol (AIP) as a future protocol
layer for autonomous AI systems. The objective is NOT to replace TCP/IP,
HTTP, DNS, OAuth, OpenAPI, or MCP — it is to research whether an additional
protocol layer is needed for trusted AI-to-AI interaction: standardized
mechanisms for agent identity, capability discovery, trust establishment,
policy negotiation, permission delegation, execution governance, receipt
generation, reputation, settlement, and observability.

Honesty note (from capture session): many underlying pieces exist (agent
frameworks, identity systems, MCP, OpenAPI, OAuth), but no widely adopted
protocol combines identity + trust + discovery + negotiation + receipts +
governance into one standard. Treat as research reserve, not novelty claim.

## The missing question

DNS answers "where is this service?" OAuth answers "can I access it?"
OpenAPI answers "how do I call it?" Still missing: **"can I trust this
autonomous system to act on my behalf?"** — a fundamentally different
question.

## Core research questions

Can this autonomous agent be trusted; what actions is it permitted to
perform; how does another agent discover it; how is execution verified;
how are receipts standardized; how is identity preserved across
organizations; how are policies exchanged; how are failures represented.

## Candidate layers (research only)

Identity (public key, owner, model, version, capabilities — no anonymous
production agents); Discovery (searchable by capability: "verified
logistics optimizer, <100ms, SOC2, supports MCP"); Capability
Advertisement; Trust Negotiation (identity check -> signature -> policy ->
history -> risk score); Policy Exchange; Permission Delegation; Execution
Lifecycle (every execution: ID, inputs, outputs, cost, latency, signature,
receipt — receipts as first-class artifacts, not debug logs); Receipt
Specification; Reputation (observed behavior: success rate, verified
executions, failure rate, policy compliance — not stars); Settlement
(agent-to-agent value exchange with receipts); Governance (org policy:
allowed models/countries/costs/vendors/data classes — satisfied before
execution); Observability (live execution graph, not logs); Memory
Continuity (structured memory: projects, customers, policies, receipts,
knowledge, skills).

## Design principles

Vendor-neutral; transport-independent; model-independent; human-auditable;
cryptographically verifiable where appropriate; extensible; backward
compatible with existing internet infrastructure; compatible with multiple
execution environments.

## Relationships

Complementary to HTTP/HTTPS, OAuth, OpenAPI, DNS, TLS, MCP, existing agent
protocols — interoperability, not replacement. HELIX may become ONE
implementation; HELIX must not define the protocol; the protocol remains
implementation-agnostic.

## Reserve doctrine

Investigate whether autonomous AI systems eventually require a common
protocol for identity, trust, governance, and execution beyond today's API
standards. No implementation planned. Companion reserves: Agent DNS /
AI Discovery Layer (how AI systems discover each other), Continuous
Security Receipts (how AI systems prove what they did). Together: how AI
systems communicate (AIP), discover (Agent DNS), prove (CSR).
