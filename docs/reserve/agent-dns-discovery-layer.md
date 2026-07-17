# RESERVE — Agent DNS / AI Discovery Layer

Status: Reserve Only — Do Not Build Yet
Classification: Future Internet Infrastructure Research
Captured: 2026-07-16

## Vision

Reserve the concept of an Agent DNS / AI Discovery Layer enabling AI
services to publish standardized machine-readable metadata describing their
identity, capabilities, trust posture, and interaction requirements. NOT a
DNS replacement, and NOT a plan to operate a TLD — owning ".ai" is a naming
business; this is the identity-and-discovery layer that works on top of
existing domains (the more realistic and more valuable play).

## The manifest concept

Websites expose robots.txt / sitemap.xml / openapi.json. AI services could
expose `agent.manifest.json`:

    {
      "service": "SoundKeep",
      "agents": ["sonic", "playlist", "music-discovery"],
      "identity": { "verified": true, "owner": "SoundKeep Inc." },
      "permissions": ["spotify.read", "youtube.read"],
      "receipts": true,
      "pricing": "subscription",
      "contact": "api@soundkeep.ai"
    }

Any AI system immediately understands how to interact with the service.
Exact format is an open research question.

## Research goals

Mechanisms for publishing: agent identity, capabilities, supported
protocols, authentication methods, public keys, trust metadata, governance
policies, version info, pricing, rate limits, receipt support, contact
endpoints. Discovery records could extend DNS conventions (standardized TXT
records or future record types — AGENT / MODEL / TRUST / RECEIPTS / POLICY)
without assuming new record types are required.

## AI Passport (per-agent)

Agent ID, name, owner, domain, public key, signature, capabilities, trust
history, execution history, policy version, model hash — letting clients
answer: who built this agent, is it authentic, what is it allowed to do,
has it behaved reliably.

## Domain reputation

Beyond SSL: trust level, receipt volume, failed-execution rate, average
response, policy violations, verified models — evaluable BEFORE
integration. Verifiable rather than self-asserted wherever possible.

## Discovery engine

Search engines index pages; tomorrow they may index AI services by
capability, trust, and governance ("find a verified medical coding agent
with SOC2"), not keywords.

## Relationships

Compatibility research: DNS, DNSSEC, HTTPS, OpenAPI, MCP, OAuth, existing
service discovery. Agent DNS may become one component of the broader AI
Internet Protocol (docs/reserve/ai-internet-protocol.md):
Internet -> DNS -> Agent Discovery -> Identity -> Trust -> Execution ->
Receipts -> Governance -> Settlement. HELIX may consume or publish this
metadata, but the specification remains ecosystem-neutral. IAM/VERITY/
SURVIVOR/receipts/HelixAtlas already touch these concepts internally.

Follow-on (staged separately): AI Capability Registry — an application
built on top of Agent DNS/AIP that indexes AI services by capability
across providers; kept separate because it is an application, not part of
the protocol.

## Reserve doctrine

Explore standardized AI service discovery rather than proprietary product
features. Prioritize interoperability, openness, and compatibility with
existing internet infrastructure.
