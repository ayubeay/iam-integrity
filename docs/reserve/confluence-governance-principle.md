# RESERVE — Confluence Governance Principle (CGP)

Status: RESERVED ARCHITECTURAL DOCTRINE / NOT ACTIVE BUILD.
Parent: cross-stack doctrine (no single module owner; a lens, not a product).
Captured: 2026-08-23. Origin: systems abstraction inspired by the historical role of Idah / the Igala Kingdom at the Niger–Benue confluence. Historical inspiration only; the abstraction is independent of whether every historical claim is verified, and historical provenance classes must not be flattened (oral tradition ≠ archaeology ≠ contemporary record ≠ modern interpretation).

## What is genuinely new

A doctrine for governing the conditions under which independent flows interact, without owning or centralizing them. No existing reserve states this as a principle.

## Core insight & governing principle

Complex systems contain **confluences** — places where independent flows (providers, rails, agents, APIs, marketplaces, data sources, robots, transport, identity domains, execution environments, counterparties) meet and become mutually consequential. The objective is generally not to own every flow but to **govern the conditions under which flows may safely interact**:

> Do not own every flow. Govern the conditions under which flows may safely interact.

Core invariant:

> **Routes may change. Governance must remain interpretable.**

Changing a provider, rail, marketplace, counterparty, agent, or physical route must not silently eliminate the applicable identity, evidence, authorization, admissibility, accountability, or receipt requirements.

## Choke point ≠ confluence

Prefer a **governed confluence** ("multiple legitimate routes, but interaction must satisfy governance") over a **centralized choke point** ("nothing proceeds unless it passes through me"). Prefer many routes + shared governance semantics + independently verifiable receipts over one mandatory route + centralized dependency. Do not convert this doctrine into a universal central gateway — the goal is interoperability under governance, not architectural monopoly.

## Edge discipline (the safeguard)

> **Everything may be connected. Every connection must earn its edge.**

Classify proposed relationships: (1) CONCEPTUAL EDGE (shared abstraction) → (2) ARCHITECTURAL EDGE (one system produces a defined capability/constraint/signal/receipt another consumes) → (3) OPERATIONAL EDGE (evidence it measurably improves execution/resilience/safety/economics/trust). Do not promote conceptual similarity directly into production coupling. Receipts are the primary tissue that lets independent systems compose without surrendering autonomy (one system's consequence may become another's evidence, but provenance must survive the transition). UNKNOWN is a valid confluence state — do not fabricate a route/identity/permission just because the architecture expects one (defer to AHP).

## Relationship (cross-reference, do not duplicate)

Explains relationships among existing systems rather than duplicating them: API Connect (governed provider access) · KONIGO Connect (network continuity/reroute) · Universal Money Router (multi-rail value movement) · IAM · VERITY · Information Admissibility Governor · vLOID · OROS · Shield Router/SURVIVOR · DRIFT (route/assumption change) · Computable Accountability (evidence→…→consequence across boundaries) · HelixAtlas (flows/junctions/edges) · AHP (unknown at the confluence) · Commerce Sniper (marketplace/settlement confluences) · WIRE/robotics (physical confluences).

## Non-goals

No new Confluence product; no universal central gateway; do not duplicate vLOID/IAM/VERITY/OROS/DRIFT/AHP/API Connect/KONIGO/Universal Money Router; do not force every system through one path; do not connect modules on analogy alone; do not treat historical political control as a mandate for modern monopoly or romanticize historical societies as software designers; do not treat disputed history as fact; no dynamic routing without demonstrated operational need; route flexibility must never bypass governance.

## Activation

Reserved until concrete cross-flow governance is required (multiple execution rails must satisfy common admissibility; KONIGO/API Connect must make evidence-based routing across materially different paths; Universal Money Router coordinates multiple settlement rails; agents from different trust domains interact; WIRE introduces human+machine+sensor+network boundaries; Commerce Sniper routes across marketplaces; modules begin independently inventing incompatible boundary governance; HelixAtlas needs a common flow/edge model). At activation, first identify FLOW/ACTOR/BOUNDARY/EVIDENCE/INTENT/AUTHORITY/ADMISSIBILITY/ROUTES/SELECTED ROUTE/EXECUTION/CONSEQUENCE/RECEIPT and check whether an existing module already owns each responsibility; create new architecture only for genuinely unowned ones. Until then: RESERVE ONLY.
