# RESERVE — Organizational State Transition Governor (OSTG)

Status: RESERVED — research / future architecture. NOT an active build.
Captured: 2026-08-27.
Origin: fragmentation across incorporation, cap tables, corporate documents, hiring,
equity issuance, financing, compliance, data rooms, investor communications and legal
escalation.

## Core observation

Many organizational, legal and compliance problems are not document problems. They are
**state-transition problems**.

Events — incorporation, founder addition/removal, hiring, contractor on/offboarding,
equity issuance, option grants, SAFE/note execution, financing, director changes,
investor onboarding, material contracts, jurisdiction changes, compliance events,
contributor departures, dissolution — each mutate several organizational states at once
and generate new obligations elsewhere.

## Core loop

    intent → current organizational state → event classification
    → jurisdiction / entity context → required approvals → required artifacts
    → dependencies → execution gates → state mutations → generated obligations
    → evidence → receipt → next required actions

"Hire employee" is not a document. The transition may affect identity/access,
employment records, compensation, payroll, equity, confidentiality/IP, tax, permissions
and future obligations. "Founder leaves" may trigger governance, equity treatment,
cap-table mutation, access revocation, IP verification, continuity handoff, unresolved
obligation detection, successor context and a final departure receipt.

## Obligation Propagation Graph (OPG)

    event → state mutation → generated obligation(s) → dependencies
    → responsible party → deadline / condition → completion evidence → receipt
    → new state

An organizational obligation graph rather than a disconnected collection of reminders.

## Legal / professional escalation boundary

Not an autonomous lawyer. Classify transitions:

    SELF-SERVICE · TEMPLATE-SAFE · REVIEW-RECOMMENDED
    PROFESSIONAL-REQUIRED · BLOCKED / INSUFFICIENT EVIDENCE

## GhostLedger / ILF relationship

GhostLedger's existing boundary is valuable and must be preserved: structured matter
intake → classification → evidence organization → coordinated handling → status tracking
→ escalation → professional referral → audit trail. It remains a coordination layer, not
a law firm. ILF provides the professional escalation boundary where independent attorneys
independently determine representation.

**OSTG sits above GhostLedger, not inside it.** GhostLedger becomes one execution
destination when an organizational transition creates a dispute, recovery or
legal-coordination condition. Do not expand GhostLedger into a general business OS, and
do not silently widen ILF's scope.

## Product doctrine

Do not read this as "build Carta + Clerky + banking + payroll + CRM + DocuSign + lawyers."

The stronger architecture is an **organizational control plane** orchestrating existing
systems. Its central question: *what needs to happen because this organizational event
occurred, who or what is authorized to perform each consequence, and can we prove every
required consequence was completed?*

Own the transition intelligence, governance, obligation propagation, orchestration and
receipts. Leave cap tables, banking, payroll, e-signature, accounting, filings, CRM,
data rooms, insurance and legal professionals external.

## Relationship to existing canonical reserves

- vLOID — admissibility of consequential execution.
- OROS — coordination of authorized work.
- IAM — identity, authority, access and revocation consequences.
- VERITY — reliability of evidence, counterparties, documents and assertions.
- `contributor-continuity-handoff-gate.md` — specialized continuity gate on departure.
- `computable-accountability.md` — full transition chain preservation.
- `human-recovery-mesh.md`, `hidden-asset-discovery-engine.md` — adjacent recovery layers.

## Strategic abstraction

    ORGANIZATIONAL EVENT → OSTG → OBLIGATION PROPAGATION GRAPH
    → ADMISSIBILITY / AUTHORITY → EXECUTION ROUTING
    → SPECIALIZED SYSTEMS / PROFESSIONALS → EVIDENCE + RECEIPTS
    → UPDATED ORGANIZATIONAL STATE

An organization becomes machine-legible not as files and accounts but as an evolving
graph of state, authority, obligations, dependencies, evidence, execution and receipts.

## Activation

Reserve only. Revisit when GhostLedger/Ops or ILF repeatedly encounters
organizational-transition workflows; CCHG requires a broader organizational-state model;
vLOID needs explicit organizational-event semantics; multiple products independently need
obligation propagation; real customers repeatedly experience cross-system fragmentation;
or external systems expose APIs making orchestration more economical than rebuilding.

RESERVED. DO NOT BUILD.
