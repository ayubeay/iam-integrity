# RESERVE — Operational Sovereignty & Dependency Independence

Status: RESERVED — DO NOT BUILD. Cross-stack doctrine. Not a multi-cloud mandate, not a
redundancy programme, not a procurement policy.
Captured: 2026-08-29.

## Scope boundary — read first

This reserve consumes four existing owners and **must not restate any of them**. If it
ever drifts into general resilience prose, it has stopped being distinct and should be
folded into the reserves below rather than maintained separately.

- `provider-qualification-and-routing.md` — provider evaluation, workload-aware routing,
  jurisdiction as a routing dimension, and the rule that *"a fallback provider does not
  inherit another provider's trust or evidentiary status. Continuity of supply must not
  become continuity of assumed credibility."*
- `sovereign-intelligence-routing.md` — intelligence routing sovereignty.
- `verity-provenance-resilience.md` — *"one removable or forgeable signal must not become
  a single point of truth."*
- `independent-validation-capability-promotion.md` — the Evidence Independence Graph, and
  in particular `SHARES_SOURCE_WITH`, which is the evidence-side analogue of the ancestry
  question asked here.

**What is genuinely new:** PQR is provider-scoped by its own purpose and evaluates
providers within a capability. No reserve asks whether dependencies *across different
classes* share a control ancestor — a question no provider-routing model can see, because
the shared ancestor is not a provider.

## Core invariant

    REDUNDANCY  ≠  SOVEREIGNTY

Redundancy counts alternatives. Sovereignty asks whether the alternatives can fail
independently. Three providers on one substrate are one provider with three invoices.

## Dependency classes

Ancestry must be reasoned about across classes, not within one:

    compute / hosting · network transit · DNS · certificate authority
    identity provider · package registry · model vendor · model lineage
    data source · oracle · payment rail · settlement venue
    jurisdiction · legal entity · key custody · observability

A single upstream — one cloud region, one registrar, one CA, one foundation model
lineage, one clearing venue, one jurisdiction — can appear in several of these
simultaneously without appearing twice in any one of them.

## Common-control ancestry

For any capability the system depends on, the question is not *how many providers serve
it* but *how many independent failure opportunities exist*:

    declared dependency → resolved actual dependency → upstream ancestry
    → shared-ancestor detection across classes → effective independence
    → mission impact if the ancestor fails

Ancestry states, indicative: `INDEPENDENT · SHARED_SUBSTRATE · SHARED_CONTROL ·
SHARED_JURISDICTION · SHARED_LINEAGE · UNRESOLVED`.

**`UNRESOLVED` is a first-class state and must not default to `INDEPENDENT`.** An
unexamined dependency is not an independent one; treating it as independent is the
failure this reserve exists to prevent.

## Mission Independence Envelope

Sovereignty is not absolute and should not be pursued absolutely. The envelope states,
for a given mission: what must keep operating, for how long, under which named failures,
at what degraded level, and what may legitimately stop.

    mission → essential capabilities → tolerable degradation → survivable failures
    → required independent failure opportunities → envelope
    → CONTINUE / DEGRADE / DEFER / SUSPEND when the envelope is breached

Declaring that everything is essential produces no envelope and no decisions. The
envelope's value is in what it deliberately declines to protect.

## Doctrine

**Independence is a property of failure, not of contracts.** Two vendors, two invoices and
two support channels create no independence if one power grid, one registrar, one model
lineage or one legal regime can remove both.

**Sovereignty scales with consequence.** Independence is bought with cost and complexity,
so the burden should rise with what fails when the ancestor does — the same shape as
IVCP's *"independence scales with consequence."*

## Non-goals

Not a mandate to duplicate every dependency · not a multi-cloud requirement · not an
argument for self-hosting · not a claim that local infrastructure is inherently more
sovereign · not a procurement or vendor-selection policy · not a reason to reject a
superior single provider where the envelope tolerates its failure.

## Relationship to existing canonical reserves

`provider-qualification-and-routing.md` (provider-scoped evaluation and routing; supplies
the resolved dependency facts this reserve reasons over) · `sovereign-intelligence-routing.md`
· `verity-provenance-resilience.md` · `independent-validation-capability-promotion.md`
(`SHARES_SOURCE_WITH`) · `execution-jurisdiction-gap.md` (institutional admissibility as a
dependency class) · KONIGO Connect (continuity and failover) · DRIFT (ancestry changing
without notice) · `computable-accountability.md` (receipts recording which ancestry state
was believed at execution).

## Research questions

How is actual ancestry discovered rather than declared, given that providers do not
publish their own upstreams? How often does resolved ancestry change without notice, and
what invalidates a prior independence assessment? Can effective independence be expressed
without collapsing to a single score — the same restraint IVCP applies to evidence
independence? How should an envelope breach interact with obligations already accepted?

## Activation

Revisit when a single upstream failure degrades two or more capabilities believed
independent; when a mission acquires a stated continuity requirement; when jurisdiction
becomes an operational rather than theoretical constraint; or when PQR's routing needs an
independence input it cannot compute from provider observations alone.

## Extension 2026-08-31 — Control, Capability & Operating Obligation

Status: RESERVED — DO NOT BUILD. Interpretive correction to this reserve's own
measurements, not a separate reserve.

Originating intakes: Prismatic (control-acquisition surface) and Production Agent
Operations (production-operation surface). Two observation surfaces, one mechanism,
one placement.

This file measures dependency ancestry: where a capability comes from, whose failure
removes it, and whether the Mission Independence Envelope holds. That measurement is
necessary and it is not sufficient. An operator can reduce external dependency
ancestry — legitimately and verifiably — and still be less able to sustain the mission
afterwards than before, because the capability required to operate what is now
internally controlled was never established.

Subordinate to REDUNDANCY ≠ SOVEREIGNTY:

    CONTROL ACQUISITION ≠ EVIDENCE OF SUFFICIENT CAPABILITY ACQUISITION
    MORE CONTROL ≠ MORE SOVEREIGNTY
    DEPENDENCY INDEPENDENCE ≠ OPERATOR CAPABILITY SUFFICIENCY
    AUTHORIZED TO DEPLOY ≠ CAPABLE OF SUSTAINING PRODUCTION OPERATION

Capability can transfer. Staff, tooling, runbooks, procedures, training, operational
history, observability, automation, support arrangements and institutional knowledge
all move between parties, and an arrangement designed to move them may transfer a
great deal of capability. The claim here is narrower and harder to escape: a change of
control does not by itself establish that the capability the mission requires
transferred, or transferred in sufficient measure. Control is evidence of authority.
It is not evidence of capability.

### Control topology

Control does not move as a single object. Its topology changes. Control may be
partial, phased or layered; technically transferred while institutionally retained
elsewhere; contractually transferred without full technical authority; or split across
code, keys, infrastructure, identity, data, recovery paths and legal rights, each
moving on its own schedule. The mechanism does not require that any of these move
together, and reasoning that assumes they do is the first place this failure hides.

### The obligation set

Obligations do not follow control automatically.

    CONTROL TRANSFER ≠ AUTOMATIC TRANSFER OF EVERY OBLIGATION

An obligation may be inherited, newly created, retained by another party, shared,
terminated, contractually reassigned, technically unavoidable, or UNKNOWN. The
doctrine requires reasoning about the actual obligation set of the arrangement in
front of you. It does not authorise substituting an assumed worst case for that
reasoning:

    UNKNOWN OBLIGATION ≠ NO OBLIGATION
    UNKNOWN OBLIGATION ≠ UNBOUNDED OBLIGATION

UNKNOWN is a permitted and honest output here, on the same terms as elsewhere in this
file: it must not silently resolve in either direction.

### Conceptual shape

Where control topology changes, the reasoning runs:

    control topology changes
    → the actual obligation set is determined
    → the operating capabilities those obligations require are identified
    → available and demonstrated capabilities are compared against them
    → deficits become evidence relevant to the Mission Independence Envelope

This is a conceptual ordering. It is not a state machine, not a workflow, and not a
computation. No ratio, index or grade relates control, capability and obligation. None
should be introduced later.

### A. Control-Acquisition Surface

Originating intake: Prismatic.

The surface is transfer or internalization: an external controller is removed or
replaced, and capability that controller supplied must now exist somewhere else.
Illustrative situations, none of which this file argues against — taking code
in-house, self-hosting, assuming key control, replacing a managed provider,
maintaining a fork, internalizing infrastructure.

Each of these can be the correct decision. The failure is not in making them. It is in
treating the removal of a dependency as proof that the mission is better protected
afterwards.

A prior arrangement may have been supplying capability implicitly — never itemized,
never invoiced separately, and therefore never inventoried when it was removed.
Depending on the arrangement, that may have included patching, incident response,
recovery, security maintenance, compatibility work, staffing, observability, upstream
intelligence or operational knowledge. No arrangement supplies all of these and some
supply none of them; the list is a prompt for inspection, not a description of any
actual counterparty.

The governing question is:

    WHAT CAPABILITIES DID THE PRIOR ARRANGEMENT ACTUALLY SUPPLY,
    AND WHICH OF THEM MUST NOW EXIST ELSEWHERE?

Asked honestly, the answer may be "few" or "none". Left unasked, the answer is
assumed.

### B. Production-Operation Surface

Originating intake: Production Agent Operations.

This surface is not ownership. Nothing changes hands. The transition is temporal:
deployment is an event, production operation is an ongoing condition, and evidence
about the event does not settle the condition.

    AUTHORIZED TO DEPLOY ≠ CAPABLE OF SUSTAINING PRODUCTION OPERATION
    SUCCESSFUL INITIAL EXECUTION ≠ EVIDENCE OF SUSTAINABLE OPERATION

Admissibility resolves at a point before execution; verification confirms that an
execution did what it claimed. Neither speaks to whether the arrangement can keep a
runtime operating as conditions change. Capability questions of that kind —
illustrative, not a checklist and not a gate — include whether the operating
arrangement can diagnose, contain, recover, update, supervise, or safely suspend the
runtime once it is running.

This subsection establishes only that deployment and execution evidence do not
establish continuing operating capability. It defines no paging policy, no
service-level framework, no on-call model, no readiness checklist, and no additional
lifecycle stage.

### Relationship between the two surfaces

Prismatic reaches the gap through a change of control. Production Agent Operations
reaches it through a change of runtime condition. They are recorded separately because
they arise at different moments, are detected differently, and are remedied
differently. They are placed together because they expose the same weakness: acquiring
the right to run something is not acquiring the ability to keep it running.

### Interpretation of the Mission Independence Envelope

This does not create a second envelope. It corrects how the existing one is read.

The envelope is not satisfied merely because the dependencies the mission requires
have independent failure ancestry. Where the mission depends on an internally
controlled capability, the envelope also requires admissible evidence that the
operating capability the mission assumes actually exists at the level assumed. Where
that evidence is absent, the correct record is UNKNOWN — neither sufficiency nor a
declared deficit. No threshold, grade or sufficiency level is defined here, and
capability must not be recorded as demonstrated on the strength of control alone.

Sovereignty theater is the failure mode in which sovereignty appears to improve —
formal control increases, external dependency ancestry decreases — while the
capability required to sustain the mission remains absent, insufficient or unverified.
It names a condition for discussion. It is not a state, a classification to be
assigned, or a module.

### Why this belongs here and not in its own file

This is a correction to the interpretation of measurements this file already owns. The
dependency classes, the ancestry states, the rule that UNRESOLVED must not default to
INDEPENDENT, and the Mission Independence Envelope are all defined here, and the
boundary being added says that those instruments can register improvement while the
mission becomes less survivable. A limit on a measurement belongs beside the
measurement; hosted elsewhere, a reader can acquire the ancestry apparatus without
ever meeting its limit. REDUNDANCY ≠ SOVEREIGNTY and CONTROL ACQUISITION ≠ EVIDENCE OF
SUFFICIENT CAPABILITY ACQUISITION are the same error with different proxies, and the
second belongs next to the first.

This introduces no new product, no new lifecycle, no maintenance platform, no
production-operations system and no self-hosting doctrine. A separate file would have
to re-import this file's dependency classes, ancestry states and envelope before it
could say anything at all, which is the signature of a child that should have been an
extension.

### Evidence boundary

The originating intakes are motivating observations drawn from external cases. They
are the reason this boundary was examined. They are not independent verification of a
universal law, and nothing here is a factual claim about any named company, vendor or
product, nor an assessment of any external system's capability, obligations or
production readiness.

    EXTERNAL CASE → MOTIVATING SIGNAL → CANONICAL MECHANISM
    ≠
    CLAIM THAT THE COUNTERPARTY'S SYSTEM HAS BEEN INDEPENDENTLY VERIFIED

### Open research questions introduced by this extension

- How is sufficient operating capability evidenced, and which evidence classes are
  admissible for it?
- Which obligations actually move when control topology changes, and which are created
  rather than transferred?
- Which capabilities was a dependency supplying implicitly, and how are they identified
  before removal rather than after?
- How should capability sufficiency interact with the Mission Independence Envelope
  without either collapsing into a single measure?
- When is continued external dependence operationally safer than premature
  internalization?

The last question must remain answerable in the counterintuitive direction:

    A DEPENDENCY-EXPOSED ARRANGEMENT
    MAY BE MORE MISSION-SURVIVABLE
    THAN AN INDEPENDENT-BUT-INCAPABLE ARRANGEMENT.

That is a possible comparative outcome for a particular arrangement. It is not a
general preference for dependence.

RESERVED — DO NOT BUILD.
