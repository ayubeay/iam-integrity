# EXP-COMPUTE-PLACEMENT-001 — Mobile Compute Jurisdiction & Resource Placement

Status: `PROPOSED`
Registered: 2026-08-28
Canonical relationship: `docs/reserve/adaptive-infrastructure-topology.md` ·
`docs/reserve/execution-jurisdiction-gap.md` ·
`docs/reserve/human-fairness-dignity-accountability-institute.md` · KONIGO Connect as
continuity/routing substrate · vLOID as execution-admissibility authority · DRIFT as
changing-condition detector · HelixAtlas as eventual topology visualization.

**Research only. No new canonical reserve, repository, product, deployment commitment or
infrastructure acquisition is authorized.**

## Observation

Floating data centres are real-world evidence that compute infrastructure is becoming less
tightly coupled to conventional terrestrial data-centre geography. One vendor has a 50 MW
floating-data-centre concept with classification approvals and commercialization
partnerships, while simultaneously investigating the failure modes offshore placement
creates — vibration, inclination, salt exposure and humidity effects on servers.

**Treat floating infrastructure as one example of a broader heterogeneous execution
topology, not as a product direction.** Relocating infrastructure does not eliminate
constraints; it transforms the constraint set.

## Hypothesis

As compute becomes deployable across heterogeneous and potentially mobile physical
environments, **execution placement becomes a governed decision** involving not only
performance and cost but jurisdiction, resource externalities, resilience, environmental
conditions, trust and human/community impact.

**Do not assume floating compute is superior to terrestrial compute.** The experiment must
be able to conclude that it is not.

## Phase-Zero design — synthetic nodes only

Construct synthetic nodes rather than acquiring real infrastructure:

    LAND_A   conventional terrestrial DC
    LAND_B   renewable-heavy terrestrial DC
    FLOAT_A  nearshore floating DC
    EDGE_A   community / industrial edge compute
    MOBILE_A relocatable compute platform
    ORBIT_A  hypothetical future orbital compute

Each node carries an evidence envelope:

    compute_capacity · energy_source · energy_availability · cooling_capacity
    freshwater_demand · latency · network_reliability · jurisdiction
    data_residency_classes · physical_location_confidence · environmental_risk
    maintenance_accessibility · physical_security · resource_externality_score
    execution_cost · confidence · evidence_timestamp

Introduce workloads with conflicting constraints. A medical workload may reject the cheapest
node on residency grounds. A latency-sensitive robotic workload may reject an
environmentally preferable node because safe physical execution requires low latency. A
batch AI job may tolerate migration toward temporarily abundant energy. A critical
public-service workload may choose redundancy over both price and environmental
optimization.

The governor outputs `ALLOW` / `DENY` / `DEFER` / `ESCALATE`, plus selected node, rejected
alternatives, decisive constraints, evidence confidence and receipt.

## Adversarial cases — the part that matters most

**Do not design the test so the doctrine wins.** Deliberately construct conflict:

    A  cheapest node consumes substantially more scarce freshwater
    B  lowest-resource-impact node creates unacceptable latency
    C  best technical node is jurisdictionally inadmissible
    D  floating node changes location or regulatory status after placement
    E  environmental telemetry becomes stale
    F  supposedly renewable energy becomes unavailable; node silently falls back to a
       dirtier source
    G  two communities experience unequal resource externalities from infrastructure
       serving users elsewhere
    H  moving the workload itself consumes enough energy and bandwidth that migration is
       worse than remaining in place

Case H is load-bearing: a governor that endlessly chases locally optimal conditions can make
the total system less efficient.

## Invariant under test — no invisible externality optimization

A system must not claim a placement is "efficient," "green," "fair" or "sustainable" merely
because one favourable variable improved. The receipt must preserve:

    benefit gained → cost displaced → affected resource → affected population/environment
    → uncertainty → alternatives considered → authorization → resulting outcome

This is what separates the primitive from an ESG score. The question is not *"are floating
data centres ethical?"* but **"who receives the computational benefit, who bears its physical
cost, what evidence establishes both, and was that distribution admissible?"**

## Acceptance / rejection criteria

Promote beyond research **only** if simulation demonstrates that ordinary cost/performance
scheduling cannot adequately represent decisions involving dynamic jurisdiction + physical
resource externalities + execution admissibility.

If Adaptive Infrastructure Topology and Execution Jurisdiction Gap already handle everything
cleanly, **reject the new primitive and extend those reserves.** That is the outcome that
prevents canon inflation, and it is a success, not a failure.

## Evidence boundary

Conclusions hold for the synthetic node set, workload mix and constraint weights defined in
the run. Vendor engineering claims about offshore deployment are cited as public statements,
not as measured evidence, and are not required for the architectural finding.

## Provenance

    source artifact:       public floating-data-centre engineering and commercialization
                           reporting (origin signal only)
    registered:            2026-08-28
    implementation commit: none — no implementation authorized
    evidence boundary:     Phase-Zero synthetic simulation
    conclusion date:       pending
