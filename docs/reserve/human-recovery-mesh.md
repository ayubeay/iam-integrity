# RESERVE - Human Recovery Mesh (HRM)

**Status:** Reserved architecture under WIRE + ShiftTrust + vLOID. Not a separate product.
**Urgency:** LOW to build. The compounding-loop doctrine applies now.

**Capability name:** Human Recovery Mesh.
**Mechanism:** Human-in-the-Loop Execution Recovery.
Naming matters here - neither WIRE nor ShiftTrust should become a teleoperation system.

## The observed reality
Autonomous systems handle the routine 90% and fail on the unpredictable 10%. Commercial
robotics answers this with a human takeover layer: intervene quickly, resolve the edge case,
return control.

The signal is not teleoperation. It is that the exception layer is where the hard cases
live - and therefore where the most valuable data is.

## Governed intervention, not a backdoor
    robot or agent -> anomaly detected -> vLOID admissibility -> operator dispatch
    -> minimum necessary control transferred -> intervention -> recovery criteria met
    -> autonomy resumed -> receipt

A human takeover must be subject to identity, permissions, task safety, location and
context, allowed controls, maximum intervention scope, and a complete record of what the
operator changed. An ungoverned human is not safer than an ungoverned agent; it is only
differently unaccountable.

## Intervention receipt
    execution_id, failure or anomaly type, autonomous_state_before_handoff,
    operator_id, operator_capability, authorization_scope, handoff_timestamp,
    controls_granted, actions_taken, duration, recovery_result, autonomy_resumed,
    policy_version, evidence hash

## Ownership boundaries
    WIRE            learns from demonstrations and recovered edge cases
    ShiftTrust      finds and dispatches qualified humans
    HRM             manages controlled handoff and recovery
    vLOID           governs admissibility of the takeover
    IAM / VERITY    authority and operator trust
    OROS            coordinates
    KONIGO Connect  connectivity continuity during remote control
    DRIFT           detects abnormal behaviour
    receipts        preserve the intervention lineage

## The failure-to-capability compounding loop
Worth preserving as doctrine independently of HRM:

    failure -> governed human recovery -> captured demonstration -> validated skill
    evidence -> model or skill improvement -> reduced future intervention

Not "robot fails, human fixes it." Every rescue becomes evidence, and the same failure
requires less intervention next time.

Important qualification: an intervention should NOT automatically become training data or a
deployable skill. WIRE should preserve the episode, segment the useful action, validate it,
establish provenance and licensing, and measure whether it actually improves performance.
Only then does it graduate toward a reusable skill.

That gives WIRE a stronger position than collecting demonstrations in advance - deployed
robots generate the highest-value examples, being exactly the situations autonomy could not
solve.

## Principle
Mature autonomy is not the elimination of humans. It is the disciplined reduction of
unnecessary human intervention while making necessary intervention safe, authorized,
observable, receipted, and capable of improving future autonomy.

## Grading
Real-time teleoperation involves networking, robotics interfaces, safety, liability and
hardware. Not a near-term build. The doctrine is the part that applies now.
