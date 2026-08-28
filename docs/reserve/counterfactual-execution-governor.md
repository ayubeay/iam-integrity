# RESERVE — Counterfactual Execution Governor (CEG)

Status: RESERVED — research / future architecture. NOT an active build.
Parent: vLOID (pre-execution consequence layer; not a competing governor).
Captured: 2026-08-27.
Origin: scientific examination of the "precrime" thesis, reformulated as prediction
of unsafe *actions and system states* rather than human criminality. Embodied branch
added the same session from robotic-compromise analysis.

## Core research question

Can observable precursor states predict a dangerous or policy-violating agent
execution before it occurs, early enough to intervene, at an acceptable
false-intervention rate?

The system must never claim "this person will commit a crime." Its domain is:
"given currently observable evidence, this proposed action has some probability of
producing specified adverse consequences."

Primary experimental subject is software agents, because their proposed actions,
permissions, context, tools, environment and policies are directly observable.

## Governing principle

**Predict hazardous actions and consequences, not future guilt.**

## Core loop

    observed state
    → proposed action
    → precursor-signal extraction
    → evidence quality assessment
    → counterfactual futures
    → probability × severity × reversibility
    → policy / admissibility evaluation
    → ALLOW / ALLOW+MONITOR / THROTTLE / REQUEST EVIDENCE / CHALLENGE /
      SANDBOX / DEFER / ESCALATE / DENY
    → execution or non-execution
    → actual outcome
    → receipt
    → calibration

The essential property is that it operates **before irreversible execution**.

## Counterfactual forecasting

For materially consequential actions, construct several futures rather than one
prediction. Each branch retains scenario, probability, severity, reversibility,
supporting evidence, assumptions, counterevidence, model version, timestamp.

## Minority-path preservation

Do not collapse models into a majority vote. A minority forecast may deserve
intervention when probability × consequence × irreversibility is sufficiently high.
**Low-probability catastrophic futures must remain visible rather than averaged away.**

## Intervention proportionality

Prefer the least irreversible intervention justified by the evidence. A prediction
alone must not automatically produce the strongest response.

## Calibration requirement

If predictions labelled ~70% likely occur 30% of the time, the system is badly
calibrated and unsafe. Measure predicted-probability band against observed frequency.

**Preventive Utility** = avoided expected harm − intervention cost − false-positive
cost. A predictor that blocks everything achieves strong recall and is useless.

## Counterfactual intervention testing

For selected scenarios run both branches — ALLOW and INTERVENE — and observe
consequences. This answers the stronger question: *did intervention materially
improve the outcome?*, not merely *did we predict danger?*

---

# Embodied Execution Safety / Robotic Compromise Defense (branch)

Not a separate product. The same governor applied where a command can move the world.

## Core doctrine

    COMMAND_AUTHENTICATED
    ≠ COMMAND_AUTHORIZED
    ≠ PHYSICAL_ACTION_ADMISSIBLE

A robot may receive a syntactically valid instruction from a valid identity and still
need to refuse, constrain, sandbox or reinterpret it because the physical consequence
is unsafe.

**Authentication proves where a command came from. It does not prove that executing
the command is physically safe.**

## Threat model, both directions

External → Robot: compromised accounts, stolen credentials, malicious updates, prompt
injection, hostile remote agents, command replay, spoofed sensors, poisoned context,
compromised model providers, capability escalation.

Robot → Environment: unsafe movement, collision, excessive force, unauthorized access,
unsafe tool use, property damage, privacy invasion, dangerous machine interaction.

## Embodied loop

    sensor / external signal → provenance → identity / authority → proposed action
    → environment state → humans / animals / objects / hazards
    → counterfactual physical consequences → safety + admissibility
    → actuation → sensor verification → outcome receipt

## Least-harmful branch

    normal execution → limit speed → limit force → replan path
    → request human clearance → local safe mode → protective stop

Do not treat every anomaly as requiring total shutdown. A robot performing care,
mobility assistance or lifting may itself create danger by abruptly powering down.

## Graceful authority degradation

    NORMAL → RESTRICTED → SAFE_MOTION_ONLY → LOCAL_CONTROL_ONLY
    → HUMAN_CONFIRMATION_REQUIRED → PROTECTIVE_STOP

Authority shrinks during operation when identity trust degrades, network posture
changes, behaviour drifts, sensor disagreement rises, or provenance becomes uncertain.

## Hard safety envelope

Actuator torque, speed, collision margins, protected-human proximity, emergency stop,
joint travel, thermal and payload limits, restricted tools, physical interlocks — these
sit *beneath* higher-level AI policy. Higher layers and ordinary software updates must
not silently override them. Assume some upper layer may eventually be compromised.

## Independent authority principle

**The greater the physical power of an autonomous system, the less authority any single
software component should possess over that power.**

Validation and authorization burden scale with force, speed, payload, reach, tool
capability, environment, human proximity, reversibility and consequence of failure.

## Cyber/physical isolation

Local physical-safety enforcement should remain operational when cloud connectivity is
lost, external model services are unavailable, a remote account is compromised, or a
provider becomes inadmissible. A remote command path must not disable the local safety
envelope merely because it holds ordinary control authority.

## Human-protection boundary

Not autonomous lethal robotics. For protective use cases: detection, warning,
illumination, alarm, access control, safe interposition, evacuation assistance,
communication, containment within lawful limits, summoning authorized assistance.
**Do not delegate intentional-injury decisions to predictive software.**

## Relationship to existing canonical reserves

- `emaa-external-machine-action-admissibility.md` — hostile external machine actions.
  CEG forecasts consequences; EMAA governs admissibility of external machine action.
- `unknown-physical-object-triage.md` — uncertainty about encountered physical objects.
- `protected-execution-zones.md` — bounded execution regions.
- `evidence-governed-runtime.md`, `computable-accountability.md` — receipt substrate.
- `regime-evidence-engine.md` / DRIFT — regime change invalidating prior competence.

## Human-use boundary

Explicitly prohibited as an initial use: predictive policing, pre-arrest scoring,
criminal propensity scores, sentencing recommendations, covert behavioural profiling,
punishment for an uncommitted act.

**Prediction must remain separate from judgment, and judgment separate from punishment.**

## Phase Zero

Controlled agent sandbox only. Thousands of synthetic tasks with known ground truth.
Never connect to real funds or real-world autonomous execution first.
Experiment definition: `docs/research/EXPERIMENT_CANDIDATES_2026-08-27.md`.

## Activation

Reserve only. Activate when: vLOID needs consequence forecasting before execution;
OROS has enough real execution traces for retrospective experiments; the Inference
Evidence Ledger produces a mature challengeable evidence graph; an agent deployment
creates concrete need for pre-execution simulation; or a sandbox experiment can run
without displacing revenue work.

## One-line definition

A pre-execution safety layer that forecasts multiple plausible consequences of a
proposed action, preserves credible minority-risk paths, selects the least irreversible
intervention warranted by evidence, and learns from the outcome — without treating
predictions as proof of future wrongdoing.

RESERVED. NO ACTIVE BUILD. NO ROBOTICS HARDWARE. NO WIRE ACTIVATION.
