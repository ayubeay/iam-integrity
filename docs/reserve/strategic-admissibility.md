# RESERVE - Strategic Admissibility

**Status:** Reserved future architecture. Not an active build.

## The shift
As AI makes tactical capability abundant, the scarce layer moves up. The question stops
being "what can this agent do?" and becomes "given everything it can do, what SHOULD it do
now, under what strategy, with what constraints, and how do we later prove the action
advanced the objective?"

## The distinction
    existing control plane   is this action admissible?
    strategic layer          is this the right admissible action to spend resources on?

An action can be technically possible, authorized, safe and trustworthy - and still be
strategically useless.

## Decision vocabulary, distinct from execution
    ADVANCE  HOLD  EXPERIMENT  GATHER_EVIDENCE  DEFER  ABANDON

Different from ALLOW / THROTTLE / DEFER / DENY. A candidate action can be execution-ALLOW
and strategy-HOLD.

## Primitives worth researching
Desired state. Anti-vision - explicit failure states like dependency concentration, feature
accumulation without users, spending resources without new information. Strategic position -
capital, compute, time, distribution, customers, datasets, relationships, reliability.
Capability inventory. Bottleneck detection. Leverage and optionality. Resource allocation.
Horizon separation. Adaptation when assumptions break.

    capability != permission != strategic desirability

## Evidence from the Robinhood work
The valuable result did not begin with execution:

    discover capability surface -> identify OAuth constraints -> default-deny firewall
    -> classify all live capabilities -> establish real account-state inputs
    -> aggregate cross-account state -> define policy semantics
    -> bounded execution authorization -> replay, drift and race protections
    -> separate evidence from authority -> sign authorizations -> dry execution boundary

The final tactic became available because earlier moves improved the position.

    BUILD POSITION BEFORE USING POWER.

## Doctrine
Do not mistake activity for progress. Prefer actions that remove a real bottleneck, generate
decision-changing evidence, strengthen reusable capability, improve optionality, reduce an
important risk, establish distribution, validate demand or produce compounding
infrastructure.

A technically impressive action advancing none of those may be strategically useless.

## Strategy is not zero-sum
Not "weaken an opponent" but: repeated improvement of position, preservation and expansion
of useful optionality, accumulation of reusable capability, intelligent allocation of
constrained resources, adaptation toward a desired state. Where collaboration helps, prefer
mutual capability compounding over extraction.

## The trap to avoid
Do not build an abstract AI strategist that asks an LLM to generate strategy. That
reproduces the tactic-collection problem one level up. A real system needs explicit state,
objectives and constraints, committed evidence, deterministic policy where feasible,
measurable outcomes, feedback, receipts and adaptation. LLMs may propose; they are not the
authority on validity.

## Activation
Revisit when one product has enough execution history to ask which actions actually improved
position, which merely consumed resources, and whether that difference can be formalised.
