# EXP-NARRATIVE-ROTATION-001 — Narrative Rotation & Signal Genealogy Replay

Status: `BLOCKED` — see Precondition. **Not runnable as specified until the case set exists.**
Registered: 2026-08-29
Consumes: `docs/research/experiments/EXP-GENEALOGY-001.md` (independence machinery —
not restated here)

## Hypothesis

When a narrative category accelerates, capital may rotate from already-expanded leaders
toward recognizable but relatively under-expanded constituents. A system observing narrative
breadth, relative expansion, wallet behaviour, liquidity, attention velocity and
admissibility may identify that transition earlier than price-momentum detection alone.

## PRECONDITION — the spec is invalid without it

**A single case cannot test this.** Selecting one instrument already known to have attracted
attention, and then showing the architecture could have found it, builds a hindsight
detector and measures nothing.

Requirements before any run:

- **20–50 historical narrative candidates**, including instruments that appeared equally
  promising and **died**.
- **Case selection frozen and recorded before outcomes are consulted.**
- The losing cases are load-bearing. They are how we learn whether "relative narrative
  laggard" is alpha or a compelling story.

Case 001 is a historical replay case, not the experiment.

## Replay boundary

Freeze a timestamp immediately before the selected late social-proof event. **No evidence
generated after that timestamp may enter decision features.** Later evidence is admissible
only for outcome evaluation.

## Genealogy requirement

Do not count correlated observations as independent confirmation. An influencer call, a
whale screenshot, a group repost and wallets from the same campaign may be **one causal
source, not four**.

Record per signal: observation → originating source → upstream causal source if known →
timestamp → independence class → confidence → downstream observations derived from it.

The independence machinery is owned by `EXP-GENEALOGY-001` and consumed here.

## Separated questions

**Regime:** can the system distinguish `NARRATIVE_IGNITION` / `LEADER_EXPANSION` /
`BREADTH_EXPANSION` / `ROTATION` / `SATURATION` / `DECAY` rather than treating each
instrument independently?

**Detection:** can narrative strength × relative under-expansion × attention acceleration ×
wallet accumulation × executable liquidity surface a candidate before price expresses the
move?

**Admissibility — kept strictly separate:** market opportunity and instrument admissibility
are **separate propositions**. A strong momentum signal must not override adverse contract,
holder, liquidity, bundle, insider, authority or provenance evidence.

**Conflicting third-party risk assessments remain conflicting evidence until independently
resolved. Do not average them into artificial certainty.** For Case 001 the observed
conflict is direct: one analyzer reported locked liquidity and no mint/blacklist/honeypot
risk; another reported extreme risk alleging launch bundling, prior-rug wallet association
and coordinated-wallet behaviour. Both are third-party claims. Neither is verified here.

## Counterfactual arms

    1  price/volume only
    2  + narrative
    3  + wallets
    4  + signal genealogy
    5  + admissibility

Compare detection time, false-positive exposure, confidence and resulting action.

## Actions and success

Actions: `BUY_CANDIDATE` / `WATCH` / `DEFER` / `DENY`. **The experiment executes no trade.**

**Success is not that Case 001 went up.** Success is demonstrating whether the architecture
could produce an earlier, reproducible, causally defensible decision from contemporaneously
available evidence without hindsight leakage.

## Failure conditions

Result depends materially on post-cutoff information · supposedly independent signals trace
to one campaign · candidate surfaced only after price already expressed the move · missing
data silently read as favourable · admissibility overridden because momentum looked
attractive · outcome knowledge changed historical feature interpretation.

## Receipts

Preserve dataset cutoff, source timestamps, feature state, genealogy graph, regime, score,
admissibility decision, simulated action and later outcome **separately**.

## Evidence boundary

Historical replay only. **Not a recommendation regarding any instrument, and not evidence
that any instrument was safe.** Conclusions hold only for the frozen case set, and only if
that set was frozen before outcomes were consulted.

## Provenance

    source artifact:       external market observation, 2026-08-28
    registered:            2026-08-29
    implementation commit: none — no implementation authorized, no live-capital activation
    evidence boundary:     frozen historical case set, replay only
    conclusion date:       pending
