# RESERVE — Attention Value / Consumption Retention Intelligence (AVI)

Status: RESERVED — research reserve. NOT an active build.
Captured: 2026-08-27.
Adjacent to `revealed-preference-measurement.md`, which weights behavioural acceptance
over stated sentiment for *recommendation quality*. AVI concerns a different measure: the
gap between attention expended and value retained.

## Core observation

Digital-wellbeing systems predominantly measure **time spent**. A proposed improvement
counts **items consumed**. Neither answers the more important question:

    What did the user actually receive from the attention they spent?

    unconscious/high-velocity consumption → weak awareness of quantity
    → weak memory of consumed information → poor understanding of attention value
    → little actionable feedback → repeated automatic consumption

## Doctrine — separate measurement from interpretation

**Do not assume high consumption = wasted attention.** Watching 100 short videos may be
entertainment, research, education, music discovery, product discovery, professional
intelligence, social interaction, deliberate relaxation, or low-intent scrolling.

The measurement layer reports what is observable: items consumed · session duration ·
consumption velocity · sources encountered · topics · saves · revisits · shares ·
interaction depth · session start/stop behaviour · intended stopping point ·
continuation beyond it · repeated material · longitudinal baseline.

A separate interpretation layer may estimate intentional vs automatic consumption,
retained vs forgotten, high- vs low-value sessions, research vs entertainment vs habit,
abnormal deviation from personal baseline. **These classifications preserve uncertainty
rather than presenting inference as observation.**

    BAD:     "You wasted 43 minutes."
    BETTER:  "147 items over 43 minutes. 3 saved. 1 revisited.
              Session exceeded your stated target by 18 minutes."

## Retention sampling

Optionally and periodically sample recently consumed material and ask lightweight recall
questions, building longitudinal signal across consumption volume ↔ recall ↔
intentionality ↔ later usefulness.

## Attention Value Gap

Difference between attention expended and measurable or declared value retained.
Dimensions: volume · duration · intentionality · retention · revisit/use · source
diversity · interruption frequency · stopping control · user-declared value.
**Not a simplistic universal score without evidence.**

## Objective

Not necessarily to make users consume less. To **make invisible consumption legible**,
distinguish measurement from interpretation, help users understand what their attention
produced, and let them make their own informed choices.

## Privacy and safety

On-device processing where feasible · minimal content capture · derived metadata rather
than raw browsing history · explicit consent · user-controlled retention and deletion ·
no covert surveillance · no selling behavioural profiles · explainable derived signals.

## Platform constraint

Do not assume operating systems or third-party platforms expose enough to count or inspect
every consumed item. Any implementation must first determine platform APIs, accessibility
restrictions, browser capabilities, mobile OS limits, privacy constraints, and whether
useful measurement is possible without prohibited or fragile techniques.

## Generalization

Not restricted to short-form video: long-form video, social feeds, articles,
podcasts/audio, music discovery, educational material, research sessions, AI
conversations, professional information feeds.

Research question: *can a personal system measure not merely how much information a human
consumed, but how effectively attention was converted into something they valued or
retained?*

## Relationship to existing canonical reserves

`revealed-preference-measurement.md` · Information Admissibility Governor (observation vs
absence vs inference vs policy interpretation) · VERITY (reliability of consumed
information) · DRIFT (meaningful change against personal baselines) ·
`universal-timeline-semantic-index-engine.md` (reuse indexed content/time ranges rather
than reprocessing media) · SoundKeep (intentional listening without reducing engagement to
minutes or play counts) · `../RESERVE-VKOS.md` (knowledge → attention → retention →
capability).

These are architectural hypotheses, not required integrations.

## Activation

Revisit on strong evidence that time-based wellbeing tools fail for lack of value context;
when platform capabilities make privacy-preserving measurement feasible; when SoundKeep,
VKOS or the timeline engine develops a concrete need; when research demonstrates useful
relationships between consumption, recall and outcomes; or when a commercially credible
use case emerges that does not require surveillance-heavy collection.

RESERVED. DO NOT BUILD. DO NOT DISPLACE ACTIVE ROADMAP WORK.
