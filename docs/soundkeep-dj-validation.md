# SOUNDKEEP — DJ Validation Protocol v1

**Status:** Active
**Created:** July 13, 2026
**Purpose:** Answer the biggest open unknown: do DJs think like this?
**Method:** 5-7 structured conversations, 15-20 min each. Deejay Freddy first.

## Pre-Committed Decision Gates
Written BEFORE interviews to prevent post-hoc rationalization.
- GATE A (BPM promotion): if 3+ of 5 DJs stack-rank BPM or Camelot in their top 2 trust signals, promote BPM-aware pathways to next engineering sprint.
- GATE B (live-use viability): if 2+ DJs say they would open SoundKeep during or while planning a real set, the DJ wedge is validated. If 0 do, reposition toward curators/listeners before more DJ features.
- GATE C (tagline): whichever hero wins 4+ of 5 preferences ships. Tie = keep current.
- No gate = no build. Feedback that hits no gate goes to the reserve doc, not the roadmap.

## Screener
Counts as a DJ if they: play live sets at least monthly (club, event, radio, or livestream) OR plan sets professionally. Note genre focus and region.

## Script

### 1. Current workflow (before showing anything)
- Walk me through how you pick the next track mid-set. What are you looking at or listening for?
- What happens when a transition misses? How do you recover?
- What tools do you use now (rekordbox, Serato, USB crates, memory)? What do they get wrong?

### 2. Live demo (hand them the site, say nothing)
- Task: "Search an artist you actually play. Read what comes back."
- Observe silently: do they read the reasons? Do they scroll the pathway? Do they click listen links? Where do they hesitate?

### 3. Trust signals (the core question)
"For Sonic's suggestion to be trustworthy enough to act on, what information has to be there?" Then stack-rank these five:
- BPM
- Camelot / key compatibility
- Energy level
- Regional bridge (e.g. Nigerian Afrobeats vs Jamaican Dancehall)
- Set position (warm-up / build / peak / wind-down)
Record the full ranking, not just #1. Ask: what's missing from this list?

### 4. Tagline A/B
Show both, ask which one is FOR THEM:
- A: "Find the next song before the crowd loses the feeling."
- B: "Protect the flow. Never lose the room."
Ask why in one sentence. Record verbatim.

### 5. Commitment signals (in order of strength)
- Would you use this while planning your next set? (weak yes)
- Would you open it live in the booth? (strong yes)
- Would you pay for it? What would it have to do first? (price anchor)
- Who is one DJ you'd send this to tonight? (referral = strongest signal)

## Capture Template (one per DJ)
Name / genre / region / sets per month:
Workflow notes:
Demo observations (verbatim where possible):
Trust stack-rank (1-5):
Missing signal they named:
Tagline pick + why:
Commitment level (plan / booth / pay / refer):
One quote worth keeping:

## Anti-Patterns
- Do not pitch. Show, then shut up.
- Do not explain a confusing screen — write down WHERE they got confused instead.
- Do not count politeness ("this is cool") as signal. Only gates count.

---

# v2 - Hands-On Session Protocol (added 2026-08-07)

**Occasion:** DJ Taf, Tuesday, at his house, several hours, HIS library on HIS setup.
**Why a v2:** the v1 script was built for 15-20 minute structured interviews. This is a
different instrument - long, unstructured, with a real library. The v1 gates still stand;
this adds what a long session can reveal that a short one cannot.

## What this session is for
Not a demo. Not a feature tour. The product should disappear into the workflow while an
actual DJ manages actual music. Every place he pauses is data.

The v1 anti-patterns apply with more force over several hours: do not pitch, do not explain
a confusing screen, do not count politeness as signal.

## The one thing a long session reveals that a short one cannot
Scale. His library may be thousands of tracks. Import speed, library rendering, crate
operations and search have only ever been exercised against ~29 files. Whatever breaks
will break in the first ten minutes, in front of him.

Test this BEFORE Tuesday against a seeded library of several hundred to a few thousand
entries. If the UI collapses, that outranks every feature.

## Priority order for the build before Tuesday
1. **Fast library import.** Whatever he has, ingested without a fight. Report library
   statistics afterward - artists, albums, genres, duplicates, missing metadata.
2. **Metadata intelligence.** Detect and EXPLAIN rather than silently fix: duplicates,
   split albums, inconsistent album artists, inconsistent years, missing artwork, missing
   BPM, missing key, missing genre. Recommendations, not mutations.
3. **Library health report.** Where cleanup is needed, at a glance.
4. **Duplicate classification.** Not deletion. Exact duplicate, different bitrate, radio
   edit, extended mix, explicit version, remaster, live recording.
5. **Explain every recommendation.** Same ID, same track count, capitalisation differs -
   the reason, not a verdict. No black-box behaviour.

Stretch only if time allows: BPM estimation, harmonic key, artwork recovery, cue point
suggestions, energy analysis.

## What to watch for
Where he pauses to think. Where he searches manually. Where he renames a file. Where he
fixes metadata by hand. Where he hesitates. Where he switches applications. Where he copies
files himself. Where he asks where something is. Where he gets confused.

Each is a feature request that has not been written down yet.

## Session receipt
Afterward, record: duration, library size, issues found by category, manual corrections
prevented, top pain points in his words, and the next priorities that follow from them.

## The question this session should answer
Can SoundKeep become the intelligence layer above an existing music collection - turning
hours of manual library management into minutes, while keeping trust through explainable
recommendations?

If that trends yes, the roadmap continues toward knowledge, organisation, provenance and
explainable automation rather than more playback features.

## Relationship to the v1 gates
Gates A, B and C remain. A long hands-on session can satisfy Gate B directly - if he opens
SoundKeep while planning or playing a real set, that is the strong signal. It cannot
satisfy Gate A alone; BPM promotion still needs 3 of 5 DJs.
