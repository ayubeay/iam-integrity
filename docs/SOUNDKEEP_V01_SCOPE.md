# SoundKeep v0.1 — Scope

## Directional intent

Long-term: something genuinely better for music discovery,
coordination, taste continuity, and artist/listener alignment than
current platforms (Spotify, Apple Music, Pandora). Not a streaming
clone. Specific differentiation to be determined by observed usage,
not theory.

v0.1 is **not the product**. v0.1 is the smallest possible surface
that lets real DJs touch the core hypothesis: that intentional
taste continuity creates sticky behavior in a way that current
recommendation systems do not.

## The hypothesis under test

Current music platforms recommend via "people who listened also
liked" — a behavior-similarity heuristic that often feels random
or noisy to power users (DJs especially). The alternative being
tested is: **recommendation pathways that feel intentional** —
continuation across genre, mood, energy, region, era, underground
adjacency, remix culture, and crowd transition potential.

If intentional continuation creates engagement that
behavior-similarity does not, v0.1 should reveal it through
observed behavior (return visits, save/group actions, sharing
with other DJs, unprompted feature requests).

If it doesn't reveal anything, that's also evidence — the wedge
needs reframing, not a bigger build.

## v0.1 surface

A web page. No mobile app. No audio playback. No streaming infra.
No licensing layer.

1. **Input.** DJ pastes a few tracks, artists, genres, or vibes
   they currently like or want to explore.

2. **Output.** System generates a continuation pathway —
   recommendations that move intentionally across the taste space,
   not just adjacent picks. Each recommendation includes artist,
   title, genre, energy, and optional outbound preview links to
   YouTube, SoundCloud, Spotify, or Apple Music.

3. **Interaction.** DJ can save, skip, group, or revisit
   recommendations. State persists across visits.

4. **Observation.** System records which recommendations get
   clicked, saved, skipped, grouped, and which DJs return.
   Receipts of these interactions follow the existing signed-receipt
   pattern (vyre_v1, Ed25519) for continuity with the broader stack.

## What v0.1 explicitly does NOT include

The following are all real and potentially valuable, but **not
v0.1**. Each is captured in `RESERVED.md` or implied by the same
discipline:

- No audio playback or hosting
- No track licensing layer
- No streaming infrastructure
- No mobile app (web-first)
- No staking
- No SURVIVOR risk gating
- No DJ reputation graph
- No track provenance receipts (separate from interaction receipts)
- No social/sharing features beyond outbound preview links
- No payment tier
- No "AI DJ" marketing framing
- v0.1 is not attempting to replace existing streaming platforms

Building any of these into v0.1 turns the validation timeline from
weeks into months and confounds the signal — making it impossible
to tell whether DJs engage because the continuation is good or
because some other feature is doing the work.

## Test users

2-3 reachable DJs/producers. Names held privately; they will be
contacted directly once v0.1 is testable.

They will not be interviewed for pain points in advance. The
hypothesis is that observed behavior (what they click, save, skip,
return to, share) reveals fit better than self-reported pain. This
is consistent with how consumer products are typically validated
when the user has a stable existing workflow but no acute named
complaint.

## What success looks like

At least one of:

- One or more test users returns voluntarily, more than once,
  unprompted
- One or more test users shares the v0.1 URL with another DJ or
  producer
- One or more test users makes a specific feature request
  (signaling they see latent value worth extending)
- Quantitative: save/skip ratio departs meaningfully from random.
  A sustained save rate above random for an individual user is
  directional signal, but not proof of product-market fit.

## What failure looks like

- Test users open v0.1 once, don't return
- No save/skip pattern (random clicking)
- No sharing
- No feature requests
- Silence after initial test

If this is the outcome, **the response is not to try harder on the
same surface.** Failure here is signal that intentional-continuation
is not the wedge, and reframing is required before more build.

## Build constraints

- ≤2 weeks of focused build time, not calendar time
- Web-first, no native app
- Backend can reuse existing iam-integrity stack (signed receipts,
  agent persistence, scope contracts) where it accelerates — but
  not where it adds complexity
- Continuation logic is the differentiator; everything else is
  scaffolding
- No new infrastructure layers (no new Railway projects, no new
  databases, no new signing keys)

## What v0.2 might be (not commitments)

If v0.1 produces real signal, v0.2 could plausibly add (decided
by what the signal actually shows):

- Audio playback or preview integration
- Mobile companion
- Multi-user shared crates
- Collaboration features
- Premium tier exploration

If v0.1 produces no signal, v0.2 is not a thing — the project
restarts at scope, not at build.

## Activation condition for any expansion

No feature from `RESERVED.md` or from the "v0.2 might be" section
above gets pulled into v0.1 unless one of the success signals
above is observed. Same discipline as the rest of the doctrine.
