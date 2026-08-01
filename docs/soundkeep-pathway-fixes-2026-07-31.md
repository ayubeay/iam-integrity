# Pathway engine fixes - 2026-07-31

## What was wrong
Four defects, all the same shape: the engine asserting more than it measured.

1. ANCHOR IGNORED. /api/dynamic_pathway took only an artist and set
   anchor = lastfm_tracks[0]. Clicking "KANTE" and clicking "Unavailable" sent identical
   requests; the selected track never appeared.

2. IDENTICAL PATHWAYS. Candidates were drawn only from load_tracks() - the local seed
   catalog. Last.fm supplied the anchor and nothing else, so every afrobeats anchor produced
   the same deterministic walk. Davido, Kizz Daniel and Asake returned byte-identical
   continuations.

3. FABRICATED METADATA. lastfm_search hardcoded energy=medium, bpm_range=90-110, era=2020s,
   mood=vibe, set_position=warm-up on every track. Four of the six scoring signals were
   constants, so Sonic reported "maintains energy" and "same era" about values never observed.

4. DEFAULTS SCORED AS OBSERVATIONS. energy_map.get(None, 2) silently returned medium, so
   unmeasured energy scored as a match.

## Fixes
- anchor honours an optional title parameter, matched case-insensitively against the artist's
  top tracks; frontend passes the selected track's title.
- lastfm_similar_pool(): artist.getsimilar -> top tracks per neighbour, genre and region
  inferred from tags, everything else left None. Candidate pool = seed catalog + neighbours.
- lastfm_search limit raised 5 -> 15.
- walker scores a signal only when both sides carry it. No defaults.
- scores normalized over comparable signals, then weighted by coverage:
  fit * (0.6 + 0.4 * available/11) * 10. A perfect match on two signals scores below a
  perfect match on five - fewer signals means less evidence, not more certainty.
- artist diversity cap: no artist may appear more than twice in a walk.

## Result
Davido and Asake now produce different pathways, 8 distinct artists each, scores ranging
7.8-8.2 rather than a flat 10.0. Neighbours surfaced include Joeboy, BNXN, Young Jonn and
Wizkid - none in the seed catalog.

## Known limitation
Last.fm candidates are judged on genre and region only. The coverage weighting keeps that
honest in the ranking, but the pathway UI does not yet show how much evidence sits behind
each score. Same problem SURVIVOR solved by reporting coverage alongside the decision.
