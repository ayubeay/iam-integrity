# SOUNDKEEP — DJ Workbench Roadmap

**Created:** July 17, 2026
**Position:** The AI-powered DJ preparation and set-building workspace. Not a Serato clone — no HID/DVS/latency/driver work. Sonic assists the DJ, never replaces them.

## Phase 1 — Real DJ Crates [SHIPPED July 17, 2026]
Ordered tracks, reorder, per-track transition notes, per-crate set notes, rename, energy arc + BPM span summaries from real metadata. Camelot/duration intentionally absent until a data source exists.

## Phase 2 — Preparation Deck [GATED: requires audio access]
Waveform, cue points, loop markers, beat grid, hot cues, waveform zoom. HARD CONSTRAINT: SoundKeep holds no audio files; these tools are impossible honestly until licensed audio or user-local file import exists. Buildable earlier without audio: the metadata prep surface — adjacent-track transition analysis, ordering, notes (Phase 1 already seeds this).

## Phase 3 — Sonic in the Deck [buildable on metadata first]
Load Track A + Track B: transition confidence from genre/energy/region/era/set-position/BPM-range overlap; better-bridge suggestions (existing walker); key-shift advice gated behind key data existing. This is the moat — tied to DJ workflow, hard for Spotify/Apple to replicate. Metadata version can ship BEFORE Phase 2 audio tools.

## Phase 4 — Performance Mode [RESERVED, last]
Deck A/B, crossfader, EQ, filters, loops, effects, waveform sync. Only after prep tools have weekly DJ usage.

## Sequencing Rule
1 (done) → 3-metadata → 2 (when audio exists) → 3-full → 4. Weekly DJ usage of crates + prep is the gate between every phase.
