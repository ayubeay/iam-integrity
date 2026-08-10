# RESERVE - Music Identity Graph & Version Lineage

**Status:** Reserved architecture. Do not implement until the library and metadata layer
requires canonical cross-release identity resolution.
**Urgency:** MEDIUM-HIGH as a data model. It is the substrate two existing reserves assume.

**Not the same as** docs/soundkeep-creative-identity-graph-reserve.md, which concerns
ARTIST identity - collaborators, credits, relationships, the MySpace-adjacent layer. This
reserve concerns MUSIC identity. Both are graphs; they describe different things.

## The structural problem
Music platforms treat a song, a recording, a release appearance and a playable file as
though they are the same object. They are four things.

    Musical Work  ->  Recording / Version  ->  Release Appearance  ->  Playable Asset

**Musical Work** - the underlying composition.
**Recording / Version** - a particular realisation: original studio, radio or single edit,
remix, remaster, clean or explicit, alternate mix, live, acoustic, cover.
**Release Appearance** - that recording on an album, single, EP, compilation, soundtrack,
deluxe or regional edition.
**Playable Asset** - the actual file or stream: a local FLAC, an MP3, a provider object, a
bitrate or codec variant.

## Relationships, not flattening
    SAME_WORK        SAME_RECORDING      ALTERNATE_EDIT_OF
    REMASTER_OF      REMIX_OF            LIVE_VERSION_OF
    COVER_OF         APPEARS_ON          DERIVED_FROM

## User state attaches at the right level
Favourite the work. Rate a particular recording. Prefer a particular release. Keep play
counts per asset. Build playlists from either canonical identities or deliberately chosen
versions.

This stops duplicate album appearances fragmenting a listener's history, and equally stops
genuinely different versions - a live take, a remix, a radio edit - being collapsed into
one.

## Identity resolution
Standardised identifiers, metadata, audio fingerprints, duration and structure comparison,
release and label information, human verification. Conclusions carry confidence, provenance
and receipts, so SoundKeep can explain WHY two objects were linked, separated or classed as
variants.

**Do not assume matching artist and title means two recordings are identical.** That
assumption is the origin of most library corruption.

## Identity-aware playback
If the preferred asset is unavailable, another verified representation of the SAME RECORDING
from an authorised source may substitute. Substitution must respect identity boundaries -
never silently replace an original studio recording with a remix, live version, radio edit
or materially different master because the title matches.

## Relationship to existing reserves
    Rights Continuity & Royalty Recovery     ownership, administration, rights
    Music Verification & Provenance          determines and verifies version and master
    Music Identity Graph (this)              the canonical structure those findings populate

This is the substrate, not a third rights engine. SaaS-squared workflows would later use the
same lineage for collaboration version history, master provenance, licensing and delivery.

## Why it matters concretely
SoundKeep's own library currently holds entries like "Mantra 10 - The Fire of Divine
ProtectionOm D" - a filename fragment treated as a track identity. Without the four-layer
model there is nowhere for a corrected title, a verified recording, or a preferred master to
live.

## Principle
Do not confuse the song with the recording, the recording with the release, or the release
with the file. Understand how all four relate while preserving their individual identities.
