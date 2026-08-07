# Reserve: Availability Intelligence & Distribution Receipts

**Status:** Reserved. Do not activate until current priorities are complete.

## Motivation
The common discovery problem is not "I cannot identify this song." It is:

    I know the song but cannot find where it exists.
    It is on SoundCloud but nowhere else. It was removed from Spotify. It is Bandcamp only.
    It was on a mixtape. It was a regional release. It was taken down. It was never
    commercially distributed.

Streaming platforms answer only whether something exists in their own catalogue. SoundKeep
should answer where a recording exists across the ecosystem, and why.

## Availability receipt
Platforms searched, and per platform: FOUND, NOT FOUND, REGIONAL, REMOVED, PRIVATE,
UNLISTED, UNKNOWN.

Distribution classification: commercial DSP release, independent release, SoundCloud-only,
Bandcamp-only, mixtape, demo, bootleg, fan upload, live recording, archive, promotional.

## Distribution reasoning
Why something appears unavailable: never distributed, artist chose not to release,
licensing restriction, regional availability, contract expired, takedown, artist removal,
platform moderation, temporary outage, unknown. Every conclusion carries its evidence.

## Provenance layer
Original source, earliest observed publication, known mirrors, official versus unofficial
upload, re-upload chain, and confidence in artist ownership, distributor and rights.

## Availability timeline
How distribution changed over time - SoundCloud only, then DSPs, then removed, then
Bandcamp only - using historical indexing and archived metadata where available.

## Cross-platform discovery
Evidence from lyrics, metadata, scene descriptions, film and TV soundtrack databases,
artist websites, community databases, forums, streaming catalogues, independent
distribution platforms and archives.

## Design principle
Do not only answer "is this song here?" Answer where it exists, why it is available there,
why it is unavailable elsewhere, and what evidence supports that conclusion. Availability
becomes a first-class search dimension alongside lyrics, melody, humming, emotion, genre,
instrumentation, soundtrack matching and provenance.
