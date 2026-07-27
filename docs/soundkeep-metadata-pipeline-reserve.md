# SOUNDKEEP - Metadata Pipeline

**Status:** PARTIALLY SHIPPED July 26, 2026 / RESERVED (confidence + AI enrichment)
**Reserved:** July 26, 2026
**Binding rule:** No more parser patches once this pipeline is in place. Add pipeline stages, not regex exceptions.

## Pipeline
Import > embedded tags (ID3v2, MP4 atoms) > conservative filename parser > AI enrichment (future) > confidence score > user edits > locked metadata.

## Fields
metadata_source: embedded | filename | ai | user
metadata_confidence: 0-100
metadata_locked: true/false (set on user edit; nothing automatic may overwrite)
metadata_version: schema version for safe future migrations

## SHIPPED
Embedded tag reading (ID3v2 frames TIT2/TPE1/TALB, MP4 nam/ART/alb atoms); conservative filename parsing that refuses to invent artists from labels (mantra, frequency, morning, day, hertz, (audio), bare numbers); split on hyphen, en-dash, em-dash; signature dedupe on import (SHA-256 of first 128KB + size); safe migration that rekeys audio blobs in lockstep so playback never breaks; per-file rename with metadata_source=user and metadata_locked=true; migration receipts.

## RESERVED
Confidence scoring (embedded tags high, clean Artist - Title parse medium, label-fallback low); AI enrichment for untagged files; artwork extraction; album and track-number handling; MusicBrainz or AcoustID lookup as an enrichment stage; batch review UI for low-confidence records.

## Doctrine
The parser's job is to stop inventing facts. "Unknown Artist" is a correct answer when no artist exists - a fabricated artist is worse than an absent one. The pencil (user edit) is the escape hatch, and user edits are permanent.

## Known open item
One orphan library record with no audio blob: "(Audio) Day 4 :: 639 Hertz" - re-import to resolve, harmless until then.
