# SOUNDKEEP - Session Architecture v1

**Status:** RESERVED - next build after metadata pipeline fields
**Reserved:** July 26, 2026

## Purpose
Group individual playback receipts into coherent listening sessions that can be replayed, analyzed, resumed, and learned from by Sonic. Receipts already exist (shipped July 26); what is missing is the container that gives them an arc.

## Session lifecycle
Session created > track started > pause / resume / seek > track completed or skipped > continuation selected > next track > ... > session ended.

## Session fields
session_id, started_at, ended_at, device, provider_mix, total_duration, tracks_played, tracks_completed, tracks_skipped, pathway_id, mood, energy_band.

## Playback events
SHIPPED: start, complete, skip, playback_failed, fallback_offered, fallback_opened - each carrying provider, adapter, video_id, source (owned/catalog), position, duration, completion ratio, device.
TO ADD: pause, resume, seek, volume_change (optional), queue_change (future).

## Session receipt shape
Session > Track 1 (start > pause > resume > complete) > Track 2 (start > skip) > Track 3 (start > complete). The tree is the replayable artifact.

## Future uses
Replay a listening session; DJ session history; Sonic recommendation learning; pathway scoring; transition quality measured against real listening rather than metadata prediction; mood evolution; session analytics.

## Honest gate
The architecture should be built now; the intelligence on top of it waits for volume. Current data is 12 plays from one listener. Session receipts become valuable when testers generate real listening history - do not build learning features on a sample of one.

## Identity shift (recorded)
SoundKeep is no longer "a music player." The chain is: music > playback > receipts > sessions > learning > Sonic. The player exists to generate structured listening history that Sonic can learn from. That is the product, and it is a stronger identity than a player.
