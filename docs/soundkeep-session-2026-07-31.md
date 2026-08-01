# SoundKeep session log - 2026-07-31

## Shipped
- Session Architecture v1: playback receipts grouped into sessions with session_id on every
  event, 30-minute idle boundary, counters for played/completed/skipped, provider mix.
  mood and energy_band carried as explicit nulls - not derivable from playback events.
- pause / resume / seek events for both local and YouTube adapters.
- Seek debounced to one event per gesture. A single drag was emitting ~300 events and
  evicting every other event from the 300-entry log.
- Transport rebuilt into four honest controls: pause/resume, next, stop (halts, keeps the
  bar), close (dismisses). Previously two buttons did the same thing and one carried a skip
  icon for behaviour it could not perform.
- Single transport state: activateSessionTrack() is the only path that moves the session
  index, and it moves playback with it. Session view and audiobar previously drifted - the
  screen showed one track while audio played another.
- Session play/pause icon reads from the adapter rather than tracking its own state.
- Pathway depth raised from 6 to 13 per fetch; the frontend never passed the steps parameter
  the API already accepted.
- Continuous play: at the end of a session, fetch a new pathway anchored on the last track
  and append deduplicated results. Sets no longer dead-end.

## Known limitations
- The queue reads only from Sessions. Crates and Library are unordered, so local files cannot
  be played in sequence. Next greys out there rather than lying.
- "End of pathway" appears when the last track is reached, not when it finishes. Needs a
  Playback.hasEnded() concept that does not exist yet.
- Continuous play re-anchors on the last track, so drift compounds across extensions. Every
  hop is locally sensible; the whole journey may not be. Periodic re-anchoring on the
  original seed is the likely fix if it proves annoying in use.

## Still unexercised
The DJ workbench - crates, transition scoring, set-shape coaching, decks - has no real usage
behind it. The 29 local files are frequency tracks, not DJ material.
