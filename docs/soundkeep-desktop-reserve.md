# SOUNDKEEP DESKTOP (macOS, Tauri) - Next AAIE Milestone

**Status:** QUEUED - the next AAIE milestone, ahead of any further browser workarounds
**Reserved:** July 26, 2026

## Why desktop, not more Safari work
Safari does not implement the File System Access API, so it cannot hold a live directory handle - no rescan, no watching, no reference mode. Chromium gives linked folders; Safari can only copy files into browser storage, which is wrong for a real library (storage bloat, eviction risk, misrepresents reference mode). iOS is WebKit for every browser, so folder sync is impossible there in a browser at all. Serious local-library tools are desktop apps for exactly these reasons.

## Product line (decided)
SoundKeep Web (Safari/Chrome/Edge): playback, crates, discovery, decks, transition intelligence, limited file import.
SoundKeep for Mac (Tauri): native folder authorization, reference-mode indexing, recursive scanning, filesystem watching, external-drive support, automatic New Arrivals, autonomous Sonic intake.
Message: "SoundKeep works on the web. Automatic library sync requires SoundKeep for Mac." That is the correct product form, not a weakness.

## Phase 1 - Mac companion shell
Wrap the existing frontend in Tauri unchanged. Native folder picker; scoped read-only permissions; recursive scan; SQLite library index; reference-mode playback; AAIE receipts; New Arrivals.

## Phase 2 - Actual autonomous intake
Filesystem watching for creation, rename, move, modify, delete, new subfolders. Pipeline: new file detected > wait for copy completion > validate audio > signature > duplicate check > metadata extraction > New Arrivals > intake receipt.

## Phase 3 - DJ-native sources
Music, Downloads, Rekordbox/Serato/Traktor folders, Ableton exports, external SSDs and USB drives, sample and stems directories. WATCH AUDIO FOLDERS ONLY - do not read or modify another DJ application's database until its format and safety requirements are properly understood.

## macOS permissions doctrine
Request the narrowest permission possible; handle privacy prompts honestly; removable drives and protected locations are explicit user grants. Binding rule carries over: reference mode reads and indexes files, never moves, edits, or uploads them.

## What the browser work bought us
Not wasted: it is the intake UX prototype, the Chromium beta, the frontend Tauri will reuse, and proof that indexing, receipts, dedupe, and New Arrivals work.
