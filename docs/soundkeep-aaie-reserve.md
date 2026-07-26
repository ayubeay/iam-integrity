# SOUNDKEEP - Autonomous Audio Intake Engine (AAIE)

**Status:** PARTIALLY SHIPPED July 26, 2026 (Folder Sync) / RESERVED (full engine)
**Origin:** Manual one-file-at-a-time import is the first thing that kills tester patience - felt directly during self-testing before outreach.

## Guiding principle
SoundKeep builds the user's music library automatically. The user listens; SoundKeep discovers, organizes, indexes, and maintains.

## SHIPPED (browser, v21)
One-time folder authorization via File System Access API; recursive scan of an entire library (5000-file cap per scan); REFERENCE MODE as default - handles stored in IndexedDB, files read on demand, nothing copied or uploaded; dedupe by content signature (SHA-256 of first 128KB + byte size) catching same-audio-different-filename; import receipts (id, timestamp, source, path, mode, format, size, signature, duplicate status) capped at 500; New Arrivals inbox with Keep-all; per-source rescan and disconnect; webkitdirectory fallback for non-Chromium browsers (copies into IndexedDB).

## GATED ON DESKTOP SHELL (Tauri/Electron)
True background watching while the app is closed; filesystem events (create/modify/rename/move/delete); external and network drives; ZIP extraction watching; crash recovery and resume; millions-of-tracks scale. Browsers cannot watch the filesystem in the background - this is the single strongest argument for the desktop build.

## GATED ON DSP
Acoustic fingerprinting (current dedupe is a content signature, NOT acoustic - it will not catch the same recording at a different bitrate). Honest labeling required in UI until real fingerprinting exists.

## RESERVED
Managed Library Mode (Artist/Album/Track reorganization); metadata and artwork extraction from tags (currently filename parsing only); smart classification; searchable receipts; codec intelligence and transcoding; intake sources beyond local folders (cloud sync folders, DJ software exports, browser download integration, mobile companion).

## Constraints (binding)
Local-first: no audio uploaded, no cloud dependency, metadata processing on-device. Reference mode never moves or modifies the user's files. Disconnecting a source removes index entries only - never the files.
