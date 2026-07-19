# SOUNDKEEP — The Vault (Sound Keeping Layer)

**Status:** RESERVED — deepens Reserve Module 1 (Music Ownership Layer)
**Reserved:** July 18, 2026
**Origin:** The name's meaning. People's owned music is scattered across devices and dies with them. SoundKeep keeps your sounds as yours — forever.

## The problem
Downloaded music lives in fragments: old laptops, dead phones, USB sticks, download folders. Device lost = collection lost = years of taste and DJ history gone. Streaming does not solve this — streamed music is rented, revocable, region-locked, and disappears when catalogs change. Nobody owns their listening life.

## The promise
Your sounds, kept. Import once, keep forever: files, library, crates, transition notes, listening receipts, set history — the whole musical identity, recoverable on any device.

## What already exists (v18, shipped July 18)
Local import + IndexedDB storage + in-app playback + listening receipts. This IS the vault's on-device layer. What is reserved is survival beyond the device.

## Reserved capabilities
1. **Cloud Vault** — encrypted per-user backup of imported files. Requires accounts. Private locker only: each user's own files, accessible only to them. Never shared, never streamed to others, never redistributed — the legal line that keeps this a storage service, not a music service.
2. **Device-loss recovery** — sign in on a new device, restore everything: files, library, crates, notes, receipts, session history.
3. **Collection provenance** — import receipts per file: date, original filename, content hash, source device. Proof of what your collection is and when it became yours. (vLOID receipts doctrine applied to ownership.)
4. **Collection health** — duplicate detection by content hash, broken-file detection, metadata repair suggestions.

## Claude additions (proposed, not yet ratified)
5. **Export Everything, before accounts exist** — a one-tap "Export my SoundKeep" producing a portable archive (library JSON, crates, pathways, notes, receipts; audio optional). Ownership doctrine demands the exit door exist BEFORE the vault does — keeping must never become locking in. Buildable now, no accounts needed. Candidate for next build queue.
6. **Storage as the natural revenue tier** — vault storage is the honest thing to charge for (like iCloud), and maps cleanly onto SK staking tiers: keep more, stake more. Revenue aligned with the promise rather than against it.
7. **Crate lineage in the vault** — not just files: the evolution (found > practiced > mixed > played live > still in rotation after six months) preserved as the collection's story. Feeds the Journey engine when it wakes.

## Hard constraints (binding)
- User-owned files only; no acquisition, no sharing, no public streaming from vaults.
- Encryption at rest; the user's vault is theirs, including from us.
- Storage costs are real: vault is gated behind accounts + a sustainable tier, never a free-unlimited promise that dies.
- Export must always work. The keep is a promise, not a cage.

## Activation order
v18 on-device layer (done) > Export Everything (queue candidate) > accounts > cloud vault + recovery > provenance + health > lineage.

## Product definition update
SoundKeep = the continuous music workflow (discover, listen, organize, practice, learn) PLUS the permanent home for what you already own. Intelligence + Keeping. The name was the spec all along.
