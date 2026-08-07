# Tag-parsing fixtures

Hand-built ID3v2.3 files for testing `readTags` and `camelot`. Tag bytes plus a fake MPEG
frame header — not playable audio, which is deliberate: they exercise the parser without
shipping anyone's copyrighted music.

    01-standard-key      TIT2 TPE1 TBPM TKEY, key "Am"
    02-camelot-key       key written directly as "8A"
    03-bpm-only          BPM present, no key
    04-freetext-key      key "A minor-ish" — must store raw and yield null Camelot
    05-full              title artist album bpm(122.5) key genre year
    06-partial           title only, no artist
    07 - No Tags At All  no ID3 at all, filename fallback

## Running

The `audio/*` accept filter on the normal import rejects these, so test the parser
directly. In the browser console on SoundKeep:

    (()=>{const i=document.createElement('input');i.type='file';i.multiple=true;
    i.onchange=async e=>{for(const f of e.target.files){const t=await readTags(f);
    console.log(f.name,'->',t?JSON.stringify(t):'NO TAGS');}};i.click();})()

Then ⌘⇧G to this folder, ⌘A, Open.

## Verified 2026-08-07

All seven parsed as specified. Camelot conversion checked separately:
Am→8A, 8A→8A, F#m→11A, "A minor-ish"→null, C→8B, Dbm→12A, Bb→6B.
Harmonic neighbours: 8A~8B true, 8A~9A true, 8A~2A false.

## Still unverified
The full import path (extractMeta → library → display) needs real MP3s because of the
accept filter. The parser underneath it is proven; the plumbing above it is not.
