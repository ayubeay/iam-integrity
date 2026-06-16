import json
from pathlib import Path

output = Path.home() / "iam-integrity/soundkeep/data/soundkeep_real_seed_v01.json"
output.parent.mkdir(parents=True, exist_ok=True)
tracks = []
track_num = 1

print("SoundKeep Real Seed Builder — target 15-25 tracks")
print("Type 'done' for artist to finish.\n")
print("Genres: afrobeats | dancehall | hip-hop | r&b | soca | fuji | amapiano")
print("Energy: low | medium | high")
print("BPM range: 70-80 | 80-90 | 90-100 | 100-110 | 110-120 | 120-130")
print("Region: nigeria | jamaica | us | uk | south-africa | caribbean")
print("Era: classic | 2000s | 2010s | 2020s")
print("Mood: hype | vibe | romantic | party | spiritual | chill")
print("Set position: warm-up | build | peak | wind-down\n")

while True:
    print(f"--- Track {track_num} ---")
    artist = input("Artist (or 'done'): ").strip()
    if artist.lower() == 'done':
        break
    title = input("Title: ").strip()
    genre = input("Genre: ").strip().lower()
    energy = input("Energy: ").strip().lower()
    bpm_range = input("BPM range: ").strip()
    region = input("Region: ").strip().lower()
    era = input("Era: ").strip().lower()
    mood = input("Mood: ").strip().lower()
    set_pos = input("Set position: ").strip().lower()

    tracks.append({
        "track_id": f"real_{track_num:03d}",
        "artist": artist,
        "title": title,
        "genre": genre,
        "energy": energy,
        "bpm_range": bpm_range,
        "region": region,
        "era": era,
        "mood": mood,
        "set_position": set_pos,
        "source": "real_seed_v01"
    })
    print(f"✓ {artist} - {title}\n")
    track_num += 1

result = {"version": 1, "track_count": len(tracks), "tracks": tracks}
json.dump(result, open(output, 'w'), indent=2)
print(f"\nDone. {len(tracks)} tracks saved to {output}")
