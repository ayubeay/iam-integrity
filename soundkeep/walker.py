import json
from pathlib import Path

SEED_PATH = Path.home() / "iam-integrity/soundkeep/data/soundkeep_real_seed_v01.json"

def load_tracks():
    data = json.load(open(SEED_PATH))
    return data['tracks']

SET_ORDER = ["warm-up", "build", "peak", "wind-down"]

def next_set_position(current):
    idx = SET_ORDER.index(current) if current in SET_ORDER else 0
    return SET_ORDER[min(idx + 1, len(SET_ORDER) - 1)]

def score_candidate(anchor, candidate):
    score = 0
    reasons = []

    # Same genre = strong continuation
    if candidate['genre'] == anchor['genre']:
        score += 3
        reasons.append(f"same genre ({anchor['genre']})")

    # Same region = regional continuity
    if candidate['region'] == anchor['region']:
        score += 2
        reasons.append(f"same region ({anchor['region']})")

    # Energy progression
    energy_map = {"low": 1, "medium": 2, "high": 3}
    a_e = energy_map.get(anchor['energy'], 2)
    c_e = energy_map.get(candidate['energy'], 2)
    if c_e >= a_e:
        score += 2
        reasons.append("maintains or builds energy")
    else:
        score += 1
        reasons.append("energy winds down")

    # Set position progression
    next_pos = next_set_position(anchor['set_position'])
    if candidate['set_position'] == next_pos:
        score += 3
        reasons.append(f"correct set progression → {next_pos}")
    elif candidate['set_position'] == anchor['set_position']:
        score += 1
        reasons.append("holds set position")

    # Same era = era continuity
    if candidate['era'] == anchor['era']:
        score += 1
        reasons.append(f"same era ({anchor['era']})")

    # Cross-genre bridge (afrobeats ↔ dancehall is a natural bridge)
    bridges = {("afrobeats", "dancehall"), ("dancehall", "afrobeats"),
               ("hip-hop", "r&b"), ("r&b", "hip-hop")}
    if (anchor['genre'], candidate['genre']) in bridges:
        score += 2
        reasons.append(f"natural bridge: {anchor['genre']} → {candidate['genre']}")

    return score, reasons

def generate_pathway(anchor_id, tracks, length=5):
    anchor = next((t for t in tracks if t['track_id'] == anchor_id), None)
    if not anchor:
        print(f"Track {anchor_id} not found.")
        return

    pathway = [anchor]
    used_ids = {anchor_id}

    print(f"\n🎵 SOUNDKEEP PATHWAY")
    print(f"Starting from: {anchor['artist']} - {anchor['title']}")
    print(f"Genre: {anchor['genre']} | Energy: {anchor['energy']} | Position: {anchor['set_position']}\n")
    print("─" * 50)

    current = anchor
    for step in range(length):
        candidates = [t for t in tracks if t['track_id'] not in used_ids]
        if not candidates:
            break

        scored = []
        for c in candidates:
            score, reasons = score_candidate(current, c)
            scored.append((score, reasons, c))

        scored.sort(key=lambda x: -x[0])
        best_score, best_reasons, best = scored[0]

        print(f"Step {step + 1}: {best['artist']} - {best['title']}")
        print(f"  Genre: {best['genre']} | Energy: {best['energy']} | Position: {best['set_position']}")
        print(f"  Why: {', '.join(best_reasons)}")
        print()

        pathway.append(best)
        used_ids.add(best['track_id'])
        current = best

    return pathway

if __name__ == "__main__":
    tracks = load_tracks()

    print("Available anchor tracks:")
    for t in tracks:
        print(f"  {t['track_id']}: {t['artist']} - {t['title']} ({t['genre']}, {t['set_position']})")

    print()
    anchor = input("Enter anchor track_id (e.g. real_001): ").strip()
    generate_pathway(anchor, tracks, length=5)
