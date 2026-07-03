from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json, sys, os, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from walker import load_tracks, score_candidate, next_set_position, SET_ORDER

app = FastAPI()

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"

def lastfm_search(artist: str):
    """Search Last.fm for artist top tracks"""
    try:
        # Get artist top tracks
        res = requests.get(LASTFM_BASE, params={
            "method": "artist.gettoptracks",
            "artist": artist,
            "api_key": LASTFM_API_KEY,
            "format": "json",
            "limit": 5
        }, timeout=5)
        data = res.json()
        tracks = data.get("toptracks", {}).get("track", [])

        # Get artist info for tags/genre
        info_res = requests.get(LASTFM_BASE, params={
            "method": "artist.getinfo",
            "artist": artist,
            "api_key": LASTFM_API_KEY,
            "format": "json"
        }, timeout=5)
        info_data = info_res.json()
        tags = info_data.get("artist", {}).get("tags", {}).get("tag", [])
        tag_names = [t["name"].lower() for t in tags[:5]] if tags else []

        # Infer genre from tags
        genre = infer_genre(tag_names, artist)
        region = infer_region(tag_names, artist)

        results = []
        for i, track in enumerate(tracks):
            results.append({
                "track_id": f"lfm_{artist.lower().replace(' ', '_')}_{i}",
                "artist": track.get("name", artist),
                "title": track.get("name", "Unknown"),
                "genre": genre,
                "energy": "medium",
                "bpm_range": "90-110",
                "region": region,
                "era": "2020s",
                "mood": "vibe",
                "set_position": "warm-up",
                "source": "lastfm",
                "tags": tag_names
            })
        return results
    except Exception as e:
        print(f"Last.fm error: {e}")
        return []

def infer_genre(tags, artist):
    """Infer genre from Last.fm tags"""
    genre_map = {
        "afrobeats": "afrobeats", "afropop": "afrobeats", "afro": "afrobeats",
        "dancehall": "dancehall", "reggae": "dancehall", "ragga": "dancehall",
        "hip-hop": "hip-hop", "hip hop": "hip-hop", "rap": "hip-hop", "trap": "hip-hop",
        "r&b": "r&b", "rnb": "r&b", "soul": "r&b", "neo soul": "r&b",
        "soca": "soca", "calypso": "soca",
        "amapiano": "amapiano", "afro house": "amapiano",
        "pop": "pop", "dance": "pop",
    }
    for tag in tags:
        for key, val in genre_map.items():
            if key in tag:
                return val

    # Artist-based fallback
    afrobeats_artists = ["burna boy", "wizkid", "davido", "tems", "rema", "ckay",
                          "omah lay", "asake", "ayra starr", "fireboy", "joeboy",
                          "kizz daniel", "mayorkun", "olamide", "phyno"]
    dancehall_artists = ["vybz kartel", "popcaan", "alkaline", "sean paul", "beenie man",
                          "mavado", "skillibeng", "shenseea", "dexta daps", "konshens",
                          "busy signal", "chronixx", "protoje"]
    artist_lower = artist.lower()
    if any(a in artist_lower for a in afrobeats_artists):
        return "afrobeats"
    if any(a in artist_lower for a in dancehall_artists):
        return "dancehall"
    return "pop"

def infer_region(tags, artist):
    """Infer region from tags or artist"""
    nigerian_artists = ["burna boy", "wizkid", "davido", "tems", "rema", "ckay",
                         "omah lay", "asake", "ayra starr", "fireboy", "joeboy",
                         "kizz daniel", "mayorkun", "olamide", "phyno"]
    jamaican_artists = ["vybz kartel", "popcaan", "alkaline", "sean paul", "beenie man",
                         "mavado", "skillibeng", "shenseea", "dexta daps", "konshens",
                         "busy signal", "chronixx", "protoje"]
    artist_lower = artist.lower()
    if any(a in artist_lower for a in nigerian_artists):
        return "nigeria"
    if any(a in artist_lower for a in jamaican_artists):
        return "jamaica"
    for tag in tags:
        if "nigeria" in tag or "naija" in tag:
            return "nigeria"
        if "jamaica" in tag:
            return "jamaica"
        if "uk" in tag or "british" in tag:
            return "uk"
        if "south africa" in tag:
            return "south-africa"
    return "us"

@app.get("/api/search")
def search(q: str):
    # First search local seed
    tracks = load_tracks()
    q_lower = q.lower()
    local_matches = [t for t in tracks if
                     q_lower in t['artist'].lower() or
                     q_lower in t['title'].lower() or
                     q_lower in t['genre'].lower()]

    # Then search Last.fm
    lastfm_results = lastfm_search(q) if LASTFM_API_KEY else []

    # Combine, local first
    all_results = local_matches + [r for r in lastfm_results
                                    if r['track_id'] not in {t['track_id'] for t in local_matches}]
    return {"results": all_results[:8]}

@app.get("/api/pathway/{track_id}")
def pathway(track_id: str, steps: int = 5):
    tracks = load_tracks()

    # Check if it's a Last.fm track (not in local seed)
    anchor = next((t for t in tracks if t['track_id'] == track_id), None)

    if not anchor:
        # Rebuild from Last.fm data passed via track_id pattern
        return {"error": "Track not in local seed — pathway uses local seed tracks"}

    pathway_result = []
    used_ids = {track_id}
    current = anchor

    for _ in range(steps):
        candidates = [t for t in tracks if t['track_id'] not in used_ids]
        if not candidates:
            break
        scored = sorted(
            [(score_candidate(current, c), c) for c in candidates],
            key=lambda x: -x[0][0]
        )
        best_score, best_reasons = scored[0][0]
        best = scored[0][1]
        pathway_result.append({
            "track": best,
            "score": best_score,
            "reasons": best_reasons
        })
        used_ids.add(best['track_id'])
        current = best

    return {"anchor": anchor, "pathway": pathway_result}

@app.get("/api/pathway/dynamic")
def dynamic_pathway(artist: str, steps: int = 5):
    """Generate pathway starting from any artist via Last.fm"""
    lastfm_tracks = lastfm_search(artist)
    if not lastfm_tracks:
        return {"error": "Artist not found"}

    anchor = lastfm_tracks[0]
    local_tracks = load_tracks()

    pathway_result = []
    used_ids = {anchor['track_id']}
    current = anchor

    for _ in range(steps):
        candidates = [t for t in local_tracks if t['track_id'] not in used_ids]
        if not candidates:
            break
        scored = sorted(
            [(score_candidate(current, c), c) for c in candidates],
            key=lambda x: -x[0][0]
        )
        best_score, best_reasons = scored[0][0]
        best = scored[0][1]
        pathway_result.append({
            "track": best,
            "score": best_score,
            "reasons": best_reasons
        })
        used_ids.add(best['track_id'])
        current = best

    return {"anchor": anchor, "pathway": pathway_result}

app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")
