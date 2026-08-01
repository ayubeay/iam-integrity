from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json, sys, os, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from walker import load_tracks, score_candidate, next_set_position, SET_ORDER

app = FastAPI()

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"

def lastfm_similar_pool(artist: str, limit_artists: int = 8, per_artist: int = 4):
    """Tracks by artists Last.fm considers similar. Genre and region are inferred
    from tags; energy, bpm, era, mood and set position are left unset because we
    do not measure them for these sources."""
    pool = []
    try:
        res = requests.get(LASTFM_BASE, params={
            "method": "artist.getsimilar", "artist": artist,
            "api_key": LASTFM_API_KEY, "format": "json", "limit": limit_artists
        }, timeout=6)
        similars = res.json().get("similarartists", {}).get("artist", [])
    except Exception as e:
        print(f"Last.fm similar error: {e}")
        return pool

    for s in similars:
        name = s.get("name")
        if not name:
            continue
        try:
            tr = requests.get(LASTFM_BASE, params={
                "method": "artist.gettoptracks", "artist": name,
                "api_key": LASTFM_API_KEY, "format": "json", "limit": per_artist
            }, timeout=6).json().get("toptracks", {}).get("track", [])
            info = requests.get(LASTFM_BASE, params={
                "method": "artist.getinfo", "artist": name,
                "api_key": LASTFM_API_KEY, "format": "json"
            }, timeout=6).json()
            tags = info.get("artist", {}).get("tags", {}).get("tag", [])
            tag_names = [t["name"].lower() for t in tags[:5]] if tags else []
            g, r = infer_genre(tag_names, name), infer_region(tag_names, name)
            for i, t in enumerate(tr):
                pool.append({
                    "track_id": f"lfm_sim_{name.lower().replace(' ', '_')}_{i}",
                    "artist": name, "title": t.get("name", "Unknown"),
                    "genre": g, "region": r,
                    "energy": None, "bpm_range": None, "era": None,
                    "mood": None, "set_position": None,
                    "source": "lastfm", "tags": tag_names,
                })
        except Exception:
            continue
    return pool


def lastfm_search(artist: str):
    try:
        res = requests.get(LASTFM_BASE, params={
            "method": "artist.gettoptracks",
            "artist": artist,
            "api_key": LASTFM_API_KEY,
            "format": "json",
            "limit": 15
        }, timeout=5)
        data = res.json()
        tracks = data.get("toptracks", {}).get("track", [])

        info_res = requests.get(LASTFM_BASE, params={
            "method": "artist.getinfo",
            "artist": artist,
            "api_key": LASTFM_API_KEY,
            "format": "json"
        }, timeout=5)
        info_data = info_res.json()
        tags = info_data.get("artist", {}).get("tags", {}).get("tag", [])
        tag_names = [t["name"].lower() for t in tags[:5]] if tags else []

        # Get canonical artist name from API response
        canonical_artist = info_data.get("artist", {}).get("name", artist)

        genre = infer_genre(tag_names, artist)
        region = infer_region(tag_names, artist)

        results = []
        for i, track in enumerate(tracks):
            # Fix: artist from canonical name, title from track.name
            track_artist = track.get("artist", {}).get("name", canonical_artist) if isinstance(track.get("artist"), dict) else canonical_artist
            results.append({
                "track_id": f"lfm_{artist.lower().replace(' ', '_')}_{i}",
                "artist": track_artist,
                "title": track.get("name", "Unknown"),
                "genre": genre,
                "energy": None,          # not measured for lastfm sources
                "bpm_range": None,       # not measured
                "region": region,
                "era": None,             # not measured
                "mood": None,            # not measured
                "set_position": None,    # not measured
                "source": "lastfm",
                "tags": tag_names
            })
        return results
    except Exception as e:
        print(f"Last.fm error: {e}")
        return []

def infer_genre(tags, artist):
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
    tracks = load_tracks()
    q_lower = q.lower()
    local_matches = [t for t in tracks if
                     q_lower in t['artist'].lower() or
                     q_lower in t['title'].lower() or
                     q_lower in t['genre'].lower()]
    lastfm_results = lastfm_search(q) if LASTFM_API_KEY else []
    all_results = local_matches + [r for r in lastfm_results
                                    if r['track_id'] not in {t['track_id'] for t in local_matches}]
    return {"results": all_results[:8]}

@app.get("/api/pathway/{track_id}")
def pathway(track_id: str, steps: int = 5):
    tracks = load_tracks()
    anchor = next((t for t in tracks if t['track_id'] == track_id), None)
    if not anchor:
        return {"error": "Track not in local seed"}

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
        # avoid one similar artist dominating the whole walk
        artist_counts = {}
        for p in pathway_result:
            k = (p['track'].get('artist') or '').lower()
            artist_counts[k] = artist_counts.get(k, 0) + 1
        pick = None
        for sc, cand in scored:
            k = (cand.get('artist') or '').lower()
            if artist_counts.get(k, 0) < 2:
                pick = (sc, cand)
                break
        if pick is None:
            pick = scored[0]
        best_score, best_reasons = pick[0]
        best = pick[1]
        pathway_result.append({"track": best, "score": best_score, "reasons": best_reasons})
        used_ids.add(best['track_id'])
        current = best

    return {"anchor": anchor, "pathway": pathway_result}

@app.get("/api/dynamic_pathway")
def dynamic_pathway(artist: str, steps: int = 5, title: str = ""):
    lastfm_tracks = lastfm_search(artist)
    if not lastfm_tracks:
        return {"error": "Artist not found"}

    # honour the track the user actually selected, not just the first result
    anchor = lastfm_tracks[0]
    if title:
        want = title.strip().lower()
        for t in lastfm_tracks:
            if (t.get("title") or "").strip().lower() == want:
                anchor = t
                break
    # candidates: the curated seed catalog plus artists Last.fm considers similar
    local_tracks = load_tracks()
    pool = local_tracks + lastfm_similar_pool(anchor.get("artist", artist))

    pathway_result = []
    used_ids = {anchor['track_id']}
    seen_titles = {(anchor.get('artist','') + '::' + anchor.get('title','')).lower()}
    current = anchor

    for _ in range(steps):
        candidates = [t for t in pool
                      if t['track_id'] not in used_ids
                      and (t.get('artist','') + '::' + t.get('title','')).lower() not in seen_titles]
        if not candidates:
            break
        scored = sorted(
            [(score_candidate(current, c), c) for c in candidates],
            key=lambda x: -x[0][0]
        )
        # avoid one similar artist dominating the whole walk
        artist_counts = {}
        for p in pathway_result:
            k = (p['track'].get('artist') or '').lower()
            artist_counts[k] = artist_counts.get(k, 0) + 1
        pick = None
        for sc, cand in scored:
            k = (cand.get('artist') or '').lower()
            if artist_counts.get(k, 0) < 2:
                pick = (sc, cand)
                break
        if pick is None:
            pick = scored[0]
        best_score, best_reasons = pick[0]
        best = pick[1]
        pathway_result.append({"track": best, "score": best_score, "reasons": best_reasons})
        used_ids.add(best['track_id'])
        seen_titles.add((best.get('artist','') + '::' + best.get('title','')).lower())
        current = best

    return {"anchor": anchor, "pathway": pathway_result}


@app.get("/app")
def serve_app():
    return FileResponse(str(Path(__file__).parent / "static" / "app.html"))


# ── Feedback (beta) ─────────────────────────────────────────
import json as _json, os as _os, time as _time
from fastapi import Body, HTTPException

_FEEDBACK_FILE = Path(__file__).parent.parent / "data" / "feedback.jsonl"

@app.post("/api/feedback")
def submit_feedback(payload: dict = Body(...)):
    entry = {
        "category": str(payload.get("category", ""))[:40],
        "comment": str(payload.get("comment", ""))[:2000],
        "context": payload.get("context", {}),
        "received": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    }
    if not entry["category"]:
        raise HTTPException(400, "category required")
    line = _json.dumps(entry, ensure_ascii=False)
    print("FEEDBACK " + line, flush=True)  # survives in Railway logs even across deploys
    try:
        _FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_FEEDBACK_FILE, "a") as f:
            f.write(line + "\n")
    except Exception as e:
        print("FEEDBACK_FILE_ERROR " + str(e), flush=True)
    return {"ok": True}

@app.get("/api/feedback")
def read_feedback(key: str = ""):
    expected = _os.environ.get("FEEDBACK_KEY", "")
    if not expected or key != expected:
        raise HTTPException(403, "set FEEDBACK_KEY env var and pass ?key=")
    if not _FEEDBACK_FILE.exists():
        return {"count": 0, "items": []}
    items = [_json.loads(l) for l in open(_FEEDBACK_FILE) if l.strip()]
    return {"count": len(items), "items": items}


# -- YouTube video-id resolver (cached; no key = no embed, app falls back to redirect) --
import urllib.request as _url, urllib.parse as _parse
_YT_CACHE = {}
_YT_FILE = Path(__file__).parent.parent / "data" / "yt_cache.json"
try:
    _YT_CACHE = _json.loads(_YT_FILE.read_text())
except Exception:
    _YT_CACHE = {}

@app.get("/api/resolve/youtube")
def resolve_youtube(artist: str = "", title: str = ""):
    ck = (artist + "::" + title).lower().strip()
    if not ck.strip(":"):
        raise HTTPException(400, "artist and title required")
    if ck in _YT_CACHE:
        return {"video_id": _YT_CACHE[ck], "cached": True}
    key = _os.environ.get("YOUTUBE_API_KEY", "")
    if not key:
        raise HTTPException(503, "embedded playback not configured")
    q = _parse.urlencode({
        "part": "snippet", "type": "video", "videoEmbeddable": "true",
        "maxResults": "1", "q": artist + " " + title, "key": key,
    })
    try:
        with _url.urlopen("https://www.googleapis.com/youtube/v3/search?" + q, timeout=8) as r:
            data = _json.loads(r.read().decode())
        items = data.get("items", [])
        if not items:
            raise HTTPException(404, "no embeddable video found")
        vid = items[0]["id"]["videoId"]
        _YT_CACHE[ck] = vid
        try:
            _YT_FILE.parent.mkdir(parents=True, exist_ok=True)
            _YT_FILE.write_text(_json.dumps(_YT_CACHE))
        except Exception:
            pass
        return {"video_id": vid, "cached": False}
    except HTTPException:
        raise
    except Exception as e:
        print("YT_RESOLVE_ERROR " + str(e), flush=True)
        raise HTTPException(502, "resolver unavailable")

app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")
