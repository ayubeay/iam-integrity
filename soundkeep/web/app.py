from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from walker import load_tracks, score_candidate, next_set_position, SET_ORDER

app = FastAPI()

@app.get("/api/search")
def search(q: str):
    tracks = load_tracks()
    q_lower = q.lower()
    matches = [t for t in tracks if
               q_lower in t['artist'].lower() or
               q_lower in t['title'].lower() or
               q_lower in t['genre'].lower()]
    return {"results": matches[:5]}

@app.get("/api/pathway/{track_id}")
def pathway(track_id: str, steps: int = 5):
    tracks = load_tracks()
    anchor = next((t for t in tracks if t['track_id'] == track_id), None)
    if not anchor:
        return {"error": "Track not found"}

    pathway = []
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
        pathway.append({
            "track": best,
            "score": best_score,
            "reasons": best_reasons
        })
        used_ids.add(best['track_id'])
        current = best

    return {"anchor": anchor, "pathway": pathway}

app.mount("/", StaticFiles(directory="web/static", html=True), name="static")
