# Reserve: Lyrics Intelligence & Narrative Retrieval

**Status:** Reserved. Do not implement until the library, metadata, search and ingestion
pipelines are stable.

## Purpose
Lyrics are not static text attached to a song. They are a knowledge layer describing
narrative, emotion, theme, relationship, perspective, imagery and symbol.

Let users find music when they cannot remember artist, title, album or the exact words.

## Doctrine
People rarely remember songs the way databases store them. They remember how it felt, what
happened in the story, one misheard line, a scene from the video, who was speaking, what it
was about. Retrieve from human memory rather than requiring perfect metadata.

## Retrieval modes
Exact lyric. Approximate lyric - forgotten words, misheard lines, reordered phrases.
Narrative: "a guy gets ready for a first date". Scene: driving, rain, airport, hospital.
Emotion: nervous, nostalgic, guilty, grateful. Perspective: male narrator, duet, internal
monologue. Relationship: first love, parent, betrayal, reunion. Theme: addiction, recovery,
sacrifice, redemption. Symbol: rain, ocean, fire, train, stars.

**Music video narrative** indexed separately, because people remember scenes, clothing,
locations and actions rather than words: "he combs his hair, buttons his shirt, gets in his
car, nervous before meeting a girl."

## Knowledge graph
Lyrics as structured nodes - characters, relationships, events, emotions, locations, time,
objects, symbols, narrative arcs, themes - connected to the wider SoundKeep graph.

## Retrieval receipts
Every semantic match explains itself: which narrative, emotion, perspective and scene
matched, how strongly, and the overall confidence. Users should understand why a song
came back.

## Copyright doctrine
Operate on semantic representations, embeddings and indexed concepts - not on reproduced
lyric text. Where licensing permits, display lyrics per the rights granted. Otherwise
return references, metadata and explanation receipts without reproducing protected text.

## Guiding principle
People remember songs through stories, emotions, scenes, symbols and imperfect memories.
Retrieve music the way memory works, while respecting copyright through semantic
understanding rather than raw reproduction.
