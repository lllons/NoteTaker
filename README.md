# NoteTaker — high-fidelity knowledge capture

NoteTaker turns lectures, meetings, podcasts, tutorials, and conversations into a searchable evidence ledger. It keeps the original transcript alongside structured facts, timestamps, confidence, speaker metadata, study material, and exportable notes.

> **Design principle:** preserve useful information first; compress only filler, repetition, and false starts. Every generated or uncertain item is labeled.

## What changed

The old single-file live transcript is now a modular local-first system:

```text
NoteTaker.py                 compatibility CLI (`python NoteTaker.py`)
notetaker/
  app.py                     FastAPI routes, WebSocket capture, retrieval API
  config.py                  typed TOML + environment configuration
  transcription.py           faster-whisper, VAD, words, confidence, language
  extractor.py               conservative semantic/fact extraction
  pipeline.py                incremental note orchestration
  models.py                  typed transcript and knowledge schema
  storage.py                 SQLite persistence and search
  rendering.py               Markdown, HTML, JSON, Anki, PDF, DOCX exports
  provider.py                optional OpenAI-compatible LLM plugin boundary
  web.py                     redesigned capture and knowledge-base UI
tests/test_knowledge.py      offline unit tests
notetaker.toml               checked-in runtime defaults
requirements.txt             Python dependencies
data/knowledge.sqlite3       created on first run (ignored by deployment policy)
```

## Local setup

```bash
python -m pip install -r requirements.txt
python NoteTaker.py --host 0.0.0.0 --port 8000
```

Open the preview, press **Start capture**, and grant microphone access. Whisper model weights are downloaded from Hugging Face on first use. The default `small.en` + `base.en` pair is practical on CPU. For maximum accuracy on a capable machine:

```bash
python NoteTaker.py --model large-v3-turbo --draft base --host 0.0.0.0 --port 8000
```

Use `--lang` only when you know the language; otherwise language is detected automatically. Supply technical vocabulary with `--hotwords "pgvector, CTranslate2, NoteTaker"`.

## High-fidelity pipeline

1. **Streaming transcription** — accurate and draft faster-whisper passes run independently. Word-level timestamps, language, no-speech probability, log probability, and confidence are retained.
2. **Segmentation** — Silero VAD creates low-latency utterance boundaries. Topic terms are attached to each timestamped segment; pauses and topic terms provide semantic boundaries without deleting source evidence.
3. **Extraction** — the deterministic local extractor captures concepts, definitions, explanations, examples, analogies, formulas, statistics, code/commands, entities, dates, resources, actions, decisions, and open questions.
4. **Four note layers** — executive bullets, hierarchical detailed notes, near-verbatim reference transcript, and study prompts.
5. **Knowledge graph and timeline** — relationships are stored as `{source, relation, target, evidence_segment_ids, confidence}` and every transcript segment becomes a chronological event.
6. **Flashcards** — definitions, concepts, and formulas become Anki-compatible TSV cards.
7. **Retrieval** — SQLite stores the complete JSON note and searchable text. `/api/query` returns cited timestamp evidence and explicitly refuses to present an unsupported answer as fact.

### Speaker labels and safeguards

The schema is speaker-ready and every segment contains `speaker` and `speaker_confidence`. The built-in `labels-only` mode uses a stable `Speaker 1` label but does **not** pretend to acoustically distinguish voices. A true diarization backend (for example, a separately installed pyannote adapter with its own model terms) can implement the same boundary without changing storage or rendering.

Segments below 65% confidence are shown with a review warning and copied into `uncertain_regions`. Provider output is placed in `inferred_items` and never replaces source-backed transcript facts.

## Optional LLM plugin

No API key is required. Local deterministic extraction is the default. Any OpenAI-compatible JSON endpoint can be enabled through the Keys/API UI or environment configuration:

```text
NOTE_TAKER_LLM_BASE_URL=https://provider.example/v1
NOTE_TAKER_LLM_API_KEY=<secret>
NOTE_TAKER_LLM_MODEL=<json-capable-model>
```

The provider receives the transcript plus local facts and is instructed to preserve citations, uncertainty, technical terms, assumptions, exceptions, trade-offs, and limitations. The integration is a plugin interface, so a local model or another provider can replace it.

## API and exports

- `GET /api/health` — pipeline and provider status.
- `GET /api/notes?query=vector` — search titles and transcript evidence.
- `GET /api/notes/{id}` — complete structured JSON note.
- `GET /api/notes/{id}/export/{format}` — `md`, `pdf`, `docx`, `html`, `json`, `anki`, `obsidian`, or `notion`.
- `POST /api/notes` — import text and generate a note.
- `POST /api/query` — return timestamped retrieved evidence for a question; no unsupported answer generation.
- `WS /ws` — streaming PCM capture with live confidence-aware segment events.

Markdown exports include a table of contents, collapsible reference transcript and flashcards, callouts, code blocks, graph JSON, timeline, action items, and accuracy review sections. `obsidian` and `notion` use the same portable Markdown representation.

## Example: before vs after

**Before**

```text
Vector databases store embeddings. An embedding is a numeric representation of meaning. The action item is to compare pgvector and a hosted service by Friday.
```

**After**

```markdown
# Vector database study

## Executive summary
- Core concepts: vector, databases, store, embeddings, embedding, numeric, representation, meaning.
- [0s] Vector databases store embeddings.
- Follow-ups captured: The action item is to compare pgvector and a hosted service by Friday.

## Detailed notes
### Definitions
- **An embedding** — a numeric representation of meaning.

### Action items and decisions
- [ ] The action item is to compare pgvector and a hosted service by Friday.

## Reference transcript
> **00:00:00–00:00:08 · Speaker 1 · confidence 94%**
> Vector databases store embeddings. An embedding is a numeric representation of meaning.

> **00:00:08–00:00:15 · Speaker 1 · confidence 58%** ⚠️ low confidence
> The action item is to compare pgvector and a hosted service by Friday.
```

The second form remains auditable: each extracted item can be traced to a source segment instead of being presented as a free-floating summary.

## Performance notes

- The draft model provides live text while the accurate model processes completed utterances.
- Model inference is isolated in single-worker executors to prevent concurrent model corruption.
- SQLite writes replace large in-memory histories and keep retrieval indexed by creation time/source type.
- Extraction is incremental: every finalized live segment updates the `live-session` note.
- The first model load is the expensive step; subsequent sessions reuse the process cache.
- For higher accuracy, increase the final model and beam size; for lower latency, use `small.en`/`base.en` and fewer threads.

## Tests

The unit tests avoid model downloads and exercise fact extraction, uncertainty marking, SQLite round trips, search, and every export format:

```bash
python -m unittest discover -s tests -v
```
