# NoteTaker — high-fidelity knowledge capture

NoteTaker turns lectures, meetings, podcasts, tutorials, interviews, and conversations into a searchable evidence ledger. It keeps the source transcript alongside timestamps, confidence, speaker metadata, structured facts, study material, and portable exports.

> **Design principle:** preserve useful information first. NoteTaker removes only obvious filler, exact repetition, and false starts; it does not silently compress reasoning, examples, caveats, numbers, technical terms, or implementation details.

## Quickstart

### 1. Prerequisites

You need:

- Python **3.10 or newer**.
- `pip` and the Python virtual-environment module.
- A modern browser with microphone support.
- Network access the first time you use a Whisper model, so `faster-whisper` can download model files from Hugging Face.
- Enough memory, CPU, and disk space for the selected model. The default is intended for a practical CPU setup; larger models require substantially more resources.

NoteTaker has no required API key. Transcription and deterministic extraction run locally. An optional OpenAI-compatible provider can enrich extraction when explicitly configured; see [Optional LLM enrichment](#optional-llm-enrichment).

### 2. Create and activate a virtual environment

From the repository root, create an isolated environment:

```bash
python -m venv .venv
```

Activate it before installing dependencies:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

When the environment is active, your shell prompt normally includes `(.venv)`. To leave it later, run:

```bash
deactivate
```

### 3. Install dependencies

Upgrade packaging tools and install the project requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements are intentionally small:

- `faster-whisper` — local Whisper transcription and CTranslate2 model loading.
- `onnxruntime` — CPU Silero VAD used to find speech boundaries.
- `fastapi` — HTTP and WebSocket application server.
- `uvicorn[standard]` — ASGI server.
- `numpy` — PCM audio conversion and signal handling.

### 4. Start NoteTaker

Run the compatibility entrypoint from the repository root:

```bash
python NoteTaker.py --host 0.0.0.0 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. If the app is running in a hosted preview or container, use the preview URL supplied by that environment rather than replacing `0.0.0.0` with a public address.

The first capture can take longer because the selected Whisper model and CPU VAD assets are loaded and downloaded when capture starts. Once the page opens:

1. Choose one of the **CPU transcription model** choices. The selection is remembered in this browser and applies to the next capture. Start with **Large-v3 · Best overall** or **Large-v3 · Maximum CPU decode** when retaining every detail matters most.
2. Press **Start capture**.
3. Allow microphone access when the browser asks.
4. Speak normally or play audio near the selected microphone.
5. Press **Stop capture** when finished.
6. Wait for the final segment to flush and for the live note to be saved.
7. Search the **Knowledge base** panel or open an export from the saved note.

Audio is processed locally by default. The browser sends PCM audio to the local NoteTaker process over its WebSocket connection; no external transcription service is contacted unless you configure an optional enrichment provider.

### 5. Run the offline tests

The tests do not download Whisper models or require a provider key:

```bash
python -m unittest discover -s tests -v
```

For a syntax-only check:

```bash
python -m compileall -q NoteTaker.py notetaker tests
```

## Linux and macOS setup

The application code is platform-neutral, but Linux and macOS commonly expose `python3` rather than `python`. If `python --version` says the command is missing, use `python3` consistently for the virtual environment and all subsequent commands.

### Linux

On Debian or Ubuntu, install the system prerequisites if they are not already present:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

Then create and activate the environment with `python3`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python NoteTaker.py --host 0.0.0.0 --port 8000
```

For another Linux distribution, install its equivalent Python 3, `pip`, and virtual-environment packages. Do not run the application with `sudo`; use a user-owned checkout and `.venv` instead.

If the browser cannot use the microphone, check the browser site permission and the operating system's privacy settings. A local `http://127.0.0.1` page is normally treated as a secure browser context. A remote HTTP preview may require HTTPS for microphone access.

### macOS

Install Python 3 with the official installer or Homebrew. With Homebrew, the setup is:

```bash
brew install python
```

From the repository root, use the Homebrew/Python 3 interpreter to create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python NoteTaker.py --host 0.0.0.0 --port 8000
```

The first time the browser captures audio, allow microphone access in the browser prompt. If macOS still blocks it, open **System Settings → Privacy & Security → Microphone** and enable the browser you are using, then reload the page. If you install Python or Homebrew while a terminal is open, start a new terminal if `python3` is not found on `PATH`.

### What changes on Linux or macOS?

Usually only these details change:

| Item | Windows | Linux or macOS |
| --- | --- | --- |
| Python command | `python` | `python3` when `python` is unavailable |
| Create environment | `python -m venv .venv` | `python3 -m venv .venv` |
| Activate environment | `.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
| Stop the server | `Ctrl+C` | `Ctrl+C` |
| Microphone permission | Browser/site permission | Browser/site permission plus OS privacy permission |

Once `.venv` is activated, use `python` for the remaining commands on every platform because it points to the active environment.

## Choosing a transcription model

The web selector now lists real public Hugging Face repositories that can be loaded directly by `faster-whisper`/CTranslate2. Every choice is forced to `device="cpu"` with `compute_type="int8"`; larger choices may be very slow but preserve more detail than the small fallbacks.

| Choice | Hugging Face checkpoint | Best use |
| --- | --- | --- |
| **Large-v3 · Best overall** | `Systran/faster-whisper-large-v3` | Best general accuracy; recommended default |
| **Large-v3 · Maximum CPU decode** | `Systran/faster-whisper-large-v3` | Same weights with beam 20 and fallback passes |
| **Distil-large-v3 · High quality** | `Systran/faster-distil-whisper-large-v3` | Large distilled model with lower CPU cost |
| **Large-v3 Turbo · Faster** | `deepdml/faster-whisper-large-v3-turbo-ct2` | Faster large-model capture with a quality trade-off |
| **Medium / Medium English** | `Systran/faster-whisper-medium(.en)` | Strong lower-memory multilingual or English capture |
| **Small / Small English** | `Systran/faster-whisper-small(.en)` | Moderate CPU fallback |
| **Base / Base English** | `Systran/faster-whisper-base(.en)` | Lightweight fallback |
| **Tiny / Tiny English** | `Systran/faster-whisper-tiny(.en)` | Only for severely constrained machines |

The selector is disabled while capture is active. Changing it applies to the next WebSocket session; switching between different checkpoints releases the old model before loading the new one so CPU memory is not needlessly doubled. Public model repositories do not require an API key. The raw `openai/whisper-large-v3` Transformers repository is not listed because the current application expects a CTranslate2 conversion; the official `Systran` conversion is the compatible choice.

The checked-in defaults in `notetaker.toml` remain:

```toml
model = "large-v3"
draft_model = "large-v3"
language = "auto"
beam_size = 8
threads = 0
```

To choose a checkpoint explicitly from the command line:

```bash
python NoteTaker.py \
  --model large-v3 \
  --draft large-v3 \
  --beam 8 \
  --host 0.0.0.0 \
  --port 8000
```

Useful guidance:

- Use **Large-v3 · Best overall** first. Use **Large-v3 · Maximum CPU decode** when final accuracy matters more than wait time.
- Use `--lang en` or another known language code when the recording is consistently one language. Omit `--lang` to let Whisper detect it.
- Provide domain vocabulary so technical names are more likely to be preserved:

  ```bash
  python NoteTaker.py --hotwords "pgvector, CTranslate2, FastAPI, NoteTaker"
  ```

- Set `--threads` to a positive CPU thread count when you need to constrain resource usage. `0` lets the runtime choose.

Models are cached by faster-whisper after their first download. Check the model status pill and terminal if a large checkpoint is still downloading or initializing.

## Configuration

Runtime defaults live in `notetaker.toml`. Environment variables override values from that file, and command-line options override the model, language, hotwords, beam size, threads, host, and port for the current run.

| Setting | Environment variable | Default | Purpose |
| --- | --- | --- | --- |
| `model` | `NOTE_TAKER_MODEL` | `large-v3` | Final CPU transcription model |
| `draft_model` | `NOTE_TAKER_DRAFT_MODEL` | `large-v3` | Partial/live model; same model avoids a second cache |
| `language` | `NOTE_TAKER_LANGUAGE` | automatic | Language code, or `auto` |
| `beam_size` | `NOTE_TAKER_BEAM_SIZE` | `8` | Whisper decoding breadth |
| `threads` | `NOTE_TAKER_THREADS` | `0` | CPU thread limit; `0` is automatic |
| `hotwords` | `NOTE_TAKER_HOTWORDS` | empty | Comma-separated domain terms |
| `data_dir` | `NOTE_TAKER_DATA_DIR` | `data` | SQLite and application data directory |
| `context_chars` | `NOTE_TAKER_CONTEXT_CHARS` | `1200` | Prior transcript context for live decoding |
| `max_segment_seconds` | `NOTE_TAKER_MAX_SEGMENT_SECONDS` | `30` | Maximum VAD segment length |
| `min_segment_seconds` | `NOTE_TAKER_MIN_SEGMENT_SECONDS` | `0.15` | Minimum audio segment length |
| `vad_on_threshold` | `NOTE_TAKER_VAD_ON_THRESHOLD` | `0.35` | Recall-first speech start threshold |
| `vad_off_threshold` | `NOTE_TAKER_VAD_OFF_THRESHOLD` | `0.20` | Recall-first speech continuation threshold |
| `preroll_seconds` | `NOTE_TAKER_PREROLL_SECONDS` | `1.0` | Audio retained before speech onset |
| `end_silence_seconds` | `NOTE_TAKER_END_SILENCE_SECONDS` | `1.4` | Silence required to close an utterance |
| `partial_seconds` | `NOTE_TAKER_PARTIAL_SECONDS` | `4.0` | Interval between draft updates |
| `provider_timeout` | `NOTE_TAKER_PROVIDER_TIMEOUT` | `45` | Optional provider request timeout |
| `diarization` | `NOTE_TAKER_DIARIZATION` | `labels-only` | Speaker-label mode |

For example, a low-resource Linux or macOS run can be started with:

```bash
NOTE_TAKER_MODEL=medium.en \
NOTE_TAKER_DRAFT_MODEL=medium.en \
NOTE_TAKER_THREADS=4 \
python NoteTaker.py --host 0.0.0.0 --port 8000
```

The `data/knowledge.sqlite3` database is created automatically on first use. Keep the `data/` directory if you want to preserve notes between runs, and back it up if the notes are important.

## Optional LLM enrichment

No API key is required for the default local deterministic pipeline. To enable additional structured enrichment, configure all three values for an OpenAI-compatible JSON chat endpoint:

```text
NOTE_TAKER_LLM_BASE_URL=https://provider.example/v1
NOTE_TAKER_LLM_API_KEY=<secret>
NOTE_TAKER_LLM_MODEL=<json-capable-model>
```

For a one-session launch on Linux or macOS:

```bash
export NOTE_TAKER_LLM_BASE_URL="https://provider.example/v1"
export NOTE_TAKER_LLM_API_KEY="your-key"
export NOTE_TAKER_LLM_MODEL="your-json-capable-model"
python NoteTaker.py --host 0.0.0.0 --port 8000
```

The provider receives the transcript and local facts and is instructed to return evidence-linked JSON only. Provider output is kept in `inferred_items`; it does not replace source-backed transcript facts. Never commit API keys to the repository. In managed environments, place secrets in the environment's secret or Keys/API settings rather than in `notetaker.toml`.

## What NoteTaker captures

The pipeline is designed for retention rather than aggressive summarization:

1. **Streaming transcription** — faster-whisper, VAD boundaries, word timestamps, language metadata, no-speech probability, log probability, and confidence are retained.
2. **Semantic organization** — related utterances are grouped into topic-aware segments while preserving the original transcript segments as evidence.
3. **Structured extraction** — concepts, definitions, explanations, examples, analogies, formulas, statistics, code, commands, entities, dates, resources, action items, decisions, and open questions are extracted conservatively.
4. **Four note layers** — executive summary, hierarchical detailed notes, near-verbatim reference notes, and student-oriented study notes.
5. **Knowledge graph and timeline** — relationships retain source segment IDs, confidence, and chronological evidence.
6. **Flashcards** — definitions, concepts, formulas, processes, comparisons, and terminology can be exported as Anki-compatible TSV.
7. **Retrieval** — SQLite stores complete note JSON and searchable text. Queries return timestamped evidence and do not invent unsupported answers.

### Speaker labels and accuracy safeguards

Every transcript segment includes `speaker` and `speaker_confidence`. The built-in `labels-only` mode provides stable labels such as `Speaker 1`; it does **not** claim to distinguish voices acoustically. Low-confidence regions are marked for review and copied into `uncertain_regions`. Inferred or provider-generated material is labeled separately from transcript facts.

## API and exports

With the server running, the main endpoints are:

- `GET /api/health` — service, provider, selected model, and model status.
- `GET /api/models` — the Hugging Face CPU model catalog shown by the web selector.
- `GET /api/notes?query=vector&limit=20` — search saved notes and extracted evidence.
- `GET /api/notes/{id}` — retrieve a complete structured note.
- `GET /api/notes/{id}/export/{format}` — export as `md`, `pdf`, `docx`, `html`, `json`, `anki`, `obsidian`, or `notion`.
- `POST /api/notes` — import text and generate a note.
- `POST /api/query` — retrieve timestamped evidence for a question without unsupported answer generation.
- `WS /ws` — stream 16 kHz PCM audio and receive live transcript events.

Example text import:

```bash
curl -X POST http://127.0.0.1:8000/api/notes \
  -H 'Content-Type: application/json' \
  -d '{"title":"Vector database study","source_type":"import","text":"Vector databases store embeddings. An embedding is a numeric representation of meaning."}'
```

Example evidence query:

```bash
curl -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What was said about embeddings?"}'
```

Markdown exports include a table of contents, collapsible reference transcript and flashcards, timestamped sections, callouts, code blocks, graph JSON, timeline, action items, decisions, questions, and accuracy review sections. `obsidian` and `notion` use the same portable Markdown representation.

## Repository structure

```text
NoteTaker.py                 compatibility CLI (`python NoteTaker.py`)
listening/
  audio.py                  PCM resampling, Silero/energy VAD, recall-first boundaries
notetaker/
  app.py                     FastAPI routes, WebSocket capture, retrieval API
  config.py                  typed TOML and environment configuration
  transcription.py           CPU Whisper loading, words, confidence, language
  extractor.py               conservative semantic and fact extraction
  pipeline.py                incremental note orchestration
  models.py                  typed transcript and knowledge schema
  storage.py                 SQLite persistence and search
  rendering.py               Markdown, HTML, JSON, Anki, PDF, DOCX exports
  provider.py                optional OpenAI-compatible LLM plugin boundary
  web.py                     embedded capture and knowledge-base UI
tests/test_knowledge.py      offline knowledge-pipeline tests
tests/test_model_profiles.py offline CPU model-selector tests
notes/notes.md               developer guide and operational notes
notetaker.toml               checked-in runtime defaults
requirements.txt             Python dependencies
data/knowledge.sqlite3       created on first run
```

## Example: before vs. after

**Before**

```text
Vector databases store embeddings. An embedding is a numeric representation of meaning. The action item is to compare pgvector and a hosted service by Friday.
```

**After**

```markdown
# Vector database study

## Executive summary
- Vector databases store embeddings.
- An embedding is a numeric representation of meaning.
- Action item: compare pgvector and a hosted service by Friday.

## Detailed notes
### Definitions
- **Embedding** — a numeric representation of meaning.

### Action items
> [!todo]
> Compare pgvector and a hosted service by Friday.

## Reference transcript
> **00:00:00–00:00:08 · Speaker 1 · confidence 94%**
> Vector databases store embeddings. An embedding is a numeric representation of meaning.
```

The second form remains auditable: extracted items can be traced to source segments instead of becoming unsupported free-floating summaries.

## Performance and privacy notes

- The first VAD/model load and download are the most expensive startup steps; later sessions reuse cached weights.
- Large-v3 runs on CPU with int8 weights; expect high memory use and slower-than-real-time decoding on many machines.
- The selector exposes actual Hugging Face CTranslate2 checkpoints plus a maximum-decode large-v3 profile; only one checkpoint is kept loaded at a time.
- The draft path uses a small beam on the selected model, and stale partial work is dropped so a slow CPU cannot delay final utterances.
- Model inference uses bounded worker execution so concurrent transcription does not corrupt model state.
- Each capture receives a unique note ID and the browser waits for a final `flushed` confirmation before declaring the note saved.
- SQLite keeps durable notes on disk instead of retaining every session only in memory.
- Extraction is incremental for live sessions, and the optional provider is not required for saving notes.
- Larger models and higher beam sizes generally improve recognition but increase memory use and latency.
- Local deterministic mode keeps microphone audio and notes in the local application process and `data/` directory. Configure an LLM provider only when you accept sending transcript text to that provider.

## Troubleshooting

### `python` or `python3` is not found

Install Python 3.10+ and make sure it is on `PATH`. On Linux and macOS, use `python3 -m venv .venv`; after activation, use `python` for the remaining commands.

### `No module named ...`

Activate `.venv` and reinstall with the same interpreter that starts the app:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python NoteTaker.py --host 0.0.0.0 --port 8000
```

### The browser cannot access the microphone

Allow the site in the browser's microphone permissions. On macOS, also enable the browser under **System Settings → Privacy & Security → Microphone**. Remote non-HTTPS pages may be blocked by the browser; use `127.0.0.1` locally or an HTTPS preview.

### Capture ends with no speech detected

Run `python -m pip install -r requirements.txt` again so `onnxruntime` is installed. The app uses Silero VAD on CPU when available and shows `energy-fallback` when it is not. If the fallback is shown, the microphone frames are still accepted, but installing `onnxruntime` gives better speech boundaries. Watch the level meter and wait for the `saved` confirmation after pressing **Stop capture**.

### The first run is slow or appears idle

The server prints its local URL immediately, then the selected model and CPU VAD initialize when you press **Start capture**. Check the model/VAD status pill, the final `flushed` message, and the terminal. The first large-v3 download can take a while.

### Transcription quality is poor

The default checkpoint is already the largest configured model. Start with **Large-v3 · Best overall**, use **Large-v3 · Maximum CPU decode** when wait time does not matter, and choose a smaller/turbo/distilled option only when CPU time or RAM is limiting. Set a known `--lang`, pass domain-specific `--hotwords`, and check microphone input level. The listening layer keeps a one-second pre-roll, longer pauses, and trailing partial frames; low-confidence segments remain visible for manual review rather than being silently “corrected.”

### The port is already in use

Choose another local port:

```bash
python NoteTaker.py --host 0.0.0.0 --port 8001
```

Then open `http://127.0.0.1:8001`.

### Optional provider requests fail

The local pipeline continues to work without enrichment. Verify that `NOTE_TAKER_LLM_BASE_URL`, `NOTE_TAKER_LLM_API_KEY`, and `NOTE_TAKER_LLM_MODEL` are all set, that the endpoint supports `/chat/completions` and JSON response format, and that the configured timeout is long enough.

## License

NoteTaker is released under the MIT License. See [LICENSE](LICENSE).
