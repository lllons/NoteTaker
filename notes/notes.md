# NoteTaker developer notes

This file is the short operational guide for developers who need to change, run, or debug NoteTaker.

## What the application is

NoteTaker is a local-first Python 3.10+ application:

- `FastAPI` and `uvicorn` serve the embedded browser UI and HTTP/WebSocket API.
- Browser microphone audio is sent as signed 16-bit PCM over `/ws`.
- The `listening/` package resamples browser audio to 16 kHz, runs Silero VAD, and emits partial/final utterances.
- `notetaker/transcription.py` loads faster-whisper on **CPU only** and converts decoder output into timestamped `TranscriptSegment` objects.
- `notetaker/pipeline.py` runs deterministic, source-cited extraction and saves `KnowledgeNote` objects.
- `notetaker/storage.py` stores complete notes in SQLite under `data/knowledge.sqlite3`.
- `notetaker/web.py` contains the complete browser page as an embedded HTML string.

The normal path is:

```text
microphone
  -> browser AudioContext / PCM
  -> WebSocket /ws
  -> listening.audio.resample_audio
  -> listening.audio.Segmenter + Silero VAD
  -> TranscriptionRuntime.transcribe
  -> KnowledgePipeline.create_note
  -> KnowledgeStore.save
  -> browser transcript + searchable note
```

## Model choice and CPU behavior

The checked-in default is the full `large-v3` Whisper model for both final and partial decoding. It is loaded with:

```python
WhisperModel(repo, device="cpu", compute_type="int8", cpu_threads=threads)
```

There is no GPU requirement or GPU fallback. The web page exposes five selectable CPU profiles. Model 1 is the existing `large-v3` behavior; Models 2–5 keep the same largest standard Whisper checkpoint but increase final beam width and fallback passes. That is the honest way to make Model 5 more thorough in this stack: `large-v3` is already the largest standard checkpoint, so there is no larger official Whisper weight to download.

| Profile | Beam | Temperature passes | Intended use |
| --- | ---: | ---: | --- |
| `model-1` | 8 | 1 | Existing behavior |
| `model-2` | 10 | 1 | Careful decoding |
| `model-3` | 12 | 2 | High accuracy |
| `model-4` | 16 | 3 | Difficult/noisy speech |
| `model-5` | 20 | 4 | Maximum CPU accuracy |

The selector is applied when a new WebSocket capture starts and is disabled during capture. It is stored in the browser's local storage for convenience. All profiles report `device="cpu"` and `compute_type="int8"`; only the final decoding breadth changes. The draft path uses a smaller beam and reuses the same loaded bundle, avoiding a second large model in memory.

Expect the following:

- The first capture downloads a large model cache from Hugging Face and can take a long time.
- CPU large-v3 can be slower than real time, especially on older or low-power machines.
- Model 5 can take much longer to finish an utterance. This is expected; final audio is not discarded.
- Model status is available at `GET /api/health`, the five choices at `GET /api/models`, and live state is sent over the WebSocket as `model` messages.

Model checkpoint aliases are defined in `notetaker/transcription.py`; the selectable profile definitions live in `notetaker/config.py`. A custom value is still passed through to faster-whisper, so a new checkpoint can be tested with `--model` without changing the loader.

## Running locally

From the repository root, after activating the virtual environment:

```bash
python -m pip install -r requirements.txt
python NoteTaker.py --host 127.0.0.1 --port 8000
```

Open the URL printed by the CLI. The first **Start capture** initializes Silero VAD and the Whisper model. Allow the browser microphone prompt, speak, then press **Stop capture** and wait for the final note event.

The command-line options override environment/config values for the current process. `notetaker.toml` is the checked-in default source, and environment variables override that file. The browser selector is a per-browser preference for the next capture; it does not edit `notetaker.toml`. Do not put API keys in the TOML file or in committed notes.

## Listening layer

`listening/audio.py` is intentionally separate from Whisper decoding. It is the place to change capture boundaries and sample handling.

Current recall-oriented behavior:

- Browser audio is converted to finite float32 values and resampled to 16 kHz before VAD.
- A one-second pre-roll is retained so the first consonants of a phrase are not cut off.
- Speech starts after two positive frames, using a lowered VAD onset threshold plus a bounded energy hint for quiet speakers.
- An utterance remains open through about 1.4 seconds of silence, preserving pauses and trailing words.
- Utterances can run for 30 seconds before a final event is emitted.
- Partial updates are emitted about every four seconds. Slow CPU decoding is throttled in `notetaker/app.py`; stale partial tasks are skipped, but final utterances are not dropped.
- `flush()` appends the browser's final short frame instead of discarding it.
- The raw transcript keeps short words and filler. The extractor may remove obvious filler from derived note layers, but source transcript evidence should not be discarded in the STT layer.

Tuning values are in `notetaker.toml` and can be overridden with `NOTE_TAKER_*` environment variables. Lower VAD thresholds improve recall but can include more background noise. Increase `preroll_seconds` or `end_silence_seconds` when words are being clipped at boundaries; increase `partial_seconds` when large-v3 saturates the CPU.

## Transcription layer

`TranscriptionRuntime` owns two cached `ModelBundle` references (`final` and `draft`) and a lock for each model. `ensure_loaded()` is intentionally lazy until capture starts, so importing the app does not block on a multi-gigabyte model download.

Important decoding safeguards:

- `vad_filter=False` is intentional. The listening layer already found the speech boundaries; applying Whisper VAD a second time can remove quiet edge words.
- Word timestamps, language, confidence, no-speech probability, and decoder metadata are retained in the schema.
- `condition_on_previous_text=False` limits repetition cascades between independent VAD utterances.
- The prompt asks Whisper to preserve technical terms, names, numbers, URLs, paths, commands, and capitalization. `hotwords` can add domain vocabulary.
- Low-confidence segments are marked for review. Do not turn low confidence into invented corrections.
- The built-in speaker mode is `labels-only`. `Speaker 1` is a schema label, not acoustic diarization.

If changing `TranscriptSegment`, check the storage round-trip, rendering, exports, and tests because the object is used throughout the pipeline.

## Data and exports

Runtime data is local and should normally not be committed:

- `data/knowledge.sqlite3` contains saved structured notes.
- The browser searches `/api/notes` and opens complete notes through `/api/notes/{id}`.
- Exports are generated through `/api/notes/{id}/export/{format}`.
- Supported formats include Markdown, JSON, HTML, PDF, DOCX, Anki, Obsidian, and Notion-compatible Markdown.

The storage index serializes the complete note body, not only the summary, so concepts, URLs, actions, questions, people, dates, and transcript evidence remain searchable.

## Tests and checks

The offline tests intentionally do not load Whisper weights:

```bash
python -m unittest discover -s tests -v
python -m compileall -q NoteTaker.py listening notetaker tests
```

For a browser-script syntax check when Node.js is available:

```bash
node -e 'const fs=require("fs"); const t=fs.readFileSync("notetaker/web.py","utf8"); const s=t.match(/<script>([\s\S]*?)<\/script>/)[1]; new Function(s); console.log("browser script parses")'
```

A live test needs the requirements installed, network access for the first model download, a microphone, and enough CPU/RAM for large-v3. Do not use a live model download in the offline unit suite.

## Common failure points

### The page loads but Start capture does nothing

Check the browser console for JavaScript syntax errors, then check that the page is served from localhost or HTTPS. The handler must call `navigator.mediaDevices.getUserMedia()` from the button path. The status pill should show a useful error instead of remaining idle.

### Model download appears stuck

Large-v3 is intentionally slow to download and initialize. Check the selected profile, the status pill, the terminal, `/api/health`, `/api/models`, available disk space, and outbound access to the model host. A model cache may be partially present; retrying should reuse completed files. Models 2–5 normally reuse the same large-v3 cache and spend more time decoding rather than downloading another checkpoint.

### Words at the start or end are missing

Inspect the `listening/` thresholds first. Increase `preroll_seconds` and `end_silence_seconds`; do not immediately add more aggressive Whisper post-processing. Verify that `flush()` receives the final browser frame and that `vad_filter` remains disabled for already-segmented audio.

### The transcript lags

This is expected with full large-v3 on CPU, especially with Model 4 or Model 5. Switch back to Model 1–2 for quicker final events, or increase `partial_seconds`; preserve final events. Never solve lag by dropping final utterances or silently changing the selected accuracy profile.

### Speaker names are wrong

The current project does not perform acoustic diarization. It emits stable `Speaker 1` labels only. Adding real diarization requires a separate model, dependency, and privacy/performance decision; do not claim that labels-only distinguishes voices.

### Notes contain unsupported claims

The deterministic extractor is source-first. Optional provider output is stored separately as `inferred_items`. Preserve source segment IDs when adding extractors, and keep any generated interpretation clearly labeled as requiring verification.

## Change checklist

When modifying capture or STT code:

1. Keep the CPU-only loader explicit (`device="cpu"`, `compute_type="int8"`).
2. Keep raw transcript segments and timestamps intact.
3. Keep resampling before VAD and Whisper.
4. Test final-frame flushing and quiet boundary behavior.
5. Update `notetaker.toml`, the README configuration table, and this file when adding a setting.
6. Run the offline tests and compile check; run a live capture only when model/dependency resources are available.
7. Avoid committing `data/`, model caches, API keys, or machine-specific files.
