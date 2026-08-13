# NoteTaker developer notes

This file is the short operational guide for developers who need to change, run, or debug NoteTaker.

## What the application is

NoteTaker is a local-first Python 3.10+ application:

- `FastAPI` and `uvicorn` serve the embedded browser UI and HTTP/WebSocket API.
- Browser microphone audio is sent as signed 16-bit PCM over `/ws`.
- The `listening/` package resamples browser audio to 16 kHz, runs CPU Silero VAD when `onnxruntime` is available, and emits partial/final utterances.
- `notetaker/transcription.py` loads faster-whisper or an optional Transformers audio backend on **CPU only** and converts decoder output into timestamped `TranscriptSegment` objects.
- `notetaker/pipeline.py` runs deterministic, source-cited extraction and saves `KnowledgeNote` objects.
- `notetaker/storage.py` stores complete notes in SQLite under `data/knowledge.sqlite3`.
- `notetaker/web.py` contains the complete browser page as an embedded HTML string.

The normal path is:

```text
microphone
  -> browser AudioContext / PCM
  -> WebSocket /ws
  -> listening.audio.resample_audio
  -> listening.audio.Segmenter + Silero/energy VAD
  -> TranscriptionRuntime.transcribe
  -> KnowledgePipeline.create_note
  -> KnowledgeStore.save
  -> browser transcript + searchable note
```

## Model choice and CPU behavior

The checked-in default is the official `large-v3` CTranslate2 conversion. It is loaded with:

```python
WhisperModel(repo, device="cpu", compute_type="int8", cpu_threads=threads)
```

There is no GPU requirement or GPU fallback. The catalog contains twelve faster-whisper CTranslate2 choices plus three larger Transformers audio backends:

- `Qwen/Qwen3-ASR-1.7B-hf` — 1.7B dedicated multilingual ASR.
- `mistralai/Voxtral-Mini-3B-2507` — 3B audio model with a transcription mode.
- `Qwen/Qwen2-Audio-7B-Instruct` — 7B audio-language model prompted for verbatim transcription.

The three larger models use CPU float32 because they are not CTranslate2 Whisper checkpoints. Install their optional dependencies with `python -m pip install -r requirements-large-models.txt`. Each API entry includes its repository URL, parameter size, backend, precision, and quality/speed notes.

The selector is applied when a new WebSocket capture starts and is disabled during capture. Switching checkpoints closes the old model bundle before loading the new one, keeping memory bounded. The draft path uses beam 2 for faster-whisper; the Transformers adapters run the same model for partial and final segments.

The official large-v3 checkpoint is the best general choice when retaining names, numbers, and technical details matters more than latency. The `large-v3-max` entry uses the same weights with beam 20 and multiple fallback temperatures. Qwen3-ASR is the first larger model to try for multilingual speech; Voxtral is useful for long-form transcription; Qwen2-Audio is the largest option but can require 30GB or more of CPU RAM in float32. The larger adapters currently preserve a timestamped segment but do not emit Whisper-style word timestamps.

Expect the following:

- The first selected model downloads its Hugging Face cache and can take a long time.
- CPU large-v3 can be slower than real time; 1.7B–7B float32 models can be dramatically slower and use substantial RAM.
- Model status is available at `GET /api/health`, the full catalog at `GET /api/models`, and live state is sent over the WebSocket as `model` messages.
- Public model repositories do not require an API key. The optional Transformers dependencies are local-only and do not contact an inference API.

Model aliases and backend adapters are in `notetaker/transcription.py`; catalog metadata is in `notetaker/config.py`.

## Running locally

From the repository root, after activating the virtual environment:

```bash
python -m pip install -r requirements.txt
# Only needed for the 1.7B, 3B, or 7B selector choices:
python -m pip install -r requirements-large-models.txt
python NoteTaker.py --host 127.0.0.1 --port 8000
```

Open the URL printed by the CLI. The first **Start capture** initializes CPU VAD and the selected Whisper model. Allow the browser microphone prompt, speak, then press **Stop capture** and wait for the `saved`/`flushed` confirmation.

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
- `flush()` appends the browser's final short frame instead of discarding it, and the WebSocket waits for all final tasks before sending `flushed`.
- Silero VAD uses `onnxruntime` on CPU. If that package is unavailable, `Vad` switches to an explicitly reported energy fallback instead of terminating the WebSocket before transcription.
- The raw transcript keeps short words and filler. The extractor may remove obvious filler from derived note layers, but source transcript evidence should not be discarded in the STT layer.

Tuning values are in `notetaker.toml` and can be overridden with `NOTE_TAKER_*` environment variables. Lower VAD thresholds improve recall but can include more background noise. Increase `preroll_seconds` or `end_silence_seconds` when words are being clipped at boundaries; increase `partial_seconds` when large-v3 saturates the CPU.

## Transcription layer

`TranscriptionRuntime` owns two cached `ModelBundle` references (`final` and `draft`) and a lock for each model. `ensure_loaded()` is intentionally lazy until capture starts, so importing the app does not block on a multi-gigabyte model download. Faster-whisper bundles use CTranslate2; the Qwen3-ASR, Voxtral, and Qwen2-Audio bundles use lazy Transformers adapters with `torch` explicitly moved to CPU.

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

The offline tests intentionally do not load Whisper weights or VAD assets:

```bash
python -m unittest discover -s tests -v
python -m compileall -q NoteTaker.py listening notetaker tests
```

For a browser-script syntax check when Node.js is available:

```bash
node -e 'const fs=require("fs"); const t=fs.readFileSync("notetaker/web.py","utf8"); const s=t.match(/<script>([\s\S]*?)<\/script>/)[1]; new Function(s); console.log("browser script parses")'
```

A live test needs the requirements installed (including `onnxruntime`; also install `requirements-large-models.txt` for the three larger choices), network access for the first model download, a microphone, and enough CPU/RAM for the selected checkpoint. Do not use a live model download in the offline unit suite.

## Common failure points

### The page loads but Start capture does nothing

Check the browser console for JavaScript syntax errors, then check that the page is served from localhost or HTTPS. The handler must call `navigator.mediaDevices.getUserMedia()` from the button path. The status pill should show a useful error instead of remaining idle.

### Model download appears stuck

All models are intentionally slow to download and initialize. Check the selected model, the status pill, the terminal, `/api/health`, `/api/models`, available disk space, and outbound access to Hugging Face. For Qwen3-ASR, Voxtral, or Qwen2-Audio, install `requirements-large-models.txt` first. A model cache may be partially present; retrying should reuse completed files.

### Capture says no speech was detected

Verify the level meter moves. Install the checked-in requirements so `onnxruntime` is present; the status message reports `backend: silero` when the real CPU VAD is active. If it reports `energy-fallback`, audio is still accepted, but installing `onnxruntime` improves speech boundaries. Press **Stop capture** and wait for `flushed`; it reports whether a note was saved.

### Words at the start or end are missing

Inspect the `listening/` thresholds first. Increase `preroll_seconds` and `end_silence_seconds`; do not immediately add more aggressive Whisper post-processing. Verify that `flush()` receives the final browser frame and that `vad_filter` remains disabled for already-segmented audio.

### The transcript lags

This is expected with full large-v3 on CPU and even more pronounced with the 1.7B–7B Transformers choices. Choose Qwen3-ASR for the best larger-model starting point, or choose a smaller CTranslate2/turbo/distilled checkpoint for quicker final events; preserve final events and do not silently discard audio.

### Speaker names are wrong

The current project does not perform acoustic diarization. It emits stable `Speaker 1` labels only. Adding real diarization requires a separate model, dependency, and privacy/performance decision; do not claim that labels-only distinguishes voices.

### Notes contain unsupported claims

The deterministic extractor is source-first. Optional provider output is stored separately as `inferred_items`. Preserve source segment IDs when adding extractors, and keep any generated interpretation clearly labeled as requiring verification.

## Change checklist

When modifying capture or STT code:

1. Keep the CPU-only loader explicit: CTranslate2 uses `device="cpu"`, `compute_type="int8"`; Transformers adapters use CPU float32 and never `device_map="cuda"`.
2. Keep raw transcript segments and timestamps intact.
3. Keep resampling before VAD and Whisper.
4. Test final-frame flushing and quiet boundary behavior.
5. Update `notetaker.toml`, the README configuration table, and this file when adding a setting.
6. Run the offline tests and compile check; run a live capture only when model/dependency resources are available.
7. Avoid committing `data/`, model caches, API keys, or machine-specific files.
8. Keep model repository URLs and CPU-only settings accurate when changing the catalog.
