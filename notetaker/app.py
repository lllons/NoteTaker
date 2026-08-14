"""FastAPI surface for live capture, durable knowledge notes, search, and export."""

from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
import math
import time
from typing import Any, Awaitable, Callable
from uuid import uuid4

import numpy as np

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import AppConfig, DEFAULT_MODEL_PROFILE_ID
from .models import TranscriptSegment
from .pipeline import KnowledgePipeline
from .provider import provider_from_config
from .rendering import render
from listening.audio import SR, Segmenter, resample_audio

from .storage import KnowledgeStore
from .transcription import TranscriptionRuntime
from .web import PAGE


logger = logging.getLogger(__name__)


class DecodeScheduler:
    """Bound live decode work and prioritize final utterances over drafts.

    Finals are never discarded. If the final buffer is full, submit() applies
    backpressure to the websocket receive loop instead of allocating another
    audio buffer. Partials have one pending slot and are deliberately dropped
    when a final is pending or another partial is already queued.
    """

    def __init__(
        self,
        process: Callable[[str, Any, int, float], Awaitable[None]],
        max_inflight: int = 2,
        max_pending_finals: int = 8,
    ) -> None:
        self.process = process
        self.max_inflight = max(1, int(max_inflight))
        self.max_pending_finals = max(1, int(max_pending_finals))
        self._condition = asyncio.Condition()
        self._finals: deque[tuple[str, Any, int, float]] = deque()
        self._partial: tuple[str, Any, int, float] | None = None
        self._workers: set[asyncio.Task[Any]] = set()
        self._active = 0
        self._closed = False
        self._last_depth = -1

    @property
    def depth(self) -> int:
        return self._active + len(self._finals) + (1 if self._partial is not None else 0)

    @property
    def active(self) -> int:
        return self._active

    @property
    def pending_finals(self) -> int:
        return len(self._finals)

    def _log_depth(self) -> None:
        depth = self.depth
        if depth == self._last_depth:
            return
        self._last_depth = depth
        logger.info(
            "decode queue depth=%d active=%d pending_finals=%d partial_pending=%s",
            depth,
            self._active,
            len(self._finals),
            self._partial is not None,
        )

    async def start(self) -> None:
        if self._workers:
            return
        for _ in range(self.max_inflight):
            task = asyncio.create_task(self._worker())
            self._workers.add(task)

    async def submit(self, event_type: str, audio: Any, sid: int, offset: float) -> bool:
        await self.start()
        event = (event_type, audio, sid, offset)
        async with self._condition:
            if self._closed:
                if event_type == "final":
                    raise RuntimeError("decode scheduler is closed; final audio was not accepted")
                logger.warning("dropped partial reason=scheduler-closed")
                return False
            if event_type == "partial":
                if self._finals:
                    logger.info("dropped partial reason=final-pending queue_depth=%d", self.depth)
                    return False
                if self._partial is not None:
                    logger.info("dropped partial reason=partial-queue-full queue_depth=%d", self.depth)
                    return False
                self._partial = event
                self._condition.notify_all()
                self._log_depth()
                return True
            while len(self._finals) >= self.max_pending_finals and not self._closed:
                await self._condition.wait()
            if self._closed:
                raise RuntimeError("decode scheduler closed before final audio was accepted")
            self._finals.append(event)
            self._condition.notify_all()
            self._log_depth()
            return True

    async def _worker(self) -> None:
        try:
            while True:
                async with self._condition:
                    while not self._finals and self._partial is None and not self._closed:
                        await self._condition.wait()
                    if self._closed and not self._finals and self._partial is None:
                        return
                    if self._finals:
                        event = self._finals.popleft()
                    else:
                        event = self._partial
                        self._partial = None
                    self._active += 1
                    self._condition.notify_all()
                    self._log_depth()
                assert event is not None
                try:
                    await self.process(*event)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("decode worker failed kind=%s", event[0])
                finally:
                    async with self._condition:
                        self._active -= 1
                        self._condition.notify_all()
                        self._log_depth()
        except asyncio.CancelledError:
            return

    async def drain(self) -> None:
        async with self._condition:
            while self._active or self._finals or self._partial is not None:
                await self._condition.wait()

    async def shutdown(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
        workers = tuple(self._workers)
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        self._workers.clear()


def empty_decode_event(
    scope: str,
    audio_seconds: float,
    metadata: dict[str, Any] | None = None,
    audio: np.ndarray | None = None,
    reason: str = "decoder returned no segments",
) -> dict[str, Any]:
    """Build the explicit client event used for every empty decode path."""
    details = metadata or {}
    values = np.asarray(audio, dtype=np.float32).reshape(-1) if audio is not None and len(audio) else None
    rms = details.get("rms")
    peak = details.get("peak")
    if rms is None:
        rms = float(np.sqrt(np.mean(values * values))) if values is not None and len(values) else 0.0
    if peak is None:
        peak = float(np.abs(values).max()) if values is not None and len(values) else 0.0
    return {
        "t": "empty",
        "scope": scope,
        "reason": reason,
        "audio_seconds": float(audio_seconds),
        "rms": float(rms),
        "peak": float(peak),
        "decode_seconds": details.get("decode_seconds"),
        "rtf": details.get("rtf"),
        "no_speech_prob": details.get("no_speech_prob"),
        "avg_logprob": details.get("avg_logprob"),
    }


config = AppConfig.from_sources()
store = KnowledgeStore(config.data_dir / "knowledge.sqlite3")
provider = provider_from_config(config)
pipeline = KnowledgePipeline(store, provider)
runtime = TranscriptionRuntime(config)
app = FastAPI(title="NoteTaker Knowledge Capture", version="2.0.0")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.get("/api/health")
def health() -> dict[str, Any]:
    status = runtime.status()
    return {
        "ok": True,
        "service": "notetaker",
        "provider": provider.name if provider else "local-deterministic",
        "model": status["profile"]["checkpoint"],
        "models": status,
        "model_options": runtime.model_options(),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    status = runtime.status()
    return {
        "selected": status["profile_id"],
        "models": runtime.model_options(),
        "device": "cpu",
        "compute_type": status["compute_type"],
    }


@app.get("/api/notes")
def notes(query: str = "", limit: int = 50) -> list[dict[str, Any]]:
    return store.search(query, limit)


@app.get("/api/notes/{note_id}")
def note(note_id: str) -> JSONResponse:
    result = store.get(note_id)
    if result is None:
        return JSONResponse({"error": "note not found"}, status_code=404)
    return JSONResponse(result)


@app.get("/api/notes/{note_id}/export/{format_name}")
def export_note(note_id: str, format_name: str) -> Response:
    raw = store.get(note_id)
    if raw is None:
        return JSONResponse({"error": "note not found"}, status_code=404)
    from .models import KnowledgeNote, TranscriptSegment, TranscriptWord, GraphEdge, TimelineEvent, Flashcard

    def rebuild(data: dict[str, Any]) -> KnowledgeNote:
        transcript = []
        for item in data.get("transcript", []):
            words = [TranscriptWord(**word) for word in item.get("words", [])]
            transcript.append(TranscriptSegment(**{**item, "words": words}))
        semantic_segments = [TranscriptSegment(**{**item, "words": [TranscriptWord(**word) for word in item.get("words", [])]}) for item in data.get("semantic_segments", [])]
        return KnowledgeNote(
            **{**data, "transcript": transcript, "semantic_segments": semantic_segments,
               "graph": [GraphEdge(**edge) for edge in data.get("graph", [])],
               "timeline": [TimelineEvent(**event) for event in data.get("timeline", [])],
               "flashcards": [Flashcard(**card) for card in data.get("flashcards", [])]}
        )

    try:
        content, content_type, suffix = render(rebuild(raw), format_name)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return Response(content, media_type=content_type, headers={"Content-Disposition": f'attachment; filename="{note_id}{suffix}"'})


@app.post("/api/notes")
async def create_note(payload: dict[str, Any]) -> JSONResponse:
    text = str(payload.get("text", "")).strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    segments = []
    cursor = 0.0
    for index, part in enumerate([item.strip() for item in text.replace("\r", "").split("\n") if item.strip()] or [text]):
        duration = max(1.0, min(15.0, len(part) / 12))
        segments.append(TranscriptSegment(f"import-{index:04d}", cursor, cursor + duration, part, 0.82, "Speaker 1", 0.0, payload.get("language")))
        cursor += duration
    result = await asyncio.to_thread(pipeline.create_note, segments, payload.get("title"), payload.get("source_type", "import"))
    return JSONResponse(result.to_dict())


@app.post("/api/query")
async def query(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question", payload.get("query", ""))).strip()
    if not question:
        return {"question": "", "answer": "Provide a concept, phrase, person, date, action, or timestamp to retrieve.", "evidence": []}
    results = store.search(question, 20)
    terms = [term.casefold() for term in question.split() if len(term) > 1 and ":" not in term]
    evidence = []
    for item in results:
        body = store.get(item["id"]) or {}
        for segment in body.get("transcript", []):
            text = str(segment.get("text", ""))
            normalized = text.casefold()
            if not terms or any(term in normalized for term in terms):
                evidence.append({"note_id": item["id"], "title": item["title"], "start": segment.get("start", 0), "end": segment.get("end", 0), "text": text, "confidence": segment.get("confidence", 0), "speaker": segment.get("speaker", "Unknown speaker"), "source_segment_id": segment.get("id")})
    return {"question": question, "answer": "Retrieved evidence only; no unsupported answer was generated.", "notes": results, "evidence": evidence[:50]}


@app.websocket("/ws")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    requested_profile = ws.query_params.get("model", DEFAULT_MODEL_PROFILE_ID)
    try:
        await asyncio.to_thread(runtime.select_model, requested_profile)
    except ValueError as exc:
        await ws.send_text(json.dumps({
            "t": "error",
            "scope": "model",
            "message": str(exc),
            "models": runtime.status(),
        }))
        await ws.close(code=1008)
        return
    selected_status = runtime.status()
    await ws.send_text(json.dumps({
        "t": "ready",
        "model": selected_status["profile"]["label"],
        "model_status": selected_status,
        "model_options": runtime.model_options(),
    }))
    segmenter: Segmenter | None = None
    session_segments: list[TranscriptSegment] = []
    session_note_id = f"live-{uuid4().hex}"
    context = ""
    elapsed = 0.0
    source_sample_rate = float(SR)
    final_lock = asyncio.Lock()
    last_note_save_at: float | None = None
    last_vad_status: dict[str, Any] | None = None
    sent_first_frame_vad = False

    async def save_note(force: bool = False) -> None:
        nonlocal last_note_save_at
        if not session_segments:
            return
        now = time.monotonic()
        interval = float(getattr(config, "note_save_interval_seconds", 20.0))
        if not force and last_note_save_at is not None and now - last_note_save_at < interval:
            logger.info(
                "debounced live note regeneration segment_count=%d next_in=%.1fs",
                len(session_segments),
                interval - (now - last_note_save_at),
            )
            return
        note = await asyncio.to_thread(
            pipeline.create_note,
            list(session_segments),
            "Live capture",
            "live",
            session_note_id,
            False,
        )
        last_note_save_at = now
        logger.info("live note save note_id=%s segment_count=%d force=%s", note.id, len(session_segments), force)
        await ws.send_text(json.dumps({"t": "note", "title": note.title, "id": note.id, "language": note.language}, ensure_ascii=False))

    async def process(event_type: str, audio: Any, sid: int, offset: float) -> None:
        nonlocal context
        try:
            final = event_type == "final"
            current_status = runtime.status()
            await ws.send_text(json.dumps({
                "t": "model",
                **current_status,
                "state": "loading" if current_status["state"] == "not-loaded" else current_status["state"],
            }))
            decoded, language = await asyncio.to_thread(runtime.transcribe, audio, final, context, offset)
            await ws.send_text(json.dumps({"t": "model", **runtime.status()}))
            if not decoded:
                await ws.send_text(json.dumps(empty_decode_event(
                    event_type,
                    runtime.last_decode.get("audio_seconds", len(audio) / SR),
                    runtime.last_decode,
                    np.asarray(audio, dtype=np.float32),
                )))
                return
            text = " ".join(segment.text for segment in decoded)
            if final:
                async with final_lock:
                    session_segments.extend(decoded)
                    session_segments.sort(key=lambda segment: (segment.start, segment.end, segment.id))
                    context = " ".join(segment.text for segment in session_segments)[-200:]
                    for segment in decoded:
                        await ws.send_text(json.dumps({"t": "segment", **segment.to_dict(), "partial": False}, ensure_ascii=False))
                    await save_note()
            else:
                await ws.send_text(json.dumps({
                    "t": "partial",
                    "id": sid,
                    "start": offset,
                    "end": offset + len(audio) / SR,
                    "text": text,
                    "language": language,
                }, ensure_ascii=False))
        except Exception as exc:
            logger.exception("capture processing failed kind=%s", event_type)
            try:
                await ws.send_text(json.dumps({
                    "t": "error",
                    "scope": event_type,
                    "message": f"capture processing failed: {type(exc).__name__}: {str(exc)[:240]}",
                    "models": runtime.status(),
                }))
            except Exception:
                pass

    scheduler = DecodeScheduler(
        process,
        max_inflight=getattr(config, "max_inflight_decodes", 2),
        max_pending_finals=getattr(config, "max_pending_finals", 8),
    )
    await scheduler.start()
    try:
        segmenter = await asyncio.to_thread(Segmenter, config)
        last_vad_status = segmenter.vad_status
        await ws.send_text(json.dumps({"t": "vad", "state": "ready", "sample_rate": SR, **last_vad_status}))
        # Warm Whisper models as soon as capture starts so download/load progress
        # and the startup benchmark are visible before speech is detected.
        await ws.send_text(json.dumps({"t": "model", **runtime.status(), "state": "loading"}))
        try:
            await asyncio.to_thread(runtime.ensure_loaded)
            ready_status = runtime.status()
            await ws.send_text(json.dumps({"t": "model", **ready_status}))
            if ready_status.get("benchmark"):
                await ws.send_text(json.dumps({"t": "benchmark", **ready_status["benchmark"]}))
        except Exception as exc:
            await ws.send_text(json.dumps({
                "t": "error",
                "scope": "model",
                "message": f"model loading failed: {type(exc).__name__}: {str(exc)[:240]}",
                "models": runtime.status(),
            }))
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            events: list[tuple[str, Any, int]] = []
            flush_requested = False
            if message.get("bytes") is not None:
                raw_audio = message["bytes"]
                if not isinstance(raw_audio, (bytes, bytearray)) or not raw_audio or len(raw_audio) % 2:
                    logger.warning("discarded malformed PCM frame")
                    await ws.send_text(json.dumps({"t": "error", "scope": "audio", "message": "discarded malformed PCM frame"}))
                    continue
                if len(raw_audio) > 2_000_000:
                    logger.warning("discarded oversized PCM frame bytes=%d", len(raw_audio))
                    await ws.send_text(json.dumps({"t": "error", "scope": "audio", "message": "discarded oversized PCM frame"}))
                    continue
                pcm = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                if not np.isfinite(pcm).all():
                    logger.warning("discarded non-finite PCM frame")
                    await ws.send_text(json.dumps({"t": "error", "scope": "audio", "message": "discarded non-finite PCM frame"}))
                    continue
                pcm = resample_audio(pcm, source_sample_rate, SR)
                if not len(pcm):
                    continue
                elapsed += len(pcm) / SR
                events = await asyncio.to_thread(segmenter.feed, pcm)
                current_vad_status = segmenter.vad_status
                if not sent_first_frame_vad or current_vad_status != last_vad_status:
                    last_vad_status = current_vad_status
                    sent_first_frame_vad = True
                    await ws.send_text(json.dumps({"t": "vad", "state": "updated", **current_vad_status}))
                await ws.send_text(json.dumps({
                    "t": "level",
                    "v": min(1.0, segmenter.level / 0.12),
                    "on": segmenter.seg is not None,
                    "lag": scheduler.depth,
                    "sample_rate": source_sample_rate,
                }))
            elif message.get("text"):
                try:
                    payload = json.loads(message["text"])
                    event_name = payload.get("t")
                    if event_name == "config":
                        requested_profile = payload.get("model")
                        if requested_profile and requested_profile != runtime.status()["profile_id"]:
                            if elapsed > 0 or session_segments or scheduler.depth:
                                await ws.send_text(json.dumps({"t": "error", "scope": "model", "message": "Stop capture before changing the model."}))
                                continue
                            try:
                                selected = await asyncio.to_thread(runtime.select_model, str(requested_profile))
                                await ws.send_text(json.dumps({"t": "model", **selected, "state": "loading"}))
                                await asyncio.to_thread(runtime.ensure_loaded)
                                selected = runtime.status()
                                await ws.send_text(json.dumps({"t": "model", **selected}))
                                if selected.get("benchmark"):
                                    await ws.send_text(json.dumps({"t": "benchmark", **selected["benchmark"]}))
                            except Exception as exc:
                                await ws.send_text(json.dumps({"t": "error", "scope": "model", "message": f"model selection failed: {type(exc).__name__}: {str(exc)[:240]}", "models": runtime.status()}))
                                continue
                        requested_rate = float(payload.get("sample_rate", SR))
                        if not math.isfinite(requested_rate) or not 8000 <= requested_rate <= 192000:
                            await ws.send_text(json.dumps({"t": "error", "scope": "audio", "message": "unsupported browser sample rate"}))
                            continue
                        source_sample_rate = requested_rate
                        await ws.send_text(json.dumps({"t": "audio-config", "source_rate": source_sample_rate, "target_rate": SR}))
                    elif event_name == "flush":
                        events = await asyncio.to_thread(segmenter.flush)
                        flush_requested = True
                except (json.JSONDecodeError, TypeError, ValueError):
                    await ws.send_text(json.dumps({"t": "error", "scope": "stream", "message": "ignored malformed control message"}))
                    continue
            for event_type, audio, sid in events:
                await scheduler.submit(event_type, audio, sid, max(0.0, elapsed - len(audio) / SR))
            if flush_requested:
                await scheduler.drain()
                if session_segments:
                    # Always run one complete extraction pass on flush, even if
                    # the last debounced save happened moments earlier.
                    await save_note(force=True)
                else:
                    await ws.send_text(json.dumps(empty_decode_event(
                        "flush",
                        elapsed,
                        reason="no speech segment was captured",
                    )))
                await ws.send_text(json.dumps({
                    "t": "flushed",
                    "saved": bool(session_segments),
                    "segments": len(session_segments),
                    "note_id": session_note_id,
                }))
    except Exception as exc:
        # Disconnects and browser microphone shutdowns are normal session endings.
        try:
            await ws.send_text(json.dumps({
                "t": "error",
                "scope": "stream",
                "message": f"capture session failed: {type(exc).__name__}: {str(exc)[:240]}",
                "models": runtime.status(),
            }))
        except Exception:
            pass
    finally:
        await scheduler.shutdown()
