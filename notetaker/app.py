"""FastAPI surface for live capture, durable knowledge notes, search, and export."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import AppConfig
from .models import TranscriptSegment
from .pipeline import KnowledgePipeline
from .provider import provider_from_config
from .rendering import render
from .storage import KnowledgeStore
from .transcription import Segmenter, TranscriptionRuntime
from .web import PAGE


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
    return {"ok": True, "service": "notetaker", "provider": provider.name if provider else "local-deterministic", "model": config.model}


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
    await ws.send_text(json.dumps({"t": "ready", "model": f"{config.model} + {config.draft_model}"}))
    segmenter = await asyncio.to_thread(Segmenter, config)
    session_segments: list[TranscriptSegment] = []
    context = ""
    elapsed = 0.0
    tasks: set[asyncio.Task[Any]] = set()
    final_lock = asyncio.Lock()

    async def process(event_type: str, audio: Any, sid: int, offset: float) -> None:
        nonlocal context
        try:
            final = event_type == "final"
            decoded, language = await asyncio.to_thread(runtime.transcribe, audio, final, context, offset)
            if not decoded:
                return
            text = " ".join(segment.text for segment in decoded)
            if final:
                async with final_lock:
                    session_segments.extend(decoded)
                    session_segments.sort(key=lambda segment: (segment.start, segment.end, segment.id))
                    context = " ".join(segment.text for segment in session_segments)[-config.context_chars:]
                    for segment in decoded:
                        await ws.send_text(json.dumps({"t": "segment", **segment.to_dict(), "partial": False}, ensure_ascii=False))
                    note = await asyncio.to_thread(pipeline.create_note, session_segments, "Live capture", "live", "live-session", False)
                    await ws.send_text(json.dumps({"t": "note", "title": note.title, "id": note.id, "language": language}))
            else:
                await ws.send_text(json.dumps({"t": "partial", "id": sid, "text": text, "language": language}, ensure_ascii=False))
        except Exception as exc:
            try:
                await ws.send_text(json.dumps({"t": "error", "scope": event_type, "message": f"capture processing failed: {type(exc).__name__}"}))
            except Exception:
                pass

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            events: list[tuple[str, Any, int]] = []
            if message.get("bytes") is not None:
                raw_audio = message["bytes"]
                if not isinstance(raw_audio, (bytes, bytearray)) or not raw_audio or len(raw_audio) % 2:
                    await ws.send_text(json.dumps({"t": "error", "scope": "audio", "message": "discarded malformed PCM frame"}))
                    continue
                if len(raw_audio) > 2_000_000:
                    await ws.send_text(json.dumps({"t": "error", "scope": "audio", "message": "discarded oversized PCM frame"}))
                    continue
                pcm = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                if not np.isfinite(pcm).all():
                    await ws.send_text(json.dumps({"t": "error", "scope": "audio", "message": "discarded non-finite PCM frame"}))
                    continue
                elapsed += len(pcm) / 16000
                events = await asyncio.to_thread(segmenter.feed, pcm)
                await ws.send_text(json.dumps({"t": "level", "v": min(1.0, segmenter.level / 0.12), "on": segmenter.seg is not None, "lag": len(tasks)}))
            elif message.get("text"):
                try:
                    if json.loads(message["text"]).get("t") == "flush":
                        events = await asyncio.to_thread(segmenter.flush)
                except (json.JSONDecodeError, TypeError):
                    continue
            for event_type, audio, sid in events:
                task = asyncio.create_task(process(event_type, audio, sid, max(0.0, elapsed - len(audio) / 16000)))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
    except Exception:
        # Disconnects and browser microphone shutdowns are normal session endings.
        pass
    finally:
        for task in tasks:
            task.cancel()
