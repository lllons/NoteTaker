"""FastAPI surface for live capture, durable knowledge notes, search, and export."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

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
    return store.list(query, limit)


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
    results = store.list(question, 10)
    evidence = []
    for item in results:
        body = store.get(item["id"]) or {}
        for segment in body.get("transcript", []):
            if not question or any(term.lower() in segment.get("text", "").lower() for term in question.split() if len(term) > 2):
                evidence.append({"note_id": item["id"], "title": item["title"], "start": segment.get("start", 0), "end": segment.get("end", 0), "text": segment.get("text", ""), "confidence": segment.get("confidence", 0)})
    return {"question": question, "answer": "Retrieved evidence only; no unsupported answer was generated.", "evidence": evidence[:50]}


@app.websocket("/ws")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_text(json.dumps({"t": "ready", "model": f"{config.model} + {config.draft_model}"}))
    segmenter = await asyncio.to_thread(Segmenter)
    session_segments: list[TranscriptSegment] = []
    context = ""
    elapsed = 0.0
    tasks: set[asyncio.Task[Any]] = set()

    async def process(event_type: str, audio: Any, sid: int, offset: float) -> None:
        nonlocal context
        final = event_type == "final"
        decoded, language = await asyncio.to_thread(runtime.transcribe, audio, final, context, offset)
        if not decoded:
            return
        text = " ".join(segment.text for segment in decoded)
        if final:
            session_segments.extend(decoded)
            context = (context + " " + text)[-500:]
            for segment in decoded:
                await ws.send_text(json.dumps({"t": "segment", **segment.to_dict()}))
            note = await asyncio.to_thread(pipeline.create_note, session_segments, "Live capture", "live", "live-session")
            await ws.send_text(json.dumps({"t": "note", "title": note.title, "id": note.id, "language": language}))
        else:
            await ws.send_text(json.dumps({"t": "partial", "id": sid, "text": text, "language": language}))

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            events: list[tuple[str, Any, int]] = []
            if message.get("bytes") is not None:
                pcm = __import__("numpy").frombuffer(message["bytes"], __import__("numpy").int16).astype(__import__("numpy").float32) / 32768.0
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
