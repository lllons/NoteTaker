"""Knowledge capture orchestration and incremental note generation."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from .extractor import HighFidelityExtractor
from .models import KnowledgeNote, TranscriptSegment
from .provider import LLMProvider
from .storage import KnowledgeStore


class KnowledgePipeline:
    def __init__(self, store: KnowledgeStore, provider: LLMProvider | None = None) -> None:
        self.store = store
        self.extractor = HighFidelityExtractor(provider)

    def create_note(
        self,
        segments: Iterable[TranscriptSegment],
        title: str | None = None,
        source_type: str = "live",
        note_id: str | None = None,
    ) -> KnowledgeNote:
        transcript = list(segments)
        if not transcript:
            raise ValueError("At least one transcript segment is required")
        raw = " ".join(segment.text for segment in transcript)
        stable_id = note_id or hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        extracted = self.extractor.extract(transcript)
        semantic_segments = self.extractor.semantic_segments(transcript)
        note = KnowledgeNote(
            id=stable_id,
            title=title or self._title(raw),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source_type=source_type,
            language=next((s.language for s in transcript if s.language), None),
            duration=max(s.end for s in transcript),
            transcript=transcript,
            semantic_segments=semantic_segments,
            **extracted,
        )
        note.detailed_markdown = self.extractor.render_detailed(note)
        self.store.save(note)
        return note

    @staticmethod
    def _title(text: str) -> str:
        words = re.findall(r"\S+", text.strip())[:9]
        return " ".join(words).rstrip(".,:;") or "Untitled capture"
