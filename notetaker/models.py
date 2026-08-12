"""Typed domain objects shared by transcription, extraction, storage, and rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TranscriptWord:
    text: str
    start: float
    end: float
    confidence: float


@dataclass
class TranscriptSegment:
    id: str
    start: float
    end: float
    text: str
    confidence: float = 0.0
    speaker: str = "Unknown speaker"
    speaker_confidence: float = 0.0
    language: str | None = None
    words: list[TranscriptWord] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    source: str
    relation: str
    target: str
    evidence_segment_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class TimelineEvent:
    start: float
    end: float
    label: str
    detail: str
    segment_id: str


@dataclass
class Flashcard:
    question: str
    answer: str
    card_type: str
    source_segment_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class KnowledgeNote:
    id: str
    title: str
    created_at: str
    source_type: str
    language: str | None
    duration: float
    transcript: list[TranscriptSegment]
    semantic_segments: list[TranscriptSegment] = field(default_factory=list)
    executive_summary: list[str] = field(default_factory=list)
    detailed_markdown: str = ""
    reference_notes: list[str] = field(default_factory=list)
    study_notes: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    definitions: list[dict[str, Any]] = field(default_factory=list)
    explanations: list[dict[str, Any]] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    analogies: list[str] = field(default_factory=list)
    formulas: list[str] = field(default_factory=list)
    statistics: list[str] = field(default_factory=list)
    code_snippets: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    action_items: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[dict[str, Any]] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)
    graph: list[GraphEdge] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    flashcards: list[Flashcard] = field(default_factory=list)
    uncertain_regions: list[dict[str, Any]] = field(default_factory=list)
    inferred_items: list[dict[str, Any]] = field(default_factory=list)
    provider: str = "local-deterministic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
