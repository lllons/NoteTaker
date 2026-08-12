"""SQLite persistence for notes, segments, facts, graph edges, and flashcards."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

from .models import KnowledgeNote


SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source_type TEXT NOT NULL,
    language TEXT,
    duration REAL NOT NULL,
    body_json TEXT NOT NULL,
    search_text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_created_at ON notes(created_at);
CREATE INDEX IF NOT EXISTS idx_notes_source_type ON notes(source_type);
"""


class KnowledgeStore:
    def __init__(self, path: str | Path = "data/knowledge.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def save(self, note: KnowledgeNote) -> None:
        body = note.to_dict()
        # Index the complete structured payload so actions, people, dates,
        # resources, questions, and timestamps are searchable too.
        search_text = self._normalize(json.dumps(body, ensure_ascii=False))
        with self._connect() as db:
            db.execute(
                """INSERT INTO notes(id,title,created_at,source_type,language,duration,body_json,search_text)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET title=excluded.title, body_json=excluded.body_json,
                   search_text=excluded.search_text, duration=excluded.duration""",
                (note.id, note.title, note.created_at, note.source_type, note.language, note.duration,
                 json.dumps(body, ensure_ascii=False), search_text),
            )
            db.commit()

    def get(self, note_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT body_json FROM notes WHERE id = ?", (note_id,)).fetchone()
        return json.loads(row["body_json"]) if row else None

    @staticmethod
    def _normalize(value: str) -> str:
        value = unicodedata.normalize("NFKC", value).casefold()
        return re.sub(r"[^\w:#./%-]+", " ", value, flags=re.UNICODE)

    @classmethod
    def _terms(cls, query: str) -> list[str]:
        aliases = {
            "js": "javascript", "ts": "typescript", "py": "python", "postgresql": "postgres",
            "pgvector": "vector database", "llm": "large language model", "rag": "retrieval augmented generation",
            "k8s": "kubernetes", "tf": "tensorflow", "sk": "scikit learn",
        }
        terms: list[str] = []
        for raw in cls._normalize(query).split():
            expanded = aliases.get(raw, raw)
            terms.extend(expanded.split())
            if len(raw) > 5:
                for suffix in ("ements", "ment", "ing", "ed", "es", "s"):
                    if raw.endswith(suffix) and len(raw) - len(suffix) >= 4:
                        terms.append(raw[:-len(suffix)])
                        break
        return list(dict.fromkeys(term for term in terms if len(term) > 1))

    @staticmethod
    def _timestamp_seconds(value: str) -> int | None:
        match = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{2})", value)
        if not match:
            return None
        hours = int(match.group(1) or 0)
        return hours * 3600 + int(match.group(2)) * 60 + int(match.group(3))

    def search(self, query: str = "", limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        raw_query = query.strip()
        filters: dict[str, str] = {}
        free_terms: list[str] = []
        for token in raw_query.split():
            if ":" in token and token.split(":", 1)[0].lower() in {"concept", "person", "action", "question", "speaker", "after", "before"}:
                key, value = token.split(":", 1)
                filters[key.lower()] = value
            else:
                free_terms.append(token)
        terms = self._terms(" ".join(free_terms))
        phrase = self._normalize(" ".join(free_terms))
        timestamp = next((self._timestamp_seconds(token) for token in free_terms), None)
        with self._connect() as db:
            rows = db.execute("SELECT id,title,created_at,source_type,language,duration,body_json,search_text FROM notes").fetchall()
        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            body = json.loads(row["body_json"])
            normalized = row["search_text"]
            score = 0.0
            if phrase and phrase in normalized:
                score += 12.0
            for term in terms:
                occurrences = normalized.count(term)
                if occurrences:
                    score += min(occurrences, 12) * 2.0
                    if term in self._normalize(row["title"]):
                        score += 8.0
            if timestamp is not None and any(
                float(segment.get("start", 0)) <= timestamp <= float(segment.get("end", 0))
                for segment in body.get("transcript", [])
            ):
                score += 10.0
            if filters.get("concept") and filters["concept"].casefold() not in normalized:
                continue
            if filters.get("person") and filters["person"].casefold() not in normalized:
                continue
            if filters.get("action") and filters["action"].casefold() not in self._normalize(json.dumps(body.get("action_items", []))):
                continue
            if filters.get("question") and filters["question"].casefold() not in self._normalize(json.dumps(body.get("open_questions", []))):
                continue
            if filters.get("speaker") and filters["speaker"].casefold() not in normalized:
                continue
            if filters.get("after") and row["created_at"][:10] < filters["after"]:
                continue
            if filters.get("before") and row["created_at"][:10] > filters["before"]:
                continue
            if raw_query and score == 0 and not filters:
                continue
            if filters and score == 0:
                score = 1.0
            ranked.append((score, {key: row[key] for key in ("id", "title", "created_at", "source_type", "language", "duration")}))
        ranked.sort(key=lambda item: (item[0], item[1]["created_at"]), reverse=True)
        return [item[1] for item in ranked[:limit]]

    def list(self, query: str = "", limit: int = 50) -> list[dict[str, Any]]:
        return self.search(query, limit)

    def delete(self, note_id: str) -> bool:
        with self._connect() as db:
            result = db.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            db.commit()
            return result.rowcount > 0
