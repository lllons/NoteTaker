"""SQLite persistence for notes, segments, facts, graph edges, and flashcards."""

from __future__ import annotations

import json
import sqlite3
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
        search_text = json.dumps(body, ensure_ascii=False)
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

    def list(self, query: str = "", limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        with self._connect() as db:
            if query.strip():
                like = f"%{query.strip()}%"
                rows = db.execute(
                    "SELECT id,title,created_at,source_type,language,duration FROM notes "
                    "WHERE search_text LIKE ? COLLATE NOCASE ORDER BY created_at DESC LIMIT ?",
                    (like, limit),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT id,title,created_at,source_type,language,duration FROM notes "
                    "ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, note_id: str) -> bool:
        with self._connect() as db:
            result = db.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            db.commit()
            return result.rowcount > 0
