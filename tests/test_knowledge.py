import tempfile
import unittest
from pathlib import Path

from notetaker.extractor import HighFidelityExtractor
from notetaker.models import TranscriptSegment
from notetaker.pipeline import KnowledgePipeline
from notetaker.rendering import render
from notetaker.storage import KnowledgeStore


class KnowledgePipelineTests(unittest.TestCase):
    def setUp(self):
        self.segments = [
            TranscriptSegment("s1", 0, 8, "Vector databases store embeddings. An embedding is a numeric representation of meaning.", 0.94, "Speaker 1", 0.0, "en"),
            TranscriptSegment("s2", 8, 15, "The action item is to compare pgvector and a hosted service by Friday. See https://example.com/docs.", 0.58, "Speaker 1", 0.0, "en"),
        ]

    def test_extractor_preserves_facts_and_marks_uncertainty(self):
        result = HighFidelityExtractor().extract(self.segments)
        self.assertIn("embedding", result["concepts"])
        self.assertTrue(result["definitions"])
        self.assertTrue(result["action_items"])
        self.assertTrue(result["resources"])
        self.assertEqual(len(result["uncertain_regions"]), 1)

    def test_pipeline_round_trip_and_search(self):
        with tempfile.TemporaryDirectory() as directory:
            store = KnowledgeStore(Path(directory) / "notes.sqlite3")
            note = KnowledgePipeline(store).create_note(self.segments, "Vector database study")
            self.assertEqual(store.get(note.id)["title"], "Vector database study")
            self.assertEqual(store.list("pgvector")[0]["id"], note.id)
            self.assertIn("Executive summary", render(note, "md")[0])

    def test_exports_are_available(self):
        with tempfile.TemporaryDirectory() as directory:
            note = KnowledgePipeline(KnowledgeStore(Path(directory) / "notes.sqlite3")).create_note(self.segments, "Exports")
        for format_name in ("md", "json", "html", "anki", "pdf", "docx", "obsidian", "notion"):
            content, content_type, suffix = render(note, format_name)
            self.assertTrue(content)
            self.assertTrue(content_type)
            self.assertTrue(suffix)


if __name__ == "__main__":
    unittest.main()
