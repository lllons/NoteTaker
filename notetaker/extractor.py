"""High-fidelity extraction that favors recall and labels uncertainty.

This module intentionally uses conservative heuristics in local mode. It extracts
rather than invents; an optional provider can add structured interpretations, but
those are kept separate from source-backed facts.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from .models import GraphEdge, TranscriptSegment
from .provider import LLMProvider


STOPWORDS = {
    "about", "after", "again", "also", "because", "been", "being", "could", "does", "from", "have",
    "into", "just", "more", "most", "other", "over", "same", "some", "such", "than", "that", "their",
    "there", "these", "they", "this", "those", "through", "very", "what", "when", "where", "which", "while",
    "will", "with", "would", "your", "you", "the", "and", "for", "are", "was", "were", "but", "not", "our",
    "can", "has", "had", "how", "its", "it's", "then", "them", "we", "who", "why", "like", "said", "one",
}

URL_RE = re.compile(r"https?://[^\s)]+|www\.[^\s)]+")
DATE_RE = re.compile(r"\b(?:\d{1,2}[/-])?\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s+\d{4})?\b", re.I)
NUMBER_RE = re.compile(r"\b\d+(?:[,.]\d+)?%?\b")


def _clean(text: str) -> str:
    # Remove only low-value verbal noise; do not normalize technical content.
    text = re.sub(r"\b(um+|uh+|er+|erm|hmm+)(?=\s|[,.;!?]|$)", "", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _source_ids(segment: TranscriptSegment) -> list[str]:
    return [segment.id]


def _topic_words(text: str, limit: int = 6) -> list[str]:
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_+#.-]{2,}", text)]
    counts = Counter(w for w in words if w not in STOPWORDS)
    return [word for word, _ in counts.most_common(limit)]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


class HighFidelityExtractor:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider

    def semantic_segments(self, segments: Iterable[TranscriptSegment]) -> list[TranscriptSegment]:
        """Merge adjacent utterances into topic-aware semantic blocks.

        Raw VAD segments remain untouched for citation. A long pause or a sharp
        topic transition starts a new block; this gives readers useful headings
        without losing the original timing or wording.
        """
        source = list(segments)
        if not source:
            return []
        blocks: list[TranscriptSegment] = []
        current = source[0]
        for incoming in source[1:]:
            left = set(current.topics)
            right = set(incoming.topics)
            overlap = len(left & right) / max(1, len(left | right))
            pause = incoming.start - current.end
            split = pause > 1.5 or (left and right and overlap == 0 and len(current.text) > 80)
            if split:
                blocks.append(current)
                current = incoming
                continue
            current = TranscriptSegment(
                id=f"topic-{current.id}-{incoming.id}",
                start=current.start,
                end=incoming.end,
                text=f"{current.text} {incoming.text}".strip(),
                confidence=min(current.confidence, incoming.confidence),
                speaker=current.speaker if current.speaker == incoming.speaker else "Multiple speakers",
                speaker_confidence=min(current.speaker_confidence, incoming.speaker_confidence),
                language=current.language or incoming.language,
                words=[*current.words, *incoming.words],
                topics=list(dict.fromkeys([*current.topics, *incoming.topics])),
                title=(current.topics or ["Semantic segment"])[0].title(),
            )
        blocks.append(current)
        for block in blocks:
            if not block.title:
                block.title = (block.topics or ["Semantic segment"])[0].title()
        return blocks

    def extract(self, segments: Iterable[TranscriptSegment]) -> dict[str, Any]:
        transcript = list(segments)
        for segment in transcript:
            segment.text = _clean(segment.text)
            segment.topics = _topic_words(segment.text)

        all_text = " ".join(segment.text for segment in transcript)
        definitions = self._definitions(transcript)
        explanations = self._pattern_items(transcript, r"(?:because|therefore|so that|which means|in other words)", "explanation")
        examples = self._pattern_text(transcript, r"(?:for example|for instance|such as|e\.g\.)")
        analogies = self._pattern_text(transcript, r"(?:like|similar to|as if|analogy|think of)")
        formulas = self._formulas(transcript)
        statistics = self._statistics(transcript)
        code_snippets = self._code(transcript)
        action_items = self._pattern_items(transcript, r"(?:TODO|action item|need to|should|will|must|deadline|follow up|next step)", "action")
        decisions = self._pattern_items(transcript, r"(?:decided|decision|agreed|we(?:'| wi)ll use|the plan is)", "decision")
        open_questions = self._questions(transcript)
        resources = self._resources(transcript)
        concepts = self._concepts(transcript, definitions)
        entities = self._entities(all_text)
        uncertain_regions = [
            {"start": segment.start, "end": segment.end, "text": segment.text, "reason": "low transcript confidence", "source_segment_ids": [segment.id]}
            for segment in transcript if segment.confidence < 0.65
        ]
        graph = self._graph(transcript, concepts)
        timeline = [
            {"start": s.start, "end": s.end, "label": s.topics[0] if s.topics else "Transcript segment", "detail": s.text, "segment_id": s.id}
            for s in transcript
        ]
        flashcards = self._flashcards(definitions, concepts, formulas, transcript)
        summary = self._summary(transcript, concepts, decisions, action_items)
        study = self._study_notes(concepts, definitions, explanations, formulas, uncertain_regions)

        local_facts: dict[str, Any] = {
            "concepts": concepts, "definitions": definitions, "explanations": explanations,
            "examples": examples, "analogies": analogies, "formulas": formulas,
            "statistics": statistics, "code_snippets": code_snippets, "entities": entities,
            "action_items": action_items, "decisions": decisions, "open_questions": open_questions,
            "resources": resources, "uncertain_regions": uncertain_regions,
        }
        inferred_items: list[dict[str, Any]] = []
        provider_name = "local-deterministic"
        if self.provider:
            provider_name = self.provider.name
            try:
                generated = self.provider.enrich(all_text, local_facts)
                inferred_items = self._provider_facts(generated, local_facts)
            except Exception as exc:
                inferred_items = [{"text": f"Provider enrichment unavailable: {type(exc).__name__}", "label": "provider-error"}]

        return {
            **local_facts,
            "executive_summary": summary,
            "reference_notes": [s.text for s in transcript],
            "study_notes": study,
            "graph": graph,
            "timeline": timeline,
            "flashcards": flashcards,
            "inferred_items": inferred_items,
            "provider": provider_name,
        }

    def render_detailed(self, note: Any) -> str:
        lines: list[str] = []
        sections = [
            ("Key concepts", note.concepts),
            ("Definitions", [f"**{x['term']}** — {x['definition']}" for x in note.definitions]),
            ("Explanations", [x["text"] for x in note.explanations]),
            ("Examples", note.examples),
            ("Analogies", note.analogies),
            ("Formulas and numbers", [*note.formulas, *note.statistics]),
            ("Code and commands", [f"```\n{x}\n```" for x in note.code_snippets]),
            ("People, organizations, and products", [f"**{key}:** {', '.join(values)}" for key, values in note.entities.items() if values]),
            ("Resources", [f"{x.get('kind', 'resource')}: {x.get('text', '')}" for x in note.resources]),
            ("Open questions", [x["text"] for x in note.open_questions]),
        ]
        for title, items in sections:
            if not items:
                continue
            lines += [f"### {title}", ""]
            lines += [f"- {item}" for item in items]
            lines.append("")
        return "\n".join(lines).strip()

    def _definitions(self, segments: list[TranscriptSegment]) -> list[dict[str, Any]]:
        result = []
        pattern = re.compile(r"\b([A-Za-z][\w -]{1,50}?)\s+(?:is|means|refers to|is defined as)\s+([^.!?]{8,240})", re.I)
        for segment in segments:
            for match in pattern.finditer(segment.text):
                result.append({"term": match.group(1).strip(), "definition": match.group(2).strip(), "source_segment_ids": _source_ids(segment)})
        return result

    def _pattern_items(self, segments: list[TranscriptSegment], marker: str, kind: str) -> list[dict[str, Any]]:
        result = []
        pattern = re.compile(marker, re.I)
        for segment in segments:
            for sentence in _sentences(segment.text):
                if pattern.search(sentence):
                    result.append({"text": sentence, "kind": kind, "source_segment_ids": _source_ids(segment), "confidence": segment.confidence})
        return result

    def _pattern_text(self, segments: list[TranscriptSegment], marker: str) -> list[str]:
        return [item["text"] for item in self._pattern_items(segments, marker, "source-backed")]

    def _formulas(self, segments: list[TranscriptSegment]) -> list[str]:
        result = []
        for segment in segments:
            for sentence in _sentences(segment.text):
                if re.search(r"(?:\b[A-Za-z]\w*\s*[=<>]|\b(?:plus|minus|times|divided by)\b|∑|∫|→|=>)", sentence):
                    result.append(sentence)
        return result

    def _statistics(self, segments: list[TranscriptSegment]) -> list[str]:
        result = []
        for segment in segments:
            for sentence in _sentences(segment.text):
                if NUMBER_RE.search(sentence) or re.search(r"\b(?:percent|million|billion|average|median|probability|rate)\b", sentence, re.I):
                    result.append(sentence)
        return result

    def _code(self, segments: list[TranscriptSegment]) -> list[str]:
        result = []
        for segment in segments:
            fenced = re.findall(r"```(?:\w+)?\s*(.*?)```", segment.text, re.S)
            result.extend(fenced)
            for sentence in _sentences(segment.text):
                if re.search(r"(?:^|\s)(?:pip|npm|bun|git|docker|python|curl)\s+[-\w]|[\w./-]+\.(?:py|js|ts|json|toml|yaml)\b", sentence, re.I):
                    result.append(sentence)
        return list(dict.fromkeys(x.strip() for x in result if x.strip()))

    def _questions(self, segments: list[TranscriptSegment]) -> list[dict[str, Any]]:
        result = []
        for segment in segments:
            for sentence in _sentences(segment.text):
                if sentence.rstrip().endswith("?"):
                    result.append({"text": sentence, "source_segment_ids": _source_ids(segment), "status": "open"})
        return result

    def _resources(self, segments: list[TranscriptSegment]) -> list[dict[str, Any]]:
        result = []
        for segment in segments:
            for url in URL_RE.findall(segment.text):
                result.append({"kind": "website", "text": url, "source_segment_ids": _source_ids(segment)})
            for sentence in _sentences(segment.text):
                if re.search(r"\b(?:book|paper|article|documentation|docs|repository|repo|tool|software|course)\b", sentence, re.I):
                    result.append({"kind": "mentioned resource", "text": sentence, "source_segment_ids": _source_ids(segment)})
        return result

    def _concepts(self, segments: list[TranscriptSegment], definitions: list[dict[str, Any]]) -> list[str]:
        terms = [item["term"].strip().lower() for item in definitions]
        for segment in segments:
            terms.extend(segment.topics)
        return list(dict.fromkeys(term for term in terms if len(term) > 2))[:40]

    def _entities(self, text: str) -> dict[str, list[str]]:
        proper = re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}(?:\s+[A-Z][A-Za-z0-9-]{2,})*\b", text)
        organizations = [x for x in proper if re.search(r"\b(?:Inc|Corp|Company|University|Institute|Labs?|Org)\b", x)]
        people = [x for x in proper if x not in organizations and len(x.split()) <= 3]
        dates = DATE_RE.findall(text)
        return {"people_or_proper_nouns": list(dict.fromkeys(people))[:40], "organizations": list(dict.fromkeys(organizations))[:40], "dates": list(dict.fromkeys(dates))[:40]}

    def _graph(self, segments: list[TranscriptSegment], concepts: list[str]) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        relation_patterns = [(r"because|causes|leads to|results in", "causes"), (r"depends on|requires|built on", "depends on"), (r"unlike|versus|rather than|contrasts", "contrasts with")]
        for segment in segments:
            present = [concept for concept in concepts if re.search(rf"\b{re.escape(concept)}\b", segment.text, re.I)]
            if len(present) >= 2:
                relation = "related to"
                for pattern, candidate in relation_patterns:
                    if re.search(pattern, segment.text, re.I):
                        relation = candidate
                        break
                edges.append(GraphEdge(present[0], relation, present[1], [segment.id], segment.confidence))
        return edges

    def _summary(self, segments: list[TranscriptSegment], concepts: list[str], decisions: list[dict[str, Any]], actions: list[dict[str, Any]]) -> list[str]:
        bullets = []
        if concepts:
            bullets.append("Core concepts: " + ", ".join(concepts[:8]) + ".")
        for segment in segments[:4]:
            if segment.text:
                bullets.append(f"[{segment.start:.0f}s] {segment.text}")
        if decisions:
            bullets.append("Decisions captured: " + " ".join(item["text"] for item in decisions[:2]))
        if actions:
            bullets.append("Follow-ups captured: " + " ".join(item["text"] for item in actions[:2]))
        return bullets[:10]

    def _study_notes(self, concepts: list[str], definitions: list[dict[str, Any]], explanations: list[dict[str, Any]], formulas: list[str], uncertain: list[dict[str, Any]]) -> list[str]:
        notes = [f"Study {concept}: locate its definition and the cited transcript segment." for concept in concepts[:12]]
        notes.extend(f"Remember: {item['term']} means {item['definition']}." for item in definitions[:12])
        notes.extend(f"Reasoning: {item['text']}" for item in explanations[:8])
        notes.extend(f"Formula or numeric claim to verify: {item}" for item in formulas[:8])
        if uncertain:
            notes.append("Review the low-confidence transcript regions before relying on exact wording or numbers.")
        return notes

    def _flashcards(self, definitions: list[dict[str, Any]], concepts: list[str], formulas: list[str], segments: list[TranscriptSegment]) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        for item in definitions[:20]:
            cards.append({"question": f"What is {item['term']}?", "answer": item["definition"], "card_type": "definition", "source_segment_ids": item["source_segment_ids"], "tags": ["definition", item["term"]]})
        for concept in concepts[:12]:
            cards.append({"question": f"Where does the transcript use or explain {concept}?", "answer": next((s.text for s in segments if concept in s.topics), "See the cited transcript segments."), "card_type": "concept", "source_segment_ids": [s.id for s in segments if concept in s.topics][:3], "tags": ["concept", concept]})
        for formula in formulas[:8]:
            cards.append({"question": "What formula or numeric relationship should be remembered?", "answer": formula, "card_type": "formula", "source_segment_ids": [], "tags": ["formula"]})
        return cards

    def _provider_facts(self, generated: dict[str, Any], local_facts: dict[str, Any]) -> list[dict[str, Any]]:
        allowed = {"concepts", "definitions", "explanations", "examples", "analogies", "formulas", "statistics", "code_snippets", "action_items", "decisions", "open_questions", "resources"}
        results = []
        for key in allowed:
            for item in generated.get(key, []) if isinstance(generated.get(key, []), list) else []:
                results.append({"label": f"provider-{key}", "text": item if isinstance(item, str) else str(item), "source": "provider output; verify against cited transcript"})
        return results
