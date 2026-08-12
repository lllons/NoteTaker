"""Export renderers for the durable knowledge note format."""

from __future__ import annotations

import html
import io
import json
import zipfile
from typing import Any

from .models import KnowledgeNote


def _stamp(seconds: float) -> str:
    minutes, remainder = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{remainder:02d}"


def render_markdown(note: KnowledgeNote) -> str:
    lines = [f"# {note.title}", "", f"> Source: `{note.source_type}` · Duration: `{_stamp(note.duration)}` · Language: `{note.language or 'auto'}`", ""]
    lines += ["## Table of contents", "", "- [Executive summary](#executive-summary)", "- [Semantic segments](#semantic-segments)", "- [Detailed notes](#detailed-notes)", "- [Reference transcript](#reference-transcript)", "- [Study notes](#study-notes)", "- [Action items](#action-items)", "- [Decisions](#decisions)", "- [Open questions](#open-questions)", "- [Knowledge graph](#knowledge-graph)", "- [Timeline](#timeline)", "- [Flashcards](#flashcards)", ""]
    lines += ["## Executive summary", ""] + [f"- {item}" for item in note.executive_summary] + [""]
    lines += ["## Semantic segments", ""]
    for segment in note.semantic_segments:
        title = segment.title or (segment.topics[0].title() if segment.topics else "Semantic segment")
        lines.append(f"- **{title}** · `{_stamp(segment.start)}–{_stamp(segment.end)}` · confidence `{segment.confidence:.0%}` · {segment.speaker}")
    lines += ["", "## Detailed notes", "", note.detailed_markdown or "_No structured notes extracted._", ""]
    lines += ["## Reference transcript", "", "<details><summary>Show near-verbatim transcript</summary>", ""]
    for segment in note.transcript:
        warning = " ⚠️ **low confidence**" if segment.confidence < 0.65 else ""
        lines.append(f"<a id=\"{segment.id}\"></a>\n> **{_stamp(segment.start)}–{_stamp(segment.end)} · {segment.speaker} · confidence {segment.confidence:.0%}**{warning}\n> {segment.text}")
    lines += ["", "</details>", ""]
    lines += ["## Study notes", ""] + [f"- {item}" for item in note.study_notes] + [""]
    lines += ["## Action items", ""]
    for item in note.action_items:
        lines.append(f"> [!todo]\n> {item.get('text', '')} _(source: {', '.join(item.get('source_segment_ids', []))})_\n")
    lines += ["## Decisions", ""]
    for item in note.decisions:
        lines.append(f"> [!decision]\n> {item.get('text', '')} _(source: {', '.join(item.get('source_segment_ids', []))})_\n")
    lines += ["## Open questions", ""]
    for item in note.open_questions:
        lines.append(f"> [!question]\n> {item.get('text', '')} _(source: {', '.join(item.get('source_segment_ids', []))})_\n")
    lines += ["## Knowledge graph", "", "```json", json.dumps([edge for edge in note.graph], default=lambda o: o.__dict__, indent=2), "```", ""]
    lines += ["## Timeline", ""]
    for item in note.timeline:
        start = item.start if hasattr(item, "start") else item.get("start", 0)
        end = item.end if hasattr(item, "end") else item.get("end", 0)
        label = item.label if hasattr(item, "label") else item.get("label", "Timeline event")
        detail = item.detail if hasattr(item, "detail") else item.get("detail", "")
        lines.append(f"- `{_stamp(start)}–{_stamp(end)}` — **{label}**: {detail}")
    lines += ["", "## Flashcards", ""]
    for index, card in enumerate(note.flashcards, 1):
        question = card.question if hasattr(card, "question") else card.get("question", "")
        answer = card.answer if hasattr(card, "answer") else card.get("answer", "")
        lines += [f"<details><summary>{index}. {question}</summary>", "", answer, "", "</details>", ""]
    if note.uncertain_regions:
        lines += ["## Accuracy safeguards", "", "> ⚠️ The following regions need review:", ""]
        lines += [f"- `{_stamp(x['start'])}` — {x['reason']}: {x['text']}" for x in note.uncertain_regions]
    if note.inferred_items:
        lines += ["", "> ℹ️ Inferred or provider-generated interpretations are explicitly labeled and are not source facts.", ""]
        lines += [f"- {x.get('text', '')}" for x in note.inferred_items]
    return "\n".join(lines).strip() + "\n"


def render_json(note: KnowledgeNote) -> str:
    return json.dumps(note.to_dict(), ensure_ascii=False, indent=2, default=lambda value: value.__dict__)


def render_anki(note: KnowledgeNote) -> str:
    rows = ["#separator:Tab", "#html:true", "#tags column:3"]
    for card in note.flashcards:
        question = card.question if hasattr(card, "question") else card.get("question", "")
        answer = card.answer if hasattr(card, "answer") else card.get("answer", "")
        tags_value = card.tags if hasattr(card, "tags") else card.get("tags", [])
        tags = " ".join(["notetaker", *tags_value])
        rows.append(f"{question.replace(chr(9), ' ')}\t{answer.replace(chr(9), ' ')}\t{tags}")
    return "\n".join(rows) + "\n"


def render_html(note: KnowledgeNote) -> str:
    body = render_markdown(note)
    escaped = html.escape(body)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(note.title)}</title><style>body{{font:16px system-ui;max-width:900px;margin:40px auto;padding:0 20px;line-height:1.6;white-space:pre-wrap}}code{{font-family:ui-monospace}}</style></head><body>{escaped}</body></html>"


def render_docx(note: KnowledgeNote) -> bytes:
    """Create a dependency-free DOCX with the full Markdown body as paragraphs."""
    paragraphs = []
    for line in render_markdown(note).splitlines():
        safe = html.escape(line)
        paragraphs.append(f"<w:p><w:r><w:t xml:space='preserve'>{safe}</w:t></w:r></w:p>")
    document = "<?xml version='1.0' encoding='UTF-8' standalone='yes'?><w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'><w:body>" + "".join(paragraphs) + "<w:sectPr/></w:body></w:document>"
    content_types = "<?xml version='1.0' encoding='UTF-8'?><Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'><Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/><Default Extension='xml' ContentType='application/xml'/><Override PartName='/word/document.xml' ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'/></Types>"
    rels = "<?xml version='1.0' encoding='UTF-8'?><Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='word/document.xml'/></Relationships>"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)
    return output.getvalue()


def render_pdf(note: KnowledgeNote) -> bytes:
    """Create a small, portable text PDF without requiring a system PDF package."""
    text_lines = render_markdown(note).splitlines()[:500]
    stream_lines = ["BT", "/F1 9 Tf", "50 760 Td"]
    for line in text_lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")[:130]
        stream_lines.append(f"({escaped}) Tj 0 -12 Td")
    stream_lines.append("ET")
    stream = "\n".join(stream_lines).encode("latin-1", "replace")
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>", b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>", b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(output)); output.extend(f"{index} 0 obj\n".encode()); output.extend(obj); output.extend(b"\nendobj\n")
    xref = len(output); output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    output.extend("".join(f"{offset:010d} 00000 n \n" for offset in offsets[1:]).encode())
    output.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return bytes(output)


def render(note: KnowledgeNote, format_name: str) -> tuple[Any, str, str]:
    formats: dict[str, tuple[Any, str, str]] = {
        "md": (render_markdown, "text/markdown", ".md"),
        "markdown": (render_markdown, "text/markdown", ".md"),
        "json": (render_json, "application/json", ".json"),
        "anki": (render_anki, "text/tab-separated-values", ".txt"),
        "html": (render_html, "text/html", ".html"),
        "pdf": (render_pdf, "application/pdf", ".pdf"),
        "docx": (render_docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
        "obsidian": (render_markdown, "text/markdown", ".md"),
        "notion": (render_markdown, "text/markdown", ".md"),
    }
    if format_name not in formats:
        raise ValueError(f"Unsupported export format: {format_name}")
    function, content_type, suffix = formats[format_name]
    return function(note), content_type, suffix
