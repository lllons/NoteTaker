"""Streaming transcription primitives.

The runtime keeps the accurate and draft Whisper passes separate. Word timestamps
and confidence are retained so downstream extraction can cite exact regions.
True acoustic diarization is an adapter boundary; labels-only mode never claims
that one voice has been distinguished from another.
"""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import get_vad_model

from .models import TranscriptSegment, TranscriptWord

SR = 16000
FRAME = 512
PREROLL = 16
START_FRAMES = 2
END_FRAMES = 30
MAX_SEG = 15 * SR
VAD_ON = 0.5
VAD_OFF = 0.35
MIN_SEG = int(0.25 * SR)
PARTIAL_EVERY = SR
HALLUCINATIONS = {"you", "thank you", "thanks for watching", "bye", "okay", "oh", "hmm", "um"}

MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base.en": "Systran/faster-whisper-base.en",
    "small.en": "Systran/faster-whisper-small.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "medium.en": "Systran/faster-whisper-medium.en",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
}


@dataclass
class ModelBundle:
    model: Any
    lock: threading.Lock
    executor: ThreadPoolExecutor


class SpeakerLabeler:
    """Speaker metadata adapter.

    labels-only is intentionally conservative: it preserves a stable speaker
    field for schemas and future diarization without pretending to identify
    voices. A pyannote-compatible adapter can implement the same `label` method.
    """

    def __init__(self, mode: str = "labels-only") -> None:
        self.mode = mode

    def label(self, segment_id: str) -> tuple[str, float]:
        if self.mode == "labels-only":
            return "Speaker 1", 0.0
        return "Unknown speaker", 0.0


class TranscriptionRuntime:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.final: ModelBundle | None = None
        self.draft: ModelBundle | None = None
        self._load_lock = threading.Lock()
        self.speakers = SpeakerLabeler(config.diarization)

    def _load(self, name: str) -> ModelBundle:
        repo = MODELS.get(name, name)
        model = WhisperModel(repo, device="cpu", compute_type="int8", cpu_threads=self.config.threads)
        return ModelBundle(model, threading.Lock(), ThreadPoolExecutor(max_workers=1))

    def ensure_loaded(self) -> None:
        with self._load_lock:
            if self.final is None:
                self.final = self._load(self.config.model)
            if self.draft is None:
                self.draft = self.final if self.config.draft_model == self.config.model else self._load(self.config.draft_model)

    def transcribe(self, audio: np.ndarray, final: bool, context: str = "", offset: float = 0.0) -> tuple[list[TranscriptSegment], str | None]:
        self.ensure_loaded()
        bundle = self.final if final else self.draft
        assert bundle is not None
        if audio is None or not len(audio):
            return [], self.config.language
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.abs(audio).max())
        if 0 < peak < 0.7:
            audio = audio * min(0.85 / peak, 20.0)
        prompt_parts = [
            "Transcribe verbatim. Preserve technical terms, names, acronyms, numbers, units, punctuation, equations, code, commands, URLs, file paths, and capitalization.",
            self.config.hotwords or "",
            context[-self.config.context_chars:],
        ]
        initial_prompt = " ".join(part for part in prompt_parts if part).strip()
        with bundle.lock:
            raw_segments, info = bundle.model.transcribe(
                audio,
                language=self.config.language,
                beam_size=self.config.beam_size if final else 1,
                temperature=0.0,
                initial_prompt=initial_prompt or None,
                hotwords=self.config.hotwords if final else None,
                condition_on_previous_text=False,
                without_timestamps=False,
                word_timestamps=True,
                vad_filter=final,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
            )
            raw_segments = list(raw_segments)
        language = getattr(info, "language", None) or self.config.language
        result: list[TranscriptSegment] = []
        for index, raw in enumerate(raw_segments):
            text = str(getattr(raw, "text", "") or "").strip()
            if not text:
                continue
            raw_start = max(0.0, float(getattr(raw, "start", 0.0) or 0.0))
            raw_end = max(raw_start, float(getattr(raw, "end", raw_start) or raw_start))
            no_speech = max(0.0, min(1.0, float(getattr(raw, "no_speech_prob", 0.0) or 0.0)))
            avg_logprob = float(getattr(raw, "avg_logprob", -0.5) or -0.5)
            compression = float(getattr(raw, "compression_ratio", 0.0) or 0.0)
            log_confidence = min(1.0, max(0.05, float(np.exp(min(0.0, avg_logprob)))))
            compression_penalty = 0.35 if compression > 2.4 else 1.0
            confidence = max(0.0, min(1.0, (1.0 - no_speech) * log_confidence * compression_penalty))
            if len(audio) / SR < 1.6 and text.lower().strip(" .,!?") in HALLUCINATIONS:
                continue
            segment_id = f"seg-{int((offset + raw_start) * 1000):09d}-{index:03d}"
            speaker, speaker_confidence = self.speakers.label(segment_id)
            words: list[TranscriptWord] = []
            for word in getattr(raw, "words", None) or []:
                word_start = offset + float(getattr(word, "start", raw_start) or raw_start)
                word_end = offset + float(getattr(word, "end", raw_end) or raw_end)
                words.append(TranscriptWord(
                    text=str(getattr(word, "word", "") or "").strip(),
                    start=word_start,
                    end=max(word_start, word_end),
                    confidence=max(0.0, min(1.0, float(getattr(word, "probability", confidence) or confidence))),
                ))
            result.append(TranscriptSegment(
                id=segment_id,
                start=offset + raw_start,
                end=offset + raw_end,
                text=text,
                confidence=confidence,
                speaker=speaker,
                speaker_confidence=speaker_confidence,
                language=language,
                words=words,
            ))
        return self._merge_fragments(result), language

    @staticmethod
    def _merge_fragments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Join decoder fragments only when punctuation and timing support it."""
        merged: list[TranscriptSegment] = []
        for incoming in segments:
            if not merged:
                merged.append(incoming)
                continue
            current = merged[-1]
            gap = incoming.start - current.end
            ends_sentence = current.text.rstrip().endswith((".", "?", "!"))
            starts_continuation = incoming.text[:1].islower() or not ends_sentence
            if current.speaker == incoming.speaker and 0 <= gap <= 0.35 and starts_continuation:
                current.text = f"{current.text.rstrip()} {incoming.text.lstrip()}".strip()
                current.end = max(current.end, incoming.end)
                current.confidence = min(current.confidence, incoming.confidence)
                current.words.extend(incoming.words)
            else:
                merged.append(incoming)
        return merged


class Vad:
    def __init__(self) -> None:
        self.s = get_vad_model().session
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.ctx = np.zeros((1, 64), np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.ctx, frame.reshape(1, -1)], 1)
        out, self.h, self.c = self.s.run(None, {"input": x, "h": self.h, "c": self.c})
        self.ctx = frame[-64:].reshape(1, -1).copy()
        return float(out[0])


class Segmenter:
    """Low-latency utterance segmenter; the extractor adds topic metadata later."""

    def __init__(self, config: Any | None = None) -> None:
        self.config = config
        self.vad = Vad()
        self.tail = np.zeros(0, np.float32)
        self.pre: deque[np.ndarray] = deque(maxlen=PREROLL)
        self.seg: list[np.ndarray] | None = None
        self.speech = 0
        self.silence = 0
        self.level = 0.0
        self.last_partial = 0
        self.sid = 0

    def feed(self, x: np.ndarray) -> list[tuple[str, np.ndarray, int]]:
        events: list[tuple[str, np.ndarray, int]] = []
        if x is None or not len(x):
            return events
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        buf = np.concatenate([self.tail, x]) if len(self.tail) else x
        count = len(buf) // FRAME
        for index in range(count):
            events.extend(self._frame(buf[index * FRAME:(index + 1) * FRAME]))
        self.tail = buf[count * FRAME:].copy()
        return events

    def _frame(self, frame: np.ndarray) -> list[tuple[str, np.ndarray, int]]:
        self.level = float(np.sqrt(np.mean(frame * frame)))
        probability = self.vad(frame)
        if self.seg is None:
            self.pre.append(frame)
            self.speech = self.speech + 1 if probability > VAD_ON else 0
            if self.speech >= START_FRAMES:
                self.seg = list(self.pre)
                self.pre.clear()
                self.speech = self.silence = self.last_partial = 0
                self.sid += 1
            return []
        self.seg.append(frame)
        self.silence = 0 if probability > VAD_OFF else self.silence + 1
        size = len(self.seg) * FRAME
        max_seconds = self.config.max_segment_seconds if self.config else MAX_SEG / SR
        if self.silence >= END_FRAMES or size >= max_seconds * SR:
            return self.flush()
        if size - self.last_partial >= PARTIAL_EVERY:
            self.last_partial = size
            return [("partial", np.concatenate(self.seg), self.sid)]
        return []

    def flush(self) -> list[tuple[str, np.ndarray, int]]:
        if self.seg is None:
            return []
        audio = np.concatenate(self.seg)
        self.seg = None
        self.silence = 0
        min_seconds = self.config.min_segment_seconds if self.config else MIN_SEG / SR
        if len(audio) < min_seconds * SR:
            return []
        return [("final", audio, self.sid)]
