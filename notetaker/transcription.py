"""High-accuracy CPU transcription primitives.

Audio intake and utterance boundaries live in :mod:`listening.audio`; this module
owns Whisper model loading, decoding, confidence metadata, and fragment merging.
The default is the full large-v3 model with CPU int8 inference because recall is
more important than real-time latency for this application.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from listening.audio import SR, Segmenter, Vad, resample_audio
from .models import TranscriptSegment, TranscriptWord

__all__ = [
    "MODELS",
    "SR",
    "Segmenter",
    "TranscriptionRuntime",
    "Vad",
    "resample_audio",
]


MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base.en": "Systran/faster-whisper-base.en",
    "small.en": "Systran/faster-whisper-small.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "medium.en": "Systran/faster-whisper-medium.en",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "large-v3": "Systran/faster-whisper-large-v3",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
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
    voices. A pyannote-compatible adapter can implement the same ``label``
    method later without changing the note schema.
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
        self._model_state = "not-loaded"
        self._model_error: str | None = None
        self.speakers = SpeakerLabeler(config.diarization)

    def _load(self, name: str) -> ModelBundle:
        repo = MODELS.get(name, name)
        model = WhisperModel(
            repo,
            device="cpu",
            compute_type="int8",
            cpu_threads=max(0, int(self.config.threads)),
        )
        return ModelBundle(model, threading.Lock(), ThreadPoolExecutor(max_workers=1))

    def status(self) -> dict[str, Any]:
        """Report configured and loaded models without exposing model internals."""
        with self._load_lock:
            loaded: list[str] = []
            if self.final is not None:
                loaded.append(self.config.model)
            if self.draft is not None and self.config.draft_model != self.config.model:
                loaded.append(self.config.draft_model)
            return {
                "state": self._model_state,
                "configured": {
                    "final": self.config.model,
                    "draft": self.config.draft_model,
                },
                "loaded": loaded,
                "device": "cpu",
                "compute_type": "int8",
                "error": self._model_error,
            }

    def ensure_loaded(self) -> None:
        with self._load_lock:
            if self.final is not None and self.draft is not None:
                self._model_state = "ready"
                return
            self._model_state = "loading"
            self._model_error = None
            try:
                if self.final is None:
                    self.final = self._load(self.config.model)
                if self.draft is None:
                    self.draft = self.final if self.config.draft_model == self.config.model else self._load(self.config.draft_model)
                self._model_state = "ready"
            except Exception as exc:
                self._model_state = "error"
                self._model_error = type(exc).__name__
                raise

    def transcribe(
        self,
        audio: np.ndarray,
        final: bool,
        context: str = "",
        offset: float = 0.0,
    ) -> tuple[list[TranscriptSegment], str | None]:
        """Decode an utterance without filtering valid short words or filler.

        The listening layer already found a speech boundary, so Whisper's own
        VAD is disabled here to avoid double filtering quiet words at the edges.
        ``final`` uses the configured wide beam; partials use the same loaded
        model with a smaller beam to keep the UI responsive when possible.
        """
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
            "Transcribe verbatim. Preserve every spoken word, technical term, name, acronym, number, unit, punctuation, equation, code command, URL, file path, and capitalization.",
            self.config.hotwords or "",
            context[-self.config.context_chars:],
        ]
        initial_prompt = " ".join(part for part in prompt_parts if part).strip()
        with bundle.lock:
            raw_segments, info = bundle.model.transcribe(
                audio,
                language=self.config.language,
                beam_size=max(1, self.config.beam_size if final else min(self.config.beam_size, 2)),
                temperature=0.0,
                initial_prompt=initial_prompt or None,
                hotwords=self.config.hotwords if final else None,
                condition_on_previous_text=False,
                without_timestamps=False,
                word_timestamps=True,
                vad_filter=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.8,
                log_prob_threshold=-1.2,
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
            compression_penalty = 0.35 if compression > 2.8 else 1.0
            confidence = max(0.0, min(1.0, (1.0 - no_speech) * log_confidence * compression_penalty))
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
