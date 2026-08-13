"""Audio intake primitives used by the live NoteTaker websocket.

The listening layer is deliberately independent from Whisper decoding. It accepts
browser PCM, normalizes its sample rate, keeps a generous pre-roll, and emits
utterances without throwing away the final partial frame. Raw transcript segments
remain the source of truth; the knowledge extractor can remove filler later.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

try:
    from faster_whisper.vad import get_vad_model
except Exception:  # pragma: no cover - depends on the installed faster-whisper build
    get_vad_model = None  # type: ignore[assignment]

SR = 16_000
FRAME = 512
DEFAULT_PREROLL_SECONDS = 1.0
DEFAULT_START_FRAMES = 2
DEFAULT_END_SILENCE_SECONDS = 1.4
DEFAULT_MAX_SEGMENT_SECONDS = 30.0
DEFAULT_MIN_SEGMENT_SECONDS = 0.15
DEFAULT_PARTIAL_SECONDS = 4.0
DEFAULT_VAD_ON = 0.35
DEFAULT_VAD_OFF = 0.20


def resample_audio(audio: np.ndarray, source_rate: float, target_rate: int = SR) -> np.ndarray:
    """Convert browser PCM to the target rate without dropping non-finite guards."""
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(values):
        return values
    if not np.isfinite(values).all():
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        rate = float(source_rate)
    except (TypeError, ValueError):
        rate = float(target_rate)
    if not np.isfinite(rate) or rate <= 0:
        rate = float(target_rate)
    if abs(rate - target_rate) < 0.5:
        return values
    output_length = max(1, int(round(len(values) * target_rate / rate)))
    if len(values) == 1:
        return np.repeat(values, output_length).astype(np.float32, copy=False)
    positions = np.linspace(0.0, len(values) - 1, output_length, dtype=np.float32)
    return np.interp(positions, np.arange(len(values), dtype=np.float32), values).astype(np.float32, copy=False)


class Vad:
    """CPU VAD wrapper with an energy fallback when ONNX VAD is unavailable."""

    def __init__(self) -> None:
        self.s = None
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.ctx = np.zeros((1, 64), np.float32)
        self.backend = "silero"
        self.error: str | None = None
        if get_vad_model is None:
            self._use_fallback("faster-whisper VAD module is unavailable")
            return
        try:
            self.s = get_vad_model().session
        except Exception as exc:
            self._use_fallback(f"{type(exc).__name__}: {str(exc)[:160]}")

    def _use_fallback(self, reason: str) -> None:
        self.s = None
        self.backend = "energy-fallback"
        self.error = reason

    @staticmethod
    def _energy_probability(frame: np.ndarray) -> float:
        level = float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0
        # This is deliberately conservative: Segmenter also keeps a noise
        # floor and a separate energy hint, so the fallback cannot silently
        # discard audio when Silero/onnxruntime is missing.
        return max(0.01, min(0.99, (level - 0.004) / 0.017))

    def __call__(self, frame: np.ndarray) -> float:
        if self.s is None:
            return self._energy_probability(frame)
        try:
            x = np.concatenate([self.ctx, frame.reshape(1, -1)], 1)
            out, self.h, self.c = self.s.run(None, {"input": x, "h": self.h, "c": self.c})
            self.ctx = frame[-64:].reshape(1, -1).copy()
            return float(out[0])
        except Exception as exc:
            self._use_fallback(f"{type(exc).__name__}: {str(exc)[:160]}")
            return self._energy_probability(frame)


class Segmenter:
    """Emit partial/final utterances while favoring recall at speech boundaries."""

    def __init__(self, config: Any | None = None) -> None:
        self.config = config
        self.vad = Vad()
        self.tail = np.zeros(0, np.float32)
        preroll_seconds = max(
            0.25,
            float(getattr(config, "preroll_seconds", DEFAULT_PREROLL_SECONDS)) if config else DEFAULT_PREROLL_SECONDS,
        )
        self.pre: deque[np.ndarray] = deque(maxlen=max(1, int(round(preroll_seconds * SR / FRAME))))
        self.seg: list[np.ndarray] | None = None
        self.speech = 0
        self.silence = 0
        self.level = 0.0
        self.noise_floor = 0.008
        self.last_partial = 0
        self.sid = 0
        self.start_frames = max(1, int(getattr(config, "start_frames", DEFAULT_START_FRAMES))) if config else DEFAULT_START_FRAMES
        self.end_frames = max(
            1,
            int(round(
                (float(getattr(config, "end_silence_seconds", DEFAULT_END_SILENCE_SECONDS)) if config else DEFAULT_END_SILENCE_SECONDS)
                * SR / FRAME
            )),
        )
        self.max_segment_seconds = max(
            5.0,
            float(getattr(config, "max_segment_seconds", DEFAULT_MAX_SEGMENT_SECONDS)) if config else DEFAULT_MAX_SEGMENT_SECONDS,
        )
        self.min_segment_seconds = max(
            0.1,
            float(getattr(config, "min_segment_seconds", DEFAULT_MIN_SEGMENT_SECONDS)) if config else DEFAULT_MIN_SEGMENT_SECONDS,
        )
        self.partial_every = max(
            FRAME,
            int(round(
                (float(getattr(config, "partial_seconds", DEFAULT_PARTIAL_SECONDS)) if config else DEFAULT_PARTIAL_SECONDS)
                * SR
            )),
        )
        self.vad_on = max(0.05, min(0.99, float(getattr(config, "vad_on_threshold", DEFAULT_VAD_ON)) if config else DEFAULT_VAD_ON))
        self.vad_off = max(0.02, min(self.vad_on - 0.01, float(getattr(config, "vad_off_threshold", DEFAULT_VAD_OFF)) if config else DEFAULT_VAD_OFF))

    @property
    def vad_status(self) -> dict[str, Any]:
        return {
            "backend": self.vad.backend,
            "fallback": self.vad.backend != "silero",
            "error": self.vad.error,
        }

    def feed(self, x: np.ndarray) -> list[tuple[str, np.ndarray, int]]:
        events: list[tuple[str, np.ndarray, int]] = []
        if x is None or not len(x):
            return events
        values = np.asarray(x, dtype=np.float32).reshape(-1)
        if not np.isfinite(values).all():
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        buf = np.concatenate([self.tail, values]) if len(self.tail) else values
        count = len(buf) // FRAME
        for index in range(count):
            events.extend(self._frame(buf[index * FRAME:(index + 1) * FRAME]))
        self.tail = buf[count * FRAME:].copy()
        return events

    def _energy_suggests_speech(self) -> bool:
        # A quiet speaker can have a modest Silero score. A bounded energy hint
        # improves recall without allowing a constant low-level noise floor to
        # hold an utterance open forever.
        return self.level > max(0.012, self.noise_floor * 2.25)

    def _frame(self, frame: np.ndarray) -> list[tuple[str, np.ndarray, int]]:
        self.level = float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0
        if self.seg is None:
            self.noise_floor = min(0.08, self.noise_floor * 0.995 + self.level * 0.005)
        probability = self.vad(frame)
        energy_speech = self._energy_suggests_speech()
        if self.seg is None:
            self.pre.append(frame.copy())
            self.speech = self.speech + 1 if probability > self.vad_on or energy_speech else 0
            if self.speech >= self.start_frames:
                self.seg = list(self.pre)
                self.pre.clear()
                self.speech = self.silence = self.last_partial = 0
                self.sid += 1
            return []

        self.seg.append(frame.copy())
        self.silence = 0 if probability > self.vad_off or energy_speech else self.silence + 1
        size = len(self.seg) * FRAME
        if self.silence >= self.end_frames or size >= self.max_segment_seconds * SR:
            return self.flush()
        if size - self.last_partial >= self.partial_every:
            self.last_partial = size
            return [("partial", np.concatenate(self.seg), self.sid)]
        return []

    def flush(self) -> list[tuple[str, np.ndarray, int]]:
        """Flush the active utterance, including a final short browser frame."""
        if self.seg is None:
            self.tail = np.zeros(0, np.float32)
            self.pre.clear()
            return []
        if len(self.tail):
            self.seg.append(self.tail.copy())
        self.tail = np.zeros(0, np.float32)
        audio = np.concatenate(self.seg)
        self.seg = None
        self.speech = 0
        self.silence = 0
        self.last_partial = 0
        if len(audio) < self.min_segment_seconds * SR:
            return []
        return [("final", audio, self.sid)]
