"""Audio intake primitives used by the live NoteTaker websocket.

The listening layer is deliberately independent from Whisper decoding. It accepts
browser PCM, normalizes its sample rate, keeps a generous pre-roll, and emits
utterances without throwing away the final partial frame. Raw transcript segments
remain the source of truth; the knowledge extractor can remove filler later.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction
import logging
from typing import Any

import numpy as np

try:
    from scipy.signal import resample_poly
except Exception:  # pragma: no cover - the lightweight source tree can be imported before install
    resample_poly = None  # type: ignore[assignment]

try:
    from faster_whisper.vad import get_vad_model
except Exception:  # pragma: no cover - depends on the installed faster-whisper build
    get_vad_model = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

SR = 16_000
FRAME = 512
DEFAULT_PREROLL_SECONDS = 1.0
DEFAULT_START_FRAMES = 2
DEFAULT_END_SILENCE_SECONDS = 1.4
DEFAULT_MAX_SEGMENT_SECONDS = 14.0
DEFAULT_SOFT_MAX_SECONDS = 8.0
DEFAULT_MIN_SEGMENT_SECONDS = 0.15
DEFAULT_PARTIAL_SECONDS = 4.0
DEFAULT_VAD_ON = 0.35
DEFAULT_VAD_OFF = 0.20


def _fft_lowpass(values: np.ndarray, source_rate: float, target_rate: int) -> np.ndarray:
    """Small numpy-only anti-alias fallback used when scipy is not installed."""
    if source_rate <= target_rate or len(values) < 4:
        return values
    spectrum = np.fft.rfft(values)
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / source_rate)
    passband = 0.40 * target_rate
    stopband = 0.50 * target_rate
    mask = np.ones_like(frequencies, dtype=np.float32)
    mask[frequencies >= stopband] = 0.0
    transition = (frequencies > passband) & (frequencies < stopband)
    mask[transition] = 0.5 * (
        1.0 + np.cos(np.pi * (frequencies[transition] - passband) / (stopband - passband))
    )
    return np.fft.irfft(spectrum * mask, n=len(values)).astype(np.float32, copy=False)


def resample_audio(audio: np.ndarray, source_rate: float, target_rate: int = SR) -> np.ndarray:
    """Convert browser PCM to the target rate with an anti-aliasing filter."""
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

    if resample_poly is not None:
        ratio = Fraction(target_rate / rate).limit_denominator(1000)
        filtered = resample_poly(values, ratio.numerator, ratio.denominator)
        if len(filtered) >= output_length:
            return np.asarray(filtered[:output_length], dtype=np.float32)
        return np.pad(
            np.asarray(filtered, dtype=np.float32),
            (0, output_length - len(filtered)),
            mode="edge",
        )

    filtered = _fft_lowpass(values, rate, target_rate)
    positions = np.linspace(0.0, len(filtered) - 1, output_length, dtype=np.float32)
    return np.interp(positions, np.arange(len(filtered), dtype=np.float32), filtered).astype(np.float32, copy=False)


class Vad:
    """CPU VAD wrapper with an energy fallback when ONNX VAD is unavailable."""

    def __init__(self) -> None:
        self.s = None
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros((1, 64), np.float32)
        self.backend = "silero"
        self.error: str | None = None
        self.input_name = "input"
        self.signature = ""
        self.h_name = "h"
        self.c_name = "c"
        self.state_name = "state"
        self.sr_name: str | None = None
        self._input_info: dict[str, Any] = {}
        if get_vad_model is None:
            self._use_fallback("faster-whisper VAD module is unavailable")
            return
        try:
            self.s = get_vad_model().session
            self._configure_session(self.s)
        except ValueError as exc:
            reason = f"Unsupported Silero VAD input signature: {str(exc)[:160]}"
            self._use_fallback(reason)
            raise RuntimeError(reason) from exc
        except Exception as exc:
            self._use_fallback(f"{type(exc).__name__}: {str(exc)[:160]}")

    @staticmethod
    def _shape_or_default(info: Any, default: tuple[int, ...]) -> tuple[int, ...]:
        shape = getattr(info, "shape", None)
        if not shape or any(not isinstance(value, (int, np.integer)) or int(value) <= 0 for value in shape):
            return default
        return tuple(int(value) for value in shape)

    def _configure_session(self, session: Any) -> None:
        inputs = list(session.get_inputs())
        self._input_info = {str(item.name): item for item in inputs}
        names = set(self._input_info)
        if "input" not in names:
            raise ValueError(f"Unsupported Silero VAD signature: missing input; found {sorted(names)}")
        self.input_name = "input"
        if {"h", "c"}.issubset(names):
            self.signature = "h-c"
            self.h_name = "h"
            self.c_name = "c"
            self.h = np.zeros(self._shape_or_default(self._input_info["h"], (1, 1, 128)), np.float32)
            self.c = np.zeros(self._shape_or_default(self._input_info["c"], (1, 1, 128)), np.float32)
            return
        if "state" in names:
            self.signature = "state-sr"
            self.state_name = "state"
            state_shape = self._shape_or_default(self._input_info["state"], (2, 1, 128))
            self.state = np.zeros(state_shape, np.float32)
            self.sr_name = "sr" if "sr" in names else None
            return
        raise ValueError(
            "Unsupported Silero VAD signature: expected h/c or state inputs; "
            f"found {sorted(names)}"
        )

    def _use_fallback(self, reason: str) -> None:
        self.s = None
        self.backend = "energy-fallback"
        self.error = reason
        logger.warning("Silero VAD unavailable; using energy fallback: %s", reason)

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
            if self.signature == "h-c":
                output = self.s.run(
                    None,
                    {self.input_name: x, self.h_name: self.h, self.c_name: self.c},
                )
                if len(output) < 3:
                    raise RuntimeError("Silero VAD h/c signature returned fewer than three outputs")
                out, self.h, self.c = output[0], output[1], output[2]
            elif self.signature == "state-sr":
                feeds: dict[str, Any] = {self.input_name: x, self.state_name: self.state}
                if self.sr_name:
                    sr_info = self._input_info[self.sr_name]
                    sr_shape = getattr(sr_info, "shape", None)
                    feeds[self.sr_name] = np.asarray(
                        [SR] if sr_shape and len(sr_shape) > 0 else SR,
                        dtype=np.int64,
                    )
                output = self.s.run(None, feeds)
                if len(output) < 2:
                    raise RuntimeError("Silero VAD state signature returned fewer than two outputs")
                out, self.state = output[0], output[1]
            else:
                raise RuntimeError("Silero VAD session signature was not configured")
            self.ctx = frame[-64:].reshape(1, -1).copy()
            return float(np.asarray(out).reshape(-1)[0])
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
        self.pre_probs: deque[float] = deque(maxlen=self.pre.maxlen)
        self.seg: list[np.ndarray] | None = None
        self.seg_probs: list[float] = []
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
        self.soft_max_seconds = max(
            4.0,
            min(
                float(getattr(config, "soft_max_seconds", DEFAULT_SOFT_MAX_SECONDS)) if config else DEFAULT_SOFT_MAX_SECONDS,
                self.max_segment_seconds - 1.0,
            ),
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
        backend = getattr(self.vad, "backend", "unknown")
        return {
            "backend": backend,
            "fallback": backend != "silero",
            "error": getattr(self.vad, "error", None),
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
        # A quiet speaker can have a modest Silero score. The energy hint is
        # only used to open a segment; VAD probability alone controls hangover.
        return self.level > max(0.012, self.noise_floor * 2.25)

    def _log_events(self, events: list[tuple[str, np.ndarray, int]]) -> list[tuple[str, np.ndarray, int]]:
        for kind, audio, sid in events:
            logger.info(
                "segment emitted kind=%s id=%s duration=%.2fs vad_backend=%s",
                kind,
                sid,
                len(audio) / SR,
                getattr(self.vad, "backend", "unknown"),
            )
        return events

    def _soft_split(self) -> list[tuple[str, np.ndarray, int]]:
        assert self.seg is not None
        if not self.seg:
            return []
        window_frames = max(1, int(round(SR / FRAME)))
        window_start = max(0, len(self.seg_probs) - window_frames)
        cut = window_start + int(np.argmin(self.seg_probs[window_start:]))
        minimum_frames = max(1, int(np.ceil(self.min_segment_seconds * SR / FRAME)))
        cut = max(cut, minimum_frames - 1)
        prefix = np.concatenate(self.seg[:cut + 1])
        remainder = self.seg[cut + 1:]
        remainder_probs = self.seg_probs[cut + 1:]
        sid = self.sid
        if remainder:
            self.seg = remainder
            self.seg_probs = remainder_probs
            self.sid += 1
            self.silence = 0
            self.last_partial = 0
        else:
            self.seg = None
            self.seg_probs = []
            self.pre.clear()
            self.pre_probs.clear()
            self.speech = self.silence = self.last_partial = 0
        if len(prefix) < self.min_segment_seconds * SR:
            return []
        return self._log_events([("final", prefix, sid)])

    def _frame(self, frame: np.ndarray) -> list[tuple[str, np.ndarray, int]]:
        self.level = float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0
        # Keep adapting through an open utterance, but much more slowly so a
        # sustained voice cannot make the floor chase the speech envelope.
        alpha = 0.001 if self.seg is not None else 0.005
        self.noise_floor = min(0.08, self.noise_floor * (1.0 - alpha) + self.level * alpha)
        probability = self.vad(frame)
        energy_speech = self._energy_suggests_speech()
        if self.seg is None:
            self.pre.append(frame.copy())
            self.pre_probs.append(probability)
            self.speech = self.speech + 1 if probability > self.vad_on or energy_speech else 0
            if self.speech >= self.start_frames:
                self.seg = list(self.pre)
                self.seg_probs = list(self.pre_probs)
                self.pre.clear()
                self.pre_probs.clear()
                self.speech = self.silence = self.last_partial = 0
                self.sid += 1
            return []

        self.seg.append(frame.copy())
        self.seg_probs.append(probability)
        # Do not OR the energy hint here: room noise must be allowed to close a
        # live utterance once Silero reports a pause.
        self.silence = 0 if probability > self.vad_off else self.silence + 1
        size = len(self.seg) * FRAME
        if self.silence >= self.end_frames:
            return self.flush()
        if size >= self.soft_max_seconds * SR:
            return self._soft_split()
        if size >= self.max_segment_seconds * SR:
            return self.flush()
        if size - self.last_partial >= self.partial_every:
            self.last_partial = size
            return self._log_events([("partial", np.concatenate(self.seg), self.sid)])
        return []

    def flush(self) -> list[tuple[str, np.ndarray, int]]:
        """Flush the active utterance, including a final short browser frame."""
        if self.seg is None:
            self.tail = np.zeros(0, np.float32)
            self.pre.clear()
            self.pre_probs.clear()
            return []
        if len(self.tail):
            self.seg.append(self.tail.copy())
        self.tail = np.zeros(0, np.float32)
        audio = np.concatenate(self.seg)
        self.seg = None
        self.seg_probs = []
        self.pre.clear()
        self.pre_probs.clear()
        self.speech = 0
        self.silence = 0
        self.last_partial = 0
        if len(audio) < self.min_segment_seconds * SR:
            return []
        return self._log_events([("final", audio, self.sid)])
