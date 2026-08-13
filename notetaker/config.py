"""Configuration for the local-first NoteTaker pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    """A selectable CPU decoding profile for the web capture screen.

    ``large-v3`` is already the largest standard Whisper checkpoint used by
    this application. The higher profiles therefore spend more CPU time on
    beam search and fallback passes rather than claiming a nonexistent larger
    checkpoint. Keeping the checkpoint in the profile leaves room for adding
    another CPU-compatible model later without changing the WebSocket API.
    """

    id: str
    label: str
    checkpoint: str
    beam_size: int
    temperatures: tuple[float, ...]
    description: str
    speed: str
    quality: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "checkpoint": self.checkpoint,
            "beam_size": self.beam_size,
            "temperature_passes": list(self.temperatures),
            "description": self.description,
            "speed": self.speed,
            "quality": self.quality,
            "device": "cpu",
            "compute_type": "int8",
        }


MODEL_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        "model-1",
        "Model 1 · Current",
        "large-v3",
        8,
        (0.0,),
        "The current large-v3 behavior with the existing accuracy settings.",
        "slow",
        "high",
    ),
    ModelProfile(
        "model-2",
        "Model 2 · Careful",
        "large-v3",
        10,
        (0.0,),
        "A wider beam for fewer uncertain word choices at a modest CPU cost.",
        "slower",
        "higher",
    ),
    ModelProfile(
        "model-3",
        "Model 3 · High accuracy",
        "large-v3",
        12,
        (0.0, 0.15),
        "High-accuracy decoding with one conservative fallback pass.",
        "slower",
        "very high",
    ),
    ModelProfile(
        "model-4",
        "Model 4 · Deep accuracy",
        "large-v3",
        16,
        (0.0, 0.15, 0.3),
        "Deep CPU decoding for difficult speech, accents, and noisy recordings.",
        "very slow",
        "very high",
    ),
    ModelProfile(
        "model-5",
        "Model 5 · Maximum CPU accuracy",
        "large-v3",
        20,
        (0.0, 0.15, 0.3, 0.5),
        "The most thorough option: maximum beam width and fallback passes on CPU.",
        "slowest",
        "maximum",
    ),
)
MODEL_PROFILE_BY_ID = {profile.id: profile for profile in MODEL_PROFILES}
DEFAULT_MODEL_PROFILE_ID = "model-1"

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AppConfig:
    model: str = "large-v3"
    draft_model: str = "large-v3"
    language: str | None = None
    beam_size: int = 8
    threads: int = 0
    hotwords: str | None = None
    data_dir: Path = Path("data")
    diarization: str = "labels-only"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = ""
    context_chars: int = 1200
    max_segment_seconds: int = 30
    min_segment_seconds: float = 0.15
    vad_on_threshold: float = 0.35
    vad_off_threshold: float = 0.20
    preroll_seconds: float = 1.0
    end_silence_seconds: float = 1.4
    partial_seconds: float = 4.0
    provider_timeout: int = 45

    @classmethod
    def from_sources(cls, config_path: str | Path = "notetaker.toml") -> "AppConfig":
        values: dict[str, Any] = {}
        path = Path(config_path)
        if path.exists() and tomllib is not None:
            with path.open("rb") as handle:
                values = dict(tomllib.load(handle).get("notetaker", {}))

        def value(name: str, env_name: str, default: Any = None) -> Any:
            return os.getenv(env_name, values.get(name, default))

        language = value("language", "NOTE_TAKER_LANGUAGE", None)
        if language in ("", "auto", "None"):
            language = None
        vad_on_threshold = max(0.05, min(0.99, float(value("vad_on_threshold", "NOTE_TAKER_VAD_ON_THRESHOLD", cls.vad_on_threshold))))
        vad_off_threshold = max(0.02, min(vad_on_threshold - 0.01, float(value("vad_off_threshold", "NOTE_TAKER_VAD_OFF_THRESHOLD", cls.vad_off_threshold))))
        return cls(
            model=str(value("model", "NOTE_TAKER_MODEL", cls.model)),
            draft_model=str(value("draft_model", "NOTE_TAKER_DRAFT_MODEL", cls.draft_model)),
            language=language,
            beam_size=max(1, int(value("beam_size", "NOTE_TAKER_BEAM_SIZE", cls.beam_size))),
            threads=int(value("threads", "NOTE_TAKER_THREADS", cls.threads)),
            hotwords=value("hotwords", "NOTE_TAKER_HOTWORDS", None) or None,
            data_dir=Path(value("data_dir", "NOTE_TAKER_DATA_DIR", str(cls.data_dir))),
            diarization=str(value("diarization", "NOTE_TAKER_DIARIZATION", cls.diarization)),
            llm_base_url=value("llm_base_url", "NOTE_TAKER_LLM_BASE_URL", None) or None,
            llm_api_key=value("llm_api_key", "NOTE_TAKER_LLM_API_KEY", None) or None,
            llm_model=str(value("llm_model", "NOTE_TAKER_LLM_MODEL", "")),
            context_chars=max(200, int(value("context_chars", "NOTE_TAKER_CONTEXT_CHARS", cls.context_chars))),
            max_segment_seconds=max(5, int(value("max_segment_seconds", "NOTE_TAKER_MAX_SEGMENT_SECONDS", cls.max_segment_seconds))),
            min_segment_seconds=max(0.1, float(value("min_segment_seconds", "NOTE_TAKER_MIN_SEGMENT_SECONDS", cls.min_segment_seconds))),
            vad_on_threshold=vad_on_threshold,
            vad_off_threshold=vad_off_threshold,
            preroll_seconds=max(0.25, min(3.0, float(value("preroll_seconds", "NOTE_TAKER_PREROLL_SECONDS", cls.preroll_seconds)))),
            end_silence_seconds=max(0.5, min(4.0, float(value("end_silence_seconds", "NOTE_TAKER_END_SILENCE_SECONDS", cls.end_silence_seconds)))),
            partial_seconds=max(1.0, min(10.0, float(value("partial_seconds", "NOTE_TAKER_PARTIAL_SECONDS", cls.partial_seconds)))),
            provider_timeout=max(5, int(value("provider_timeout", "NOTE_TAKER_PROVIDER_TIMEOUT", cls.provider_timeout))),
        )
