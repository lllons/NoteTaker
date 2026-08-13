"""Configuration for the local-first NoteTaker pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    """A public Hugging Face model choice and CPU backend for transcription."""

    id: str
    label: str
    checkpoint: str
    repository: str
    beam_size: int
    temperatures: tuple[float, ...]
    description: str
    speed: str
    quality: str
    backend: str = "faster-whisper"
    parameters: str = ""
    compute_type: str = "int8"
    optional_requirements: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "checkpoint": self.checkpoint,
            "repository": self.repository,
            "beam_size": self.beam_size,
            "temperature_passes": list(self.temperatures),
            "description": self.description,
            "speed": self.speed,
            "quality": self.quality,
            "backend": self.backend,
            "parameters": self.parameters,
            "optional_requirements": self.optional_requirements,
            "device": "cpu",
            "compute_type": self.compute_type,
        }


# These are public Hugging Face checkpoints. The original entries are
# CTranslate2 repositories that faster-whisper loads directly; the three
# larger entries use the optional Transformers audio backend and still force
# CPU inference. Keeping the backend in profile metadata prevents the web
# selector from advertising a checkpoint the loader cannot actually use.
MODEL_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile("large-v3", "Large-v3 · Best overall", "large-v3", "https://huggingface.co/Systran/faster-whisper-large-v3", 8, (0.0,), "Official large-v3 conversion; strongest general-purpose option for preserving names, numbers, and technical speech.", "slowest", "best"),
    ModelProfile("large-v3-max", "Large-v3 · Maximum CPU decode", "large-v3", "https://huggingface.co/Systran/faster-whisper-large-v3", 20, (0.0, 0.15, 0.3, 0.5), "The same best checkpoint with the widest beam and fallback passes; use when latency does not matter.", "extreme", "maximum"),
    ModelProfile("qwen3-asr-1.7b", "Qwen3-ASR · 1.7B · Multilingual", "Qwen/Qwen3-ASR-1.7B-hf", "https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf", 1, (0.0,), "Dedicated multilingual ASR model with strong recognition of accents, technical speech, and long recordings.", "very slow", "excellent", backend="transformers-qwen3-asr", parameters="1.7B", compute_type="float32", optional_requirements="requirements-large-models.txt"),
    ModelProfile("voxtral-mini-3b", "Voxtral Mini · 3B · Transcription", "mistralai/Voxtral-Mini-3B-2507", "https://huggingface.co/mistralai/Voxtral-Mini-3B-2507", 1, (0.0,), "A 3B audio-language model with a dedicated transcription mode and long-form audio support.", "extreme", "very high", backend="transformers-voxtral", parameters="3B", compute_type="float32", optional_requirements="requirements-large-models.txt"),
    ModelProfile("qwen2-audio-7b", "Qwen2-Audio · 7B · Deep audio", "Qwen/Qwen2-Audio-7B-Instruct", "https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct", 1, (0.0,), "A 7B audio-language model prompted for verbatim speech transcription; best reserved for machines with substantial RAM.", "extreme", "very high", backend="transformers-qwen2-audio", parameters="7B", compute_type="float32", optional_requirements="requirements-large-models.txt"),
    ModelProfile("distil-large-v3", "Distil-large-v3 · High quality", "distil-large-v3", "https://huggingface.co/Systran/faster-distil-whisper-large-v3", 8, (0.0,), "Large distilled Whisper checkpoint designed for faster-whisper; faster and lighter with a quality trade-off.", "slow", "very high"),
    ModelProfile("large-v3-turbo", "Large-v3 Turbo · Faster", "large-v3-turbo", "https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2", 5, (0.0,), "Large-v3 Turbo CTranslate2 conversion; a practical speed choice when large-v3 is too slow.", "fast", "high"),
    ModelProfile("medium", "Medium · Multilingual", "medium", "https://huggingface.co/Systran/faster-whisper-medium", 8, (0.0,), "Official multilingual medium checkpoint for lower memory use while retaining broad language coverage.", "slow", "high"),
    ModelProfile("medium.en", "Medium English", "medium.en", "https://huggingface.co/Systran/faster-whisper-medium.en", 8, (0.0,), "English-only medium checkpoint; a strong CPU choice for clear English recordings.", "slow", "high"),
    ModelProfile("small", "Small · Multilingual", "small", "https://huggingface.co/Systran/faster-whisper-small", 8, (0.0,), "Official multilingual small checkpoint for a much lighter CPU footprint.", "medium", "good"),
    ModelProfile("small.en", "Small English", "small.en", "https://huggingface.co/Systran/faster-whisper-small.en", 8, (0.0,), "English-only small checkpoint for faster local capture.", "medium", "good"),
    ModelProfile("base", "Base · Multilingual", "base", "https://huggingface.co/Systran/faster-whisper-base", 8, (0.0,), "Lightweight multilingual fallback for machines that cannot hold a large model.", "fast", "moderate"),
    ModelProfile("base.en", "Base English", "base.en", "https://huggingface.co/Systran/faster-whisper-base.en", 8, (0.0,), "Lightweight English-only fallback.", "fast", "moderate"),
    ModelProfile("tiny", "Tiny · Multilingual", "tiny", "https://huggingface.co/Systran/faster-whisper-tiny", 8, (0.0,), "Smallest multilingual checkpoint; use only when CPU memory is severely constrained.", "fastest", "basic"),
    ModelProfile("tiny.en", "Tiny English", "tiny.en", "https://huggingface.co/Systran/faster-whisper-tiny.en", 8, (0.0,), "Smallest English-only checkpoint; not recommended when retaining every detail matters.", "fastest", "basic"),
)
MODEL_PROFILE_BY_ID = {profile.id: profile for profile in MODEL_PROFILES}
DEFAULT_MODEL_PROFILE_ID = "large-v3"

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
