"""High-accuracy CPU transcription primitives.

Audio intake and utterance boundaries live in :mod:`listening.audio`; this module
owns Whisper model loading, decoding, confidence metadata, and fragment merging.
The live default is large-v3 Turbo with CPU int8 inference and a dedicated tiny.en
draft model; large-v3 remains available for offline-quality transcription.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - offline tests can exercise prompting without model downloads
    WhisperModel = None  # type: ignore[assignment]

from listening.audio import SR, Segmenter, Vad, resample_audio
from .config import MODEL_PROFILE_BY_ID, MODEL_PROFILES, ModelProfile
from .models import TranscriptSegment, TranscriptWord

__all__ = [
    "MODELS",
    "MODEL_PROFILES",
    "SR",
    "Segmenter",
    "TranscriptionRuntime",
    "Vad",
    "resample_audio",
]


logger = logging.getLogger(__name__)


MODELS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "large-v3": "Systran/faster-whisper-large-v3",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
}


@dataclass
class ModelBundle:
    model: Any
    lock: threading.Lock
    executor: ThreadPoolExecutor


def _language_hint(language: str | None) -> str | None:
    """Return a language value accepted by the larger Transformers ASR models."""
    if not language:
        return None
    names = {
        "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish",
        "fr": "French", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
        "ko": "Korean", "nl": "Dutch", "pt": "Portuguese", "ru": "Russian",
        "th": "Thai", "tr": "Turkish", "vi": "Vietnamese", "zh": "Chinese",
    }
    return names.get(language.casefold(), language)


def _move_inputs_to_cpu(inputs: Any, torch: Any) -> Any:
    """Move a Transformers BatchFeature to CPU while preserving integer IDs."""
    try:
        return inputs.to("cpu", dtype=torch.float32)
    except (AttributeError, TypeError):
        if hasattr(inputs, "items"):
            for key, value in inputs.items():
                if not hasattr(value, "to"):
                    continue
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    inputs[key] = value.to(device="cpu", dtype=torch.float32)
                else:
                    inputs[key] = value.to(device="cpu")
        return inputs


def _load_cpu_transformers_model(model_class: Any, checkpoint: str, torch: Any) -> Any:
    """Load a Transformers model without device-map or GPU assumptions."""
    try:
        model = model_class.from_pretrained(checkpoint, dtype=torch.float32)
    except TypeError:
        # Older Transformers releases use torch_dtype instead of dtype.
        model = model_class.from_pretrained(checkpoint, torch_dtype=torch.float32)
    return model.to("cpu").eval()


class _Qwen3ASRAdapter:
    def __init__(self, model: Any, processor: Any, torch: Any) -> None:
        self.model = model
        self.processor = processor
        self.torch = torch

    def transcribe(self, audio: np.ndarray, language: str | None, prompt: str) -> tuple[str, str | None]:
        kwargs: dict[str, Any] = {"audio": audio, "sampling_rate": SR}
        if language:
            kwargs["language"] = _language_hint(language)
        if prompt:
            kwargs["prompt"] = prompt
        inputs = self.processor.apply_transcription_request(**kwargs)
        inputs = _move_inputs_to_cpu(inputs, self.torch)
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = self.processor.decode(generated_ids, return_format="parsed")[0]
        if isinstance(parsed, dict):
            return str(parsed.get("transcription", "") or "").strip(), parsed.get("language")
        return self.processor.decode(generated_ids, return_format="transcription_only")[0].strip(), language


class _VoxtralAdapter:
    def __init__(self, model: Any, processor: Any, torch: Any, checkpoint: str) -> None:
        self.model = model
        self.processor = processor
        self.torch = torch
        self.checkpoint = checkpoint

    def transcribe(self, audio: np.ndarray, language: str | None, prompt: str) -> tuple[str, str | None]:
        inputs = self.processor.apply_transcription_request(
            audio=audio,
            sampling_rate=SR,
            format="WAV",
            model_id=self.checkpoint,
            language=_language_hint(language),
        )
        inputs = _move_inputs_to_cpu(inputs, self.torch)
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return text, language


class _Qwen2AudioAdapter:
    def __init__(self, model: Any, processor: Any, torch: Any) -> None:
        self.model = model
        self.processor = processor
        self.torch = torch

    def transcribe(self, audio: np.ndarray, language: str | None, prompt: str) -> tuple[str, str | None]:
        content: list[dict[str, str]] = [{"type": "audio", "audio_url": "local-audio"}]
        if prompt:
            content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audios=[audio], sampling_rate=SR, return_tensors="pt", padding=True)
        inputs = _move_inputs_to_cpu(inputs, self.torch)
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return response, language


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
        # status() only takes this short state lock; it never waits for model
        # download or initialization on the ASGI event loop.
        self._state_lock = threading.Lock()
        self._model_state = "not-loaded"
        self._model_error: str | None = None
        self._loaded_checkpoint: str | None = None
        self._loaded_draft_checkpoint: str | None = None
        self._benchmark: dict[str, Any] | None = None
        self._last_decode: dict[str, Any] = {}
        self._decode_lock = threading.Lock()
        configured_model = str(getattr(config, "model", "") or "")
        self._profile = next(
            (profile for profile in MODEL_PROFILES if profile.id == configured_model or profile.checkpoint == configured_model),
            self._custom_profile(configured_model),
        )
        self._draft_checkpoint = str(getattr(config, "draft_model", "tiny.en") or "tiny.en")
        self._configured_draft_checkpoint = self._draft_checkpoint
        self.speakers = SpeakerLabeler(config.diarization)

    @staticmethod
    def _custom_profile(checkpoint: str) -> ModelProfile:
        return ModelProfile(
            "custom",
            f"Custom · {checkpoint or 'configured model'}",
            checkpoint or "large-v3",
            "",
            8,
            (0.0,),
            "The model configured by the command line or environment.",
            "configured",
            "configured",
        )

    def model_options(self) -> list[dict[str, Any]]:
        """Return the CPU model catalog used by the web selector."""
        return [profile.to_dict() for profile in MODEL_PROFILES]

    def select_model(self, profile_id: str) -> dict[str, Any]:
        """Select a web profile before capture; every profile remains CPU-only."""
        profile = MODEL_PROFILE_BY_ID.get(str(profile_id))
        if profile is None:
            raise ValueError(f"unknown model profile: {profile_id}")
        with self._load_lock:
            checkpoint_changed = (
                self._loaded_checkpoint is not None
                and (
                    self._loaded_checkpoint != profile.checkpoint
                    or self._loaded_draft_checkpoint != self._configured_draft_checkpoint
                )
            )
            if checkpoint_changed:
                self._close_bundle(self.final)
                self._close_bundle(self.draft)
                self.final = None
                self.draft = None
                self._loaded_checkpoint = None
                self._loaded_draft_checkpoint = None
            self._profile = profile
            # Keep a dedicated small draft checkpoint for every selected final
            # model. The two bundles are intentionally never shared.
            self._draft_checkpoint = self._configured_draft_checkpoint
            self._benchmark = None
            with self._state_lock:
                self._model_error = None
                self._model_state = "ready" if self.final is not None and self.draft is not None else "not-loaded"
        return self.status()

    @staticmethod
    def _close_bundle(bundle: ModelBundle | None) -> None:
        if bundle is None:
            return
        try:
            bundle.executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # pragma: no cover - Python 3.8 compatibility
            bundle.executor.shutdown(wait=False)

    def _profile_for_checkpoint(self, checkpoint: str) -> ModelProfile | None:
        return next(
            (profile for profile in MODEL_PROFILES if profile.id == checkpoint or profile.checkpoint == checkpoint),
            None,
        )

    def _load_transformers(self, profile: ModelProfile) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                f"{profile.label} requires the optional CPU model dependencies. "
                f"Install with `python -m pip install -r {profile.optional_requirements or 'requirements-large-models.txt'}`."
            ) from exc
        if int(self.config.threads) > 0:
            torch.set_num_threads(int(self.config.threads))
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(profile.checkpoint)
            if profile.backend == "transformers-qwen3-asr":
                from transformers import AutoModelForMultimodalLM
                model = _load_cpu_transformers_model(AutoModelForMultimodalLM, profile.checkpoint, torch)
                return _Qwen3ASRAdapter(model, processor, torch)
            if profile.backend == "transformers-voxtral":
                from transformers import VoxtralForConditionalGeneration
                model = _load_cpu_transformers_model(VoxtralForConditionalGeneration, profile.checkpoint, torch)
                return _VoxtralAdapter(model, processor, torch, profile.checkpoint)
            if profile.backend == "transformers-qwen2-audio":
                from transformers import Qwen2AudioForConditionalGeneration
                model = _load_cpu_transformers_model(Qwen2AudioForConditionalGeneration, profile.checkpoint, torch)
                return _Qwen2AudioAdapter(model, processor, torch)
        except ImportError as exc:
            raise RuntimeError(
                f"{profile.label} requires Transformers audio support. "
                f"Install with `python -m pip install -r {profile.optional_requirements or 'requirements-large-models.txt'}`."
            ) from exc
        raise RuntimeError(f"Unsupported model backend: {profile.backend}")

    def _load(self, name: str, profile: ModelProfile | None = None) -> ModelBundle:
        profile = profile or self._profile_for_checkpoint(name)
        if profile is not None and profile.backend != "faster-whisper":
            model = self._load_transformers(profile)
        else:
            if WhisperModel is None:
                raise RuntimeError("faster-whisper is not installed; install requirements.txt before loading a transcription model")
            repo = MODELS.get(name, name)
            model = WhisperModel(
                repo,
                device="cpu",
                compute_type="int8",
                cpu_threads=max(0, int(self.config.threads)),
            )
        return ModelBundle(model, threading.Lock(), ThreadPoolExecutor(max_workers=1))

    def _set_model_state(self, state: str, error: str | None = None) -> None:
        with self._state_lock:
            self._model_state = state
            self._model_error = error

    @property
    def last_decode(self) -> dict[str, Any]:
        with self._decode_lock:
            return dict(self._last_decode)

    def status(self) -> dict[str, Any]:
        """Report status without waiting for model download or initialization."""
        # Do not acquire _load_lock here. This method is called by FastAPI
        # handlers and the websocket receive loop while ensure_loaded() may be
        # downloading several gigabytes in a worker thread.
        with self._state_lock:
            profile = self._profile
            draft_checkpoint = self._draft_checkpoint
            loaded_checkpoint = self._loaded_checkpoint
            loaded_draft_checkpoint = self._loaded_draft_checkpoint
            state = self._model_state
            error = self._model_error
            benchmark = dict(self._benchmark) if self._benchmark else None
            final_loaded = self.final is not None
            draft_loaded = self.draft is not None
        loaded: list[str] = []
        if final_loaded and loaded_checkpoint == profile.checkpoint:
            loaded.append(profile.checkpoint)
        if draft_loaded and loaded_draft_checkpoint == draft_checkpoint:
            loaded.append(draft_checkpoint)
        return {
            "state": state,
            "profile_id": profile.id,
            "profile": profile.to_dict(),
            "configured": {
                "final": profile.checkpoint,
                "draft": draft_checkpoint,
            },
            "loaded": loaded,
            "device": "cpu",
            "compute_type": profile.compute_type,
            "backend": profile.backend,
            "error": error,
            "benchmark": benchmark,
            "last_decode": self.last_decode,
        }

    @staticmethod
    def _faster_model_label(profile: ModelProfile) -> str:
        preferred = ("large-v3-turbo", "distil-large-v3", "small.en", "tiny.en")
        for profile_id in preferred:
            if profile_id != profile.id:
                candidate = MODEL_PROFILE_BY_ID.get(profile_id)
                if candidate:
                    return candidate.label
        return "a faster model"

    def _run_startup_benchmark(self, bundle: ModelBundle, profile: ModelProfile) -> dict[str, Any]:
        """Measure a short local decode before the first spoken segment."""
        audio_seconds = max(1.0, min(10.0, float(getattr(self.config, "benchmark_seconds", 5.0))))
        sample_count = int(round(audio_seconds * SR))
        timeline = np.arange(sample_count, dtype=np.float32) / SR
        audio = (
            0.02 * np.sin(2.0 * np.pi * 220.0 * timeline)
            + 0.01 * np.sin(2.0 * np.pi * 330.0 * timeline)
        ).astype(np.float32)
        started = time.perf_counter()
        try:
            with bundle.lock:
                if profile.backend != "faster-whisper":
                    bundle.model.transcribe(audio, self.config.language, "")
                else:
                    raw_segments, _info = bundle.model.transcribe(
                        audio,
                        language=self.config.language,
                        beam_size=1,
                        temperature=0.0,
                        initial_prompt=None,
                        hotwords=self.config.hotwords or None,
                        condition_on_previous_text=False,
                        without_timestamps=True,
                        word_timestamps=False,
                        vad_filter=False,
                    )
                    list(raw_segments)
            decode_seconds = max(0.0, time.perf_counter() - started)
            rtf = decode_seconds / audio_seconds
            warning = rtf > 0.8
            message = None
            if warning:
                message = (
                    f"{profile.label} decodes ~{rtf:.1f}× slower than real time on this machine; "
                    f"switch to {self._faster_model_label(profile)} or expect long delays."
                )
                logger.warning("startup benchmark warning: %s", message)
            result = {
                "state": "ready",
                "model": profile.id,
                "audio_seconds": audio_seconds,
                "decode_seconds": decode_seconds,
                "rtf": rtf,
                "warning": warning,
                "message": message,
            }
            logger.info(
                "startup benchmark model=%s audio_seconds=%.2f decode_seconds=%.2f rtf=%.3f",
                profile.id,
                audio_seconds,
                decode_seconds,
                rtf,
            )
            return result
        except Exception as exc:
            logger.warning("startup benchmark failed model=%s error=%s", profile.id, exc)
            return {
                "state": "error",
                "model": profile.id,
                "audio_seconds": audio_seconds,
                "decode_seconds": None,
                "rtf": None,
                "warning": False,
                "message": f"Startup benchmark unavailable: {type(exc).__name__}",
            }

    def ensure_loaded(self) -> None:
        with self._load_lock:
            target_checkpoint = self._profile.checkpoint
            target_draft = self._draft_checkpoint
            if (
                self.final is not None
                and self.draft is not None
                and self._loaded_checkpoint == target_checkpoint
                and self._loaded_draft_checkpoint == target_draft
            ):
                self._set_model_state("ready")
                if self._benchmark is None:
                    benchmark = self._run_startup_benchmark(self.final, self._profile)
                    with self._state_lock:
                        self._benchmark = benchmark
                return
            started = time.perf_counter()
            logger.info(
                "model load start final=%s draft=%s device=cpu compute_type=%s",
                target_checkpoint,
                target_draft,
                self._profile.compute_type,
            )
            self._set_model_state("loading")
            try:
                if self.final is None:
                    self.final = self._load(target_checkpoint, self._profile)
                if self.draft is None:
                    draft_profile = self._profile_for_checkpoint(target_draft)
                    # Never alias final and draft, even when a caller explicitly
                    # requests the same checkpoint for both paths.
                    self.draft = self._load(target_draft, draft_profile)
                benchmark = self._run_startup_benchmark(self.final, self._profile)
                with self._state_lock:
                    self._loaded_checkpoint = target_checkpoint
                    self._loaded_draft_checkpoint = target_draft
                    self._benchmark = benchmark
                self._set_model_state("ready")
                logger.info(
                    "model load finish final=%s draft=%s elapsed_seconds=%.2f",
                    target_checkpoint,
                    target_draft,
                    time.perf_counter() - started,
                )
            except Exception as exc:
                self._set_model_state("error", type(exc).__name__)
                logger.exception("model load failed final=%s draft=%s", target_checkpoint, target_draft)
                raise

    def _record_decode(self, metadata: dict[str, Any]) -> None:
        with self._decode_lock:
            self._last_decode = dict(metadata)

    def _finish_decode(
        self,
        final: bool,
        audio_seconds: float,
        started: float,
        result: list[TranscriptSegment],
        language: str | None,
        rms: float,
        peak: float,
        no_speech_prob: float | None,
        avg_logprob: float | None,
    ) -> None:
        decode_seconds = max(0.0, time.perf_counter() - started)
        metadata = {
            "event_type": "final" if final else "partial",
            "audio_seconds": audio_seconds,
            "decode_seconds": decode_seconds,
            "rtf": decode_seconds / audio_seconds if audio_seconds else None,
            "segments_returned": len(result),
            "language": language,
            "rms": rms,
            "peak": peak,
            "no_speech_prob": no_speech_prob,
            "avg_logprob": avg_logprob,
        }
        self._record_decode(metadata)
        logger.info(
            "decode kind=%s audio_seconds=%.2f decode_seconds=%.2f rtf=%s segments_returned=%d language=%s",
            metadata["event_type"],
            audio_seconds,
            decode_seconds,
            f"{metadata['rtf']:.3f}" if metadata["rtf"] is not None else "n/a",
            len(result),
            language or "auto",
        )
        if not result:
            logger.warning(
                "decode returned zero segments no_speech_prob=%s avg_logprob=%s audio_seconds=%.2f rms=%.6f peak=%.6f",
                no_speech_prob,
                avg_logprob,
                audio_seconds,
                rms,
                peak,
            )

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
        ``final`` uses the configured beam; partials use the dedicated small
        draft model with a smaller beam so draft work cannot block finals.
        """
        self.ensure_loaded()
        with self._load_lock:
            profile = self._profile
            bundle = self.final if final else self.draft
        assert bundle is not None
        decode_started = time.perf_counter()
        if audio is None or not len(audio):
            metadata = {
                "event_type": "final" if final else "partial",
                "audio_seconds": 0.0,
                "decode_seconds": 0.0,
                "rtf": None,
                "segments_returned": 0,
                "language": self.config.language,
                "rms": 0.0,
                "peak": 0.0,
                "no_speech_prob": None,
                "avg_logprob": None,
            }
            self._record_decode(metadata)
            logger.warning("decode returned zero segments audio_seconds=0.00 rms=0.000000 peak=0.000000")
            return [], self.config.language
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio_seconds = len(audio) / SR
        rms = float(np.sqrt(np.mean(audio * audio))) if len(audio) else 0.0
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if 0 < peak < 0.7:
            audio = audio * min(0.85 / peak, 20.0)
        prompt_parts: list[str] = []
        context_limit = min(200, max(0, int(getattr(self.config, "context_chars", 200))))
        if getattr(self.config, "use_context_prompt", False) and context and context_limit:
            prompt_parts.append(context[-context_limit:])
        # initial_prompt is previous-text conditioning, not an instruction.
        initial_prompt = " ".join(part for part in prompt_parts if part).strip()
        if profile.backend != "faster-whisper":
            with bundle.lock:
                external_text, external_language = bundle.model.transcribe(audio, self.config.language, initial_prompt)
            language = external_language or self.config.language
            result = []
            if external_text:
                segment_id = f"seg-{int(offset * 1000):09d}-external"
                speaker, speaker_confidence = self.speakers.label(segment_id)
                result = [TranscriptSegment(
                    id=segment_id,
                    start=offset,
                    end=offset + audio_seconds,
                    text=external_text,
                    confidence=0.82,
                    speaker=speaker,
                    speaker_confidence=speaker_confidence,
                    language=language,
                )]
            self._finish_decode(
                final,
                audio_seconds,
                decode_started,
                result,
                language,
                rms,
                peak,
                None,
                None,
            )
            return result, language

        with bundle.lock:
            raw_segments, info = bundle.model.transcribe(
                audio,
                language=self.config.language,
                beam_size=max(
                    1,
                    (int(self.config.beam_size) if profile.id in {"large-v3", "large-v3-turbo", "custom"} else profile.beam_size)
                    if final else 2,
                ),
                temperature=list(profile.temperatures) if final else 0.0,
                initial_prompt=initial_prompt or None,
                hotwords=self.config.hotwords or None,
                condition_on_previous_text=False,
                without_timestamps=False,
                word_timestamps=final,
                vad_filter=False,
                no_speech_threshold=0.7,
                compression_ratio_threshold=2.8,
                log_prob_threshold=-1.5,
            )
            raw_segments = list(raw_segments)
        language = getattr(info, "language", None) or self.config.language
        raw_no_speech = [
            float(getattr(raw, "no_speech_prob"))
            for raw in raw_segments
            if getattr(raw, "no_speech_prob", None) is not None
        ]
        raw_avg_logprob = [
            float(getattr(raw, "avg_logprob"))
            for raw in raw_segments
            if getattr(raw, "avg_logprob", None) is not None
        ]
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
        merged = self._merge_fragments(result)
        self._finish_decode(
            final,
            audio_seconds,
            decode_started,
            merged,
            language,
            rms,
            peak,
            max(raw_no_speech) if raw_no_speech else None,
            min(raw_avg_logprob) if raw_avg_logprob else None,
        )
        return merged, language

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
