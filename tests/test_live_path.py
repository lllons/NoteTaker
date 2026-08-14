import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import unittest
import wave

try:
    import numpy as np
    from listening.audio import FRAME, SR, Segmenter, resample_audio
    AUDIO_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency-free source checkout
    np = None  # type: ignore[assignment]
    AUDIO_AVAILABLE = False

try:
    from notetaker.transcription import ModelBundle, TranscriptionRuntime
    TRANSCRIPTION_AVAILABLE = AUDIO_AVAILABLE
except ImportError:  # pragma: no cover - faster-whisper is optional for offline tests
    TRANSCRIPTION_AVAILABLE = False

try:
    from notetaker.app import DecodeScheduler, empty_decode_event
    APP_AVAILABLE = AUDIO_AVAILABLE
except ImportError:  # pragma: no cover - FastAPI is optional until the app is run
    APP_AVAILABLE = False


class FakeVad:
    backend = "test"
    error = None

    def __call__(self, frame):
        return 0.9 if float(np.sqrt(np.mean(frame * frame))) > 0.05 else 0.01


@unittest.skipUnless(AUDIO_AVAILABLE, "numpy is required for audio regressions")
class AudioRegressionTests(unittest.TestCase):
    def test_pauses_close_segments_before_the_old_thirty_second_cap(self):
        config = SimpleNamespace(
            preroll_seconds=0.1,
            end_silence_seconds=1.4,
            max_segment_seconds=14,
            soft_max_seconds=12,
            min_segment_seconds=0.15,
            partial_seconds=60,
            vad_on_threshold=0.35,
            vad_off_threshold=0.2,
        )
        segmenter = Segmenter(config)
        segmenter.vad = FakeVad()
        speech = np.full(FRAME, 0.12, dtype=np.float32)
        silence = np.zeros(FRAME, dtype=np.float32)
        events = []
        for _ in range(9):
            events.extend(segmenter.feed(np.tile(speech, 8)))
            events.extend(segmenter.feed(np.tile(silence, 48)))
        events.extend(segmenter.flush())
        finals = [audio for kind, audio, _sid in events if kind == "final"]
        self.assertGreaterEqual(len(finals), 8)
        self.assertLess(max(len(audio) / SR for audio in finals), 14.5)
        self.assertTrue(all(len(audio) / SR < 20 for audio in finals))

    def test_continuous_speech_uses_soft_splits(self):
        config = SimpleNamespace(
            preroll_seconds=0.1,
            end_silence_seconds=1.4,
            max_segment_seconds=14,
            soft_max_seconds=8,
            min_segment_seconds=0.15,
            partial_seconds=60,
            vad_on_threshold=0.35,
            vad_off_threshold=0.2,
        )
        segmenter = Segmenter(config)
        segmenter.vad = FakeVad()
        events = segmenter.feed(np.full(SR * 20, 0.12, dtype=np.float32))
        events.extend(segmenter.flush())
        finals = [len(audio) / SR for kind, audio, _sid in events if kind == "final"]
        self.assertGreaterEqual(len(finals), 2)
        self.assertLess(max(finals), 9.5)

    def test_resampling_removes_audible_twelve_khz_alias(self):
        source_rate = 48_000
        source = np.arange(source_rate, dtype=np.float32) / source_rate
        input_audio = np.sin(2 * np.pi * 12_000 * source).astype(np.float32)
        output = resample_audio(input_audio, source_rate, SR)
        spectrum = np.abs(np.fft.rfft(output * np.hanning(len(output))))
        alias_bin = int(round(4_000 * len(output) / SR))
        alias_amplitude = float(spectrum[alias_bin] / max(1, len(output)))
        self.assertLess(alias_amplitude, 0.02)


@unittest.skipUnless(TRANSCRIPTION_AVAILABLE, "numpy and faster-whisper dependencies are required")
class PromptRegressionTests(unittest.TestCase):
    def test_prompt_free_decode_on_fixture_wav_is_non_empty_without_prompt_echo(self):
        class Raw:
            text = "vector database"
            start = 0.0
            end = 1.0
            no_speech_prob = 0.01
            avg_logprob = -0.1
            compression_ratio = 1.0
            words = []

        class Info:
            language = "en"

        class FakeModel:
            def __init__(self):
                self.calls = []

            def transcribe(self, audio, **kwargs):
                self.calls.append(kwargs)
                return iter([Raw()]), Info()

        config = SimpleNamespace(
            model="small.en",
            draft_model="tiny.en",
            language="en",
            beam_size=2,
            threads=0,
            hotwords="pgvector, CTranslate2",
            context_chars=200,
            use_context_prompt=False,
            diarization="labels-only",
            benchmark_seconds=1,
        )
        fixture = io.BytesIO()
        fixture_audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(SR) / SR)).astype(np.float32)
        pcm = np.clip(fixture_audio * 32767, -32768, 32767).astype(np.int16)
        with wave.open(fixture, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SR)
            handle.writeframes(pcm.tobytes())
        fixture.seek(0)
        with wave.open(fixture, "rb") as handle:
            audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32) / 32767.0

        runtime = TranscriptionRuntime(config)
        final_model = FakeModel()
        draft_model = FakeModel()
        runtime.final = ModelBundle(final_model, threading.Lock(), ThreadPoolExecutor(max_workers=1))
        runtime.draft = ModelBundle(draft_model, threading.Lock(), ThreadPoolExecutor(max_workers=1))
        runtime._loaded_checkpoint = "small.en"
        runtime._loaded_draft_checkpoint = "tiny.en"
        runtime._model_state = "ready"
        runtime._benchmark = {"state": "ready"}
        try:
            partial, _ = runtime.transcribe(audio, False, "stale instruction text")
            final, _ = runtime.transcribe(audio, True, "stale instruction text")
            self.assertTrue(partial)
            self.assertTrue(final)
            self.assertEqual(partial[0].text, final[0].text)
            self.assertEqual(draft_model.calls[0]["initial_prompt"], None)
            self.assertEqual(final_model.calls[0]["initial_prompt"], None)
            self.assertEqual(draft_model.calls[0]["hotwords"], config.hotwords)
            self.assertEqual(final_model.calls[0]["hotwords"], config.hotwords)
            self.assertNotIn("Transcribe verbatim", str(draft_model.calls[0]))
            self.assertNotIn("Transcribe verbatim", str(final_model.calls[0]))
            self.assertIsNot(runtime.final, runtime.draft)
        finally:
            runtime.final.executor.shutdown(wait=False)
            runtime.draft.executor.shutdown(wait=False)


@unittest.skipUnless(APP_AVAILABLE, "numpy and FastAPI dependencies are required")
class LiveProtocolRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_partials_are_dropped_but_finals_are_buffered(self):
        gate = asyncio.Event()
        started = []

        async def process(kind, _audio, _sid, _offset):
            started.append(kind)
            await gate.wait()

        scheduler = DecodeScheduler(process, max_inflight=1, max_pending_finals=2)
        await scheduler.start()
        try:
            self.assertTrue(await scheduler.submit("partial", np.zeros(10), 1, 0.0))
            await asyncio.sleep(0)
            self.assertTrue(await scheduler.submit("final", np.zeros(10), 2, 0.0))
            self.assertFalse(await scheduler.submit("partial", np.zeros(10), 3, 0.0))
            self.assertLessEqual(scheduler.active, 1)
            gate.set()
            await scheduler.drain()
            self.assertIn("final", started)
        finally:
            await scheduler.shutdown()

    def test_empty_decode_event_is_explicit(self):
        event = empty_decode_event(
            "final",
            2.5,
            {"rms": 0.001, "peak": 0.01, "decode_seconds": 3.0, "rtf": 1.2},
        )
        self.assertEqual(event["t"], "empty")
        self.assertEqual(event["scope"], "final")
        self.assertEqual(event["reason"], "decoder returned no segments")
        self.assertEqual(event["audio_seconds"], 2.5)
        self.assertEqual(event["rtf"], 1.2)

    def test_pure_silence_has_an_explicit_no_speech_event(self):
        segmenter = Segmenter(SimpleNamespace())
        segmenter.vad = FakeVad()
        silence = np.zeros(SR, dtype=np.float32)
        self.assertEqual(segmenter.feed(silence), [])
        self.assertEqual(segmenter.flush(), [])
        event = empty_decode_event("flush", 1.0, reason="no speech segment was captured")
        self.assertEqual(event["t"], "empty")
        self.assertEqual(event["reason"], "no speech segment was captured")


if __name__ == "__main__":
    unittest.main()
