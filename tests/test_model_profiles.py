import unittest

from notetaker.config import MODEL_PROFILES, MODEL_PROFILE_BY_ID


class ModelProfileTests(unittest.TestCase):
    def test_catalog_contains_real_hugging_face_cpu_choices(self):
        expected = [
            "large-v3",
            "large-v3-max",
            "qwen3-asr-1.7b",
            "voxtral-mini-3b",
            "qwen2-audio-7b",
            "distil-large-v3",
            "large-v3-turbo",
            "medium",
            "medium.en",
            "small",
            "small.en",
            "base",
            "base.en",
            "tiny",
            "tiny.en",
        ]
        self.assertEqual([profile.id for profile in MODEL_PROFILES], expected)
        self.assertEqual(len(MODEL_PROFILE_BY_ID), len(expected))
        self.assertEqual(MODEL_PROFILES[0].checkpoint, "large-v3")
        self.assertEqual(MODEL_PROFILES[0].repository, "https://huggingface.co/Systran/faster-whisper-large-v3")
        self.assertEqual(MODEL_PROFILES[1].checkpoint, "large-v3")
        self.assertGreater(MODEL_PROFILES[1].beam_size, MODEL_PROFILES[0].beam_size)

    def test_larger_profiles_have_expected_sizes_and_backends(self):
        expected = {
            "qwen3-asr-1.7b": ("1.7B", "transformers-qwen3-asr", "Qwen/Qwen3-ASR-1.7B-hf"),
            "voxtral-mini-3b": ("3B", "transformers-voxtral", "mistralai/Voxtral-Mini-3B-2507"),
            "qwen2-audio-7b": ("7B", "transformers-qwen2-audio", "Qwen/Qwen2-Audio-7B-Instruct"),
        }
        for profile_id, (parameters, backend, checkpoint) in expected.items():
            profile = MODEL_PROFILE_BY_ID[profile_id]
            self.assertEqual(profile.parameters, parameters)
            self.assertEqual(profile.backend, backend)
            self.assertEqual(profile.checkpoint, checkpoint)
            self.assertEqual(profile.compute_type, "float32")
            self.assertEqual(profile.optional_requirements, "requirements-large-models.txt")

    def test_every_profile_is_cpu_only_and_has_a_public_repository(self):
        for profile in MODEL_PROFILES:
            details = profile.to_dict()
            self.assertEqual(details["device"], "cpu")
            self.assertIn(details["compute_type"], {"int8", "float32"})
            self.assertTrue(details["repository"].startswith("https://huggingface.co/"))
            self.assertTrue(details["description"])
            self.assertTrue(details["label"])


if __name__ == "__main__":
    unittest.main()
