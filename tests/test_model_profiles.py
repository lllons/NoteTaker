import unittest

from notetaker.config import MODEL_PROFILES, MODEL_PROFILE_BY_ID


class ModelProfileTests(unittest.TestCase):
    def test_catalog_contains_real_hugging_face_cpu_choices(self):
        expected = [
            "large-v3",
            "large-v3-max",
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

    def test_every_profile_is_cpu_only_and_has_a_public_repository(self):
        for profile in MODEL_PROFILES:
            details = profile.to_dict()
            self.assertEqual(details["device"], "cpu")
            self.assertEqual(details["compute_type"], "int8")
            self.assertTrue(details["repository"].startswith("https://huggingface.co/"))
            self.assertTrue(details["description"])
            self.assertTrue(details["label"])


if __name__ == "__main__":
    unittest.main()
