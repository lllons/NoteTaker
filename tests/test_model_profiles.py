import unittest

from notetaker.config import MODEL_PROFILES, MODEL_PROFILE_BY_ID


class ModelProfileTests(unittest.TestCase):
    def test_web_selector_exposes_five_ordered_profiles(self):
        self.assertEqual([profile.id for profile in MODEL_PROFILES], [
            "model-1",
            "model-2",
            "model-3",
            "model-4",
            "model-5",
        ])
        self.assertEqual(len(MODEL_PROFILE_BY_ID), 5)
        self.assertEqual(MODEL_PROFILES[0].checkpoint, "large-v3")
        self.assertEqual(MODEL_PROFILES[0].beam_size, 8)
        self.assertGreater(MODEL_PROFILES[-1].beam_size, MODEL_PROFILES[0].beam_size)
        self.assertGreater(len(MODEL_PROFILES[-1].temperatures), len(MODEL_PROFILES[0].temperatures))

    def test_every_profile_advertises_cpu_int8_execution(self):
        for profile in MODEL_PROFILES:
            details = profile.to_dict()
            self.assertEqual(details["device"], "cpu")
            self.assertEqual(details["compute_type"], "int8")
            self.assertTrue(details["description"])
            self.assertTrue(details["label"].startswith("Model "))


if __name__ == "__main__":
    unittest.main()
