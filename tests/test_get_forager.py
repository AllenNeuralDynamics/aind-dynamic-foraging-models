"""Test get forager
"""

import unittest

from aind_dynamic_foraging_models.generative_model import ForagerCollection


class TestGetForager(unittest.TestCase):
    """Test get forager.

    See other test frunctions for how to simulate a task and perform model fitting.
    See also https://foraging-behavior-browser.allenneuraldynamics-test.org/RL_model_playground
    """

    def test_get_forager(self):
        """Test get forager"""
        forager_collection = ForagerCollection()

        print(f"\n\nPreset foragers: {forager_collection.presets}")

        # --- Get preset models ---
        for preset in forager_collection.presets:
            forager = forager_collection.get_preset_forager(preset)
            print(f"Default params for {preset}: {forager.get_params_str(if_latex=False)}")
            self.assertIsNotNone(forager)

        # --- Get all available foragers for systematic model comparison etc. ---
        print(f"\nAvailable forager classes: {forager_collection.FORAGER_CLASSES}")
        forager_df = forager_collection.get_all_foragers()
        print(forager_df)
        self.assertEqual(len(forager_df), 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
