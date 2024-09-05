"""Test get forager
"""

import itertools
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
        print(f"\nAvailable forager classes: {forager_collection.available_agent_class}")
        foragers_looper = {
            "ForagerQLearning": dict(
                number_of_learning_rate=[1, 2],
                number_of_forget_rate=[0, 1],
                choice_kernel=["none", "one_step", "full"],
                action_selection=["softmax", "epsilon-greedy"],
            ),
            "ForagerLossCounting": dict(
                win_stay_lose_switch=[True, False],
                choice_kernel=["none", "one_step", "full"],
            ),
        }

        foragers = []
        n = 0
        for agent_class, looper in foragers_looper.items():
            specs_combinations = itertools.product(*looper.values())
            for specs in specs_combinations:
                n += 1
                agent_kwargs = dict(zip(looper.keys(), specs))
                forager = forager_collection.get_forager(
                    agent_class=agent_class,
                    agent_kwargs=agent_kwargs,
                )

                # Show its name if this exists in the preset foragers
                matched_preset = forager_collection.is_preset(
                    agent_class=agent_class,
                    agent_kwargs=agent_kwargs,
                )
                if matched_preset is not None:
                    preset_name = f"----->>>> {matched_preset} <<<<-----"
                else:
                    preset_name = ""

                print(
                    f"Model {n}: {agent_class} ["
                    f"{forager.get_params_str(if_latex=False, if_value=False)}]"
                    f" {preset_name}"
                )
                foragers.append(forager)


if __name__ == "__main__":
    unittest.main(verbosity=2)
