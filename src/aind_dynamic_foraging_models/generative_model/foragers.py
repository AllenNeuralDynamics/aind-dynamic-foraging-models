"""Presets of forager models and utility functions to create group of agents.

"""

import inspect
import itertools
from typing import get_type_hints, get_origin, Literal

import pandas as pd

from aind_dynamic_foraging_models import generative_model


class ForagerCollection:
    """A class to create foragers."""

    FORAGER_CLASSES = [
        "ForagerQLearning",
        "ForagerLossCounting",
    ]

    FORAGER_PRESETS = {
        "Bari2019": dict(
            description="The vanilla Bari2019 model",
            agent_class="ForagerQLearning",
            agent_kwargs=dict(
                number_of_learning_rate=1,
                number_of_forget_rate=1,
                choice_kernel="one_step",
                action_selection="softmax",
            ),
        ),
        "Hattori2019": dict(
            description="The vanilla Hattori2019 model",
            agent_class="ForagerQLearning",
            agent_kwargs=dict(
                number_of_learning_rate=2,
                number_of_forget_rate=1,
                choice_kernel="none",
                action_selection="softmax",
            ),
        ),
        "Rescorla-Wagner": dict(
            description="The vanilla Rescorla-Wagner model disccused in the Sutton & Barto book",
            agent_class="ForagerQLearning",
            agent_kwargs=dict(
                number_of_learning_rate=1,
                number_of_forget_rate=0,
                choice_kernel="none",
                action_selection="epsilon-greedy",
            ),
        ),
        "Win-Stay-Lose-Shift": dict(
            description="The vanilla Win-stay-lose-shift model",
            agent_class="ForagerLossCounting",
            agent_kwargs=dict(
                win_stay_lose_switch=True,
                choice_kernel="none",
            ),
        ),
    }

    def __init__(self):
        """Init"""
        self.presets = list(self.FORAGER_PRESETS.keys())

    def get_agent_class(self, agent_class_name):
        """Get an agent class by agent_class_name"""
        agent_class = getattr(generative_model, agent_class_name, None)
        if agent_class is None:
            raise ValueError(
                f"{agent_class} is not found in the generative_model. "
                f"Available agents are: {self.available_agent_class}"
            )
        return agent_class

    def get_forager(self, agent_class_name, agent_kwargs={}, **kwargs):
        """Get a forager by agent_class_name and agent_kwargs

        Parameters
        ----------
        agent_class_name : str
            The class name of the forager.
        agent_kwargs : dict
            The keyword arguments to pass to the forager.
        **kwargs : dict
            Other keyword arguments to pass to the forager (like the rng seed).
        """
        agent_class = self.get_agent_class(agent_class_name)
        return agent_class(**agent_kwargs, **kwargs)

    def get_preset_forager(self, alias, **kwargs):
        """Get a preset forager but its alias.

        Parameters
        ----------
        alias : str
            The alias of the forager.
        **kwargs : dict
            Other keyword arguments to pass to the forager (like the rng seed).
        """
        assert alias in self.FORAGER_PRESETS.keys(), (
            f"{alias} is not found in the preset foragers."
            f" Available presets are: {self.presets}"
        )

        agent = self.FORAGER_PRESETS[alias]
        return self.get_forager(agent["agent_class"], agent["agent_kwargs"], **kwargs)

    def is_preset(self, agent_class, agent_kwargs):
        """Check if an given agent is a preset forager.

        Parameters
        ----------
        agent_class : str
            The class name of the forager to query
        agent_kwargs : dict
            The keyword arguments of the forager to query

        Returns
        -------
        str or None
            The alias of the preset forager if it exists, otherwise None
        """
        for preset_name, preset_specs in self.FORAGER_PRESETS.items():
            if (
                preset_specs["agent_class"] == agent_class
                and preset_specs["agent_kwargs"] == agent_kwargs
            ):
                return preset_name
        else:
            return None

    def get_all_foragers(self, **kwargs) -> pd.DataFrame:
        """Return all available foragers in a dataframe.

        Parameters
        ----------
        **kwargs : dict
            Other keyword arguments to pass to the forager (like the rng seed).
        """
        all_foragers = []

        # Loop over all agent classes
        for agent_class_name in self.FORAGER_CLASSES:
            agent_class = self.get_agent_class(agent_class_name)
            agent_kwargs_options = {
                key: value["options"]
                for key, value in self._get_agent_kwargs_options(agent_class).items()
            }
            agent_kwargs_combinations = itertools.product(*agent_kwargs_options.values())

            # Loop over all agent_kwargs_combinations
            for specs in agent_kwargs_combinations:
                agent_kwargs = dict(zip(agent_kwargs_options.keys(), specs))
                forager = self.get_forager(agent_class_name, agent_kwargs, **kwargs)
                preset_name = self.is_preset(agent_class_name, agent_kwargs)

                all_foragers.append(
                    dict(
                        agent_class_name=agent_class_name,
                        agent_kwargs=agent_kwargs,
                        agent_alias=forager.get_agent_alias(),
                        **agent_kwargs,  # Also unpack agent_kwargs
                        preset_name=preset_name,
                        n_free_params=len(forager.params_list_free),
                        params=forager.get_params_str(if_latex=True, if_value=False),
                        forager=forager,
                    )
                )

        return pd.DataFrame(all_foragers)

    @staticmethod
    def _get_agent_kwargs_options(agent_class) -> dict:
        """Given an agent class, return the agent's agent_kwargs that are of type Literal.

        This requires the agent class to have type hints as Literal in the __init__ method.

        Example:
        >>> _get_agent_kwargs_options("ForagerQLearning")
        {'number_of_learning_rate': {'options': (1, 2), 'default': 2},
         'number_of_forget_rate': {'options': (0, 1), 'default': 1},
         'choice_kernel': {'options': ('none', 'one_step', 'full'), 'default': 'none'},
         'action_selection': {'options': ('softmax', 'epsilon-greedy'), 'default': 'softmax'}}
        """
        type_hints = get_type_hints(agent_class.__init__)
        signature = inspect.signature(agent_class.__init__)

        agent_args_options = {}
        for arg, type_hint in type_hints.items():
            if get_origin(type_hint) is Literal:  # Check if the type hint is a Literal
                # Get options
                literal_values = type_hint.__args__
                default_value = signature.parameters[arg].default

                agent_args_options[arg] = {"options": literal_values, "default": default_value}
        return agent_args_options


if __name__ == "__main__":
    forager_collection = ForagerCollection()
    forager = forager_collection.get_preset_forager("Bari2019")
    print(forager_collection.presets)
    print(forager.params)

    df = forager_collection.get_all_foragers()
    print(df.agent_alias)
    forager = df.iloc[0].forager
    forager.params
