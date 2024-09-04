"""Presets of forager models and utility functions to create group of agents.

"""

from aind_dynamic_foraging_models import generative_model


class ForagerCollection:
    """A class to create foragers."""

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
                choice_kernel="none",
            ),
        ),
    }

    def __init__(self):
        self.presets = list(self.FORAGER_PRESETS.keys())
        self.available_foragers = [f for f in dir(generative_model) 
                                   if f.startswith("Forager")
                                   and f != "ForagerCollection"]
        
    def get_forager(self, agent_class, agent_kwargs={}, **kwargs):
        """Get a forager.

        Parameters
        ----------
        agent_class : str
            The class name of the forager.
        agent_kwargs : dict
            The keyword arguments to pass to the forager.
        **kwargs : dict
            Other keyword arguments to pass to the forager (like the rng seed).
        """
        agent_class = getattr(generative_model, agent_class, None)
        if agent_class is None:
            raise ValueError(f"{agent_class} is not found in the generative_model. "
                             f"Available agents are: {self.available_foragers}")
        return agent_class(**agent_kwargs, **kwargs)
    

    def get_preset_forager(self, alias, **kwargs):
        """Get a preset forager.

        Parameters
        ----------
        alias : str
            The alias of the forager.
        **kwargs : dict
            Other keyword arguments to pass to the forager (like the rng seed).
        """
        assert alias in self.FORAGER_PRESETS.keys(), \
            f"{alias} is not found in the preset foragers."\
            f" Available presets are: {self.presets}"

        agent = self.FORAGER_PRESETS[alias]
        return self.get_forager(agent["agent_class"], agent["agent_kwargs"], **kwargs)


if __name__ == "__main__":
    foragers = ForagerCollection()
    forager = foragers.get_preset_forager("Bari2019")
    print(foragers.presets)
    print(forager.params)

