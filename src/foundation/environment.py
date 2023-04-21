from library import *

class MDP:
    """Define the MDP super class which all particular MDP should inherit from.
    """

    def __init__(self, dh):
        """Initialise the Strategic Pricing MDP class.

        Args:
            dh (DataHandler): Data handler object.
        """
        self.dh = dh

    def initialise_env(self):
        """Create the environment given the MDP information."""
        raise NotImplementedError

    def reset(self):
        """Reset environment.

        Returns:
            list: Randomised initial state.
        """
        raise NotImplementedError

    def step(self, state, action):
        """Take a step in the environment.

        A True done flag indicates that the environment terminated.

        Args:
            state (list): Current state values of agent.
            action (int): Action for agent to take.

        Returns:
            tuple: current state, action, next state, done flag.
        """
        raise NotImplementedError