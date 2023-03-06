class Agent:
    __slots__ = ['env', 'gamma']

    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma

class MDP:
    """Defines the MDP super class which all particular MDP should inherit from
    """
    __slots__ = ['dh']
    def __init__(self, dh):
        """Initialises the Strategic Pricing MDP class.
        Args:
            dh (DataHandler): Data handler object.
        """
        self.dh = dh

    def initialise_env(self):
        """Creates the environment given the MDP information.
        """
        raise NotImplementedError

    def reset(self):
        """Resets environment.
        Returns:
            list: Randomised initial state.
        """
        raise NotImplementedError

    def step(self, state, action):
        """Takes a step in the environment.
        A True done flag indicates that the environment terminated.
        Args:
            state (list): Current state values of agent.
            action (int): Action for agent to take.

        Returns:
            tuple: current state, action, next state, done flag.
        """
        raise NotImplementedError

