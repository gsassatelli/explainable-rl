class Agent:
    """Define the Agent super class which all particular agents (e.g Q-learner, SARSA).
    """
    __slots__ = ['env', 'gamma']

    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma

    def fit(self, n_episodes, n_steps, lr=0.1, lr_decay=0.05, lr_min=0.01,
            epsilon=0.1, epsilon_decay=0.05, epsilon_min=0.01, verbose=False):
        """Fit agent to the dataset.

        Args:
            n_episodes (int): number of episodes.
            n_steps (int): number of steps per episode.
            lr (float): learning rate.
            lr_decay (float): learning rate decay.
            lr_min (float): minimum learning rate.
            epsilon (float): epsilon-greedy policy parameter.
            epsilon_decay (float): epsilon decay.
            epsilon_min (float): minimum epsilon.
            verbose (bool): print training information.
        """
        raise NotImplementedError

class MDP:
    """Define the MDP super class which all particular MDP should inherit from.
    """
    __slots__ = ['dh']
    def __init__(self, dh):
        """Initialise the Strategic Pricing MDP class.
        Args:
            dh (DataHandler): Data handler object.
        """
        self.dh = dh

    def initialise_env(self):
        """Create the environment given the MDP information.
        """
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

