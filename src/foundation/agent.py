class Agent:
    """Define the Agent super class which all particular agents (e.g Q-learner, SARSA).
    """
    __slots__ = ['env', 'gamma', 'verbose']

    def __init__(self, env, gamma, verbose=False):
        self.env = env
        self.gamma = gamma
        self.verbose = verbose

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
