class Agent:
    """Parent of all child agents (e.g Q-learner, SARSA)."""

    def __init__(self, env, gamma, verbose=False):
        """Initialise the agent.

        Args:
            env (Environment): Environment object.
            gamma (float): Discount factor.
            verbose (bool): Print training information.
        """
        self.env = env
        self.gamma = gamma
        self.verbose = verbose

    # def fit(self, n_episodes, n_steps, use_uncertainty, lr=0.1, lr_decay=0.05, lr_min=0.01,
    #         epsilon=0.1, epsilon_decay=0.05, epsilon_min=0.01, verbose=False):
    def fit(self, agent_hyperparams, training_hyperparams, verbose=False):
        """Fit agent to the dataset.

        Args:
            n_episodes (int): Number of episodes.
            n_steps (int): Number of steps per episode.
            lr (float): Learning rate.
            lr_decay (float): Learning rate decay.
            lr_min (float): Minimum learning rate.
            epsilon (float): Epsilon-greedy policy parameter.
            epsilon_decay (float): Epsilon decay.
            epsilon_min (float): Minimum epsilon.
            verbose (bool): Print training information.
        """
        raise NotImplementedError

    def _epsilon_greedy_policy(self, state, epsilon):
        """Epsilon-greedy policy.

        Args:
            state (int): State.
            epsilon (float): Epsilon of epsilon-greedy policy.
                Defaults to 0 for pure exploitation.
        """
        raise NotImplementedError

    def predict_actions(self,
                        states,
                        epsilon=0):
        """Predict action for a list of states using epsilon-greedy policy.

        Args:
            states (list): States (binned).
            epsilon (float): Epsilon of epsilon-greedy policy.
                Defaults to 0 for pure exploitation.

        Returns:
            list: List of recommended actions.
        """
        actions = []
        for state in states:
            action = self._epsilon_greedy_policy(state, epsilon)
            actions.append([action])

        return actions

    def predict_rewards(self,
                        states,
                        actions):
        """Predict reward for a list of state-actions.

        This function uses the avg reward matrix (which simulates a real-life scenario).

        Args:
            states (list): States (binned).
            actions (list): Actions (binned).

        Returns:
            list: List of recommended actions.
        """

        rewards = []
        for state, action in zip(states, actions):
            _, _, reward, _ = self.env.step(state, action)
            rewards.append([reward[0]])

        return rewards
