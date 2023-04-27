from explainable_RL.foundation.library import *


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

    def fit(self, agent_hyperparams, training_hyperparams, verbose=False, pbar=None):
        """Fit agent to the dataset.

        Args:
            agent_hyperparams (dict): Dictionary of agent hyperparameters.
            training_hyperparams (dict): Dictionary of training hyperparameters.
            verbose (bool): Print training information.
            pbar (tqdm): Progress bar.
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

    def predict_actions(self, states, epsilon=0):
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

    def predict_rewards(self, states, actions):
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

    def uncertainty_informed_policy(
        self, state=None, epsilon=0.1, use_uncertainty=False, q_importance=0.7
    ):
        """Get epsilon greedy policy that favours more densely populated state-action pairs.

        Args:
            state (list): Current state of the agent.
            epsilon (float): The exploration parameter.
            use_uncertainty (bool): Whether to use uncertainty informed policy.
            q_importance (float): The importance of the q value in the policy.

        Returns:
            action (int): selected action.
        """
        raise NotImplementedError

    @staticmethod
    def _convert_to_string(state):
        """Convert a state to a string.

        Args:
            state (list): The state to convert.

        Returns:
            state_str (string): The state as a string.
        """
        return ",".join(str(s) for s in state)

    def _init_q_table(self):
        """Initialize the q-table with zeros."""
        self.Q = sparse.DOK(self.env.bins)
        self.Q_num_samples = sparse.DOK(self.env.bins)
