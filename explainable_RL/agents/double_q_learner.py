from explainable_RL.foundation.library import *

# Import functions
from explainable_RL.agents.td import TD


class DoubleQLearner(TD):
    """Double Q-Learner agent."""

    def __init__(self, env, gamma, verbose=False):
        """Initialise the agent class.

        Args:
            env (MDP): MDP object.
            gamma (float): Discount factor.
            verbose (bool): Defines whether print statements should be called.
        """
        super().__init__(env=env, gamma=gamma, verbose=verbose)
        self.Q = None
        self.Q_a = None
        self.Q_b = None
        self.state_to_action = None
        self.Q_num_samples = None
        self.state = None

    def create_tables(self, verbose=False):
        """Initialize the agent.

        This resets the environment, creates the q-table and the state to
        action mapping.

        Args:
            verbose (bool): Print information.
        """
        self.env.reset()
        if verbose:
            print("Create q-table")
        # create q-table
        self._init_q_table()
        self.Q_a = copy.deepcopy(self.Q)
        self.Q_b = copy.deepcopy(self.Q)
        self.state_to_action = self.env.state_to_action

    def _step(self, epsilon, lr, use_uncertainty=False):
        """Perform a step in the environment.

        Args:
            epsilon (float): Epsilon-greedy policy parameter.
            lr (float): Learning rate.
            use_uncertainty (bool): Whether to use uncertainty informed policy.

        Returns:
            bool: Defines whether the episode is finished.
        """
        action_a = self._epsilon_greedy_policy(
            state=self.state, epsilon=epsilon, Q=self.Q_a
        )
        action_b = self._epsilon_greedy_policy(
            state=self.state, epsilon=epsilon, Q=self.Q_b
        )
        if random.random() <= 0.5:
            state, next_state, reward, done = self.env.step(self.state, action_a)
            self._update_q_values(
                state=state,
                action=action_a,
                next_state=next_state,
                reward=reward,
                lr=lr,
                epsilon=epsilon,
                Q_a=self.Q_a,
                Q_b=self.Q_b,
            )

        else:
            state, next_state, reward, done = self.env.step(self.state, action_b)
            self._update_q_values(
                state=state,
                action=action_a,
                next_state=next_state,
                reward=reward,
                lr=lr,
                epsilon=epsilon,
                Q_a=self.Q_b,
                Q_b=self.Q_a,
            )
        self.Q = (self.Q_a + self.Q_b) / 2

        self.state = next_state

    def _update_q_values(
        self, state, action, next_state, reward, epsilon, lr, **kwargs
    ):
        """Update the Q table.

        Args:
            state (list): Current state of the agent.
            action (int): Selected action.
            next_state (list): Next state of the agent.
            reward (float): Reward for the selected action.
            epsilon (float): The exploration parameter.
            lr (float): Learning rate.
            **kwargs (dict): The keyword arguments.
        """
        Q_a = kwargs["Q_a"]
        Q_b = kwargs["Q_b"]

        a_star = np.argmax(Q_a[tuple(next_state)].todense())
        index_current = tuple(list(state) + [action])
        q_current = Q_a[index_current]
        next_index = tuple(list(next_state) + [a_star])
        q_next = Q_b[next_index]

        Q_a[index_current] += lr * (reward + self.gamma * q_next - q_current)
        self.Q_num_samples[index_current] += 1
