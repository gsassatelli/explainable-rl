from library import *

# Import functions
from src.agents.td import TD


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
        """Create the Q-tables and state_to_action table.

        Args:
            verbose (bool): Defines whether print statements should be called.
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
        """Take a step in the environment and update the Q-tables.

        Args:
            epsilon (float): The epsilon value for the epsilon-greedy policy.
            lr (float): The learning rate.
        """
        action_a = self._epsilon_greedy_policy(state=self.state,
                                               epsilon=epsilon, Q=self.Q_a)
        action_b = self._epsilon_greedy_policy(state=self.state,
                                               epsilon=epsilon, Q=self.Q_b)
        if random.random() <= 0.5:
            state, next_state, reward, done = self.env.step(self.state,
                                                            action_a)
            self._update_q_values(state=state,
                                  action=action_a,
                                  next_state=next_state,
                                  reward=reward,
                                  lr=lr,
                                  epsilon=epsilon,
                                  Q_a=self.Q_a,
                                  Q_b=self.Q_b)

        else:
            state, next_state, reward, done = self.env.step(self.state,
                                                            action_b)
            self._update_q_values(state=state,
                                  action=action_a,
                                  next_state=next_state,
                                  reward=reward,
                                  lr=lr,
                                  epsilon=epsilon,
                                  Q_a=self.Q_b,
                                  Q_b=self.Q_a)
        self.Q = (self.Q_a + self.Q_b) / 2

        self.state = next_state

    def _update_q_values(self, state, action, next_state, reward, epsilon, lr, **kwargs):
        """Update the Q-tables.

        Args:
            state (tuple): The current state.
            action (int): The action taken.
            next_state (tuple): The next state.
            reward (float): The reward received.
            epsilon (float): The epsilon value for the epsilon-greedy policy.
            lr (float): The learning rate.
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
