# Import functions
from explainable_rl.agents.td import TD


class SarsaAgent(TD):
    """Sarsa agent."""

    def __init__(self, env, gamma, verbose=False):
        """Initialise the agent class.

        Args:
            env (MDP): MDP object.
            gamma (float): Discount factor.
            verbose (bool): Defines whether print statements should be called.
        """
        super().__init__(env, gamma, verbose)

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
        index_current = tuple(list(state) + [action])
        q_current = self.Q[index_current]
        next_action = self._epsilon_greedy_policy(next_state, epsilon=epsilon)
        index_next = tuple(list(next_state) + [next_action])
        q_next = self.Q[index_next]

        self.Q[index_current] = q_current + lr * (
            reward + self.gamma * q_next - q_current
        )

        self.Q_num_samples[index_current] += 1
