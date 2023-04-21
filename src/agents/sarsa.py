# Import functions
from src.agents.td import TD


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

    def _update_q_values(self, state, action, next_state, reward, epsilon, lr, **kwargs):
        """Update the Q table using the Bellman equation and SARSA update.

        Args:
            state (list): current state of the agent.
            action (int): selected action.
            next_state (list): next state of the agent.
            reward (float): reward for the selected action.
            lr (float): learning rate.
        """
        index_current = tuple(list(state) + [action])
        q_current = self.Q[index_current]
        next_action = self._epsilon_greedy_policy(next_state, epsilon=epsilon)
        index_next = tuple(list(next_state) + [next_action])
        q_next = self.Q[index_next]

        self.Q[index_current] = \
            q_current + lr * (reward + self.gamma * q_next - q_current)

        self.Q_num_samples[index_current] += 1
