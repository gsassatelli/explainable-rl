from library import *

# Import functions
from src.agents.td import TD


class QLearningAgent(TD):
    """Q-Learning agent."""

    def __init__(self, env, gamma, verbose=False):
        super().__init__(env, gamma, verbose)

    def _update_q_values(self, state, action, next_state, reward, epsilon, lr, **kwargs):
        index_current = tuple(list(state) + [action])
        q_current = self.Q[index_current]
        index_next = tuple(next_state)
        q_next = np.max(self.Q[index_next].todense())
        self.Q[index_current] = \
            q_current + lr * (reward + self.gamma * q_next - q_current)
        self.Q_num_samples[index_current] += 1
