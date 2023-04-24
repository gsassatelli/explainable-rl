# Import functions
from src.agents.td import TD


class SarsaAgent(TD):
    """Sarsa agent."""

    def __init__(self, env, gamma, verbose=False):
        super().__init__(env, gamma, verbose)

    def _update_q_values(self, state, action, next_state, reward, epsilon, lr, **kwargs):
        index_current = tuple(list(state) + [action])
        q_current = self.Q[index_current]
        next_action = self._epsilon_greedy_policy(next_state, epsilon=epsilon)
        index_next = tuple(list(next_state) + [next_action])
        q_next = self.Q[index_next]

        self.Q[index_current] = \
            q_current + lr * (reward + self.gamma * q_next - q_current)

        self.Q_num_samples[index_current] += 1
