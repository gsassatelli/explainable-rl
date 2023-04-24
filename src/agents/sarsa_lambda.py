from library import *

# Import functions
from src.agents.td import TD


class SarsaLambdaAgent(TD):
    """Sarsa Lambda agent."""

    def __init__(self, env, gamma, verbose=False, lambda_=0.9):
        super().__init__(env=env, gamma=gamma, verbose=verbose)
        self.e = sparse.DOK(self.env.bins)
        self.lambda_ = lambda_

    def _update_q_values(self, state, action, next_state, reward, epsilon, lr, **kwargs):
        index_current = tuple(list(state) + [action])
        q_current = self.Q[index_current]
        next_action = self._epsilon_greedy_policy(next_state, epsilon=epsilon)
        index_next = tuple(list(next_state) + [next_action])
        q_next = self.Q[index_next]

        delta = reward + self.gamma * q_next - q_current

        self.Q_num_samples[index_current] += 1

        self.e[index_current] += 1

        for index in list(self.e.data):
            self.Q[index] += lr * delta * self.e[index]
            self.e[index] *= self.gamma * self.lambda_
