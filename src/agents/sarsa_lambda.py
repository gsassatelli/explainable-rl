from library import *

# Import functions
from src.agents.td import TD


class SarsaLambdaAgent(TD):
    """Sarsa Lambda agent."""

    __slots__ = ['e', 'lambda_']

    def __init__(self, env, gamma, verbose=False, lambda_=0.9):
        """Initialise the agent class.

        Args:
            env (MDP): MDP object.
            gamma (float): Discount factor.
        """
        super().__init__(env=env, gamma=gamma, verbose=verbose)
        self.e = sparse.DOK(self.env.bins)
        self.lambda_ = lambda_

    def _update_q_values(self, state, action, next_state, reward, epsilon, lr, **kwargs):
        """Update the Q table using the Bellman equation and SARSA update.

        Args:
            state (list): Current state of the agent.
            action (int): Selected action.
            next_state (list): Next state of the agent.
            reward (float): Reward for the selected action.
            lr (float): Learning rate.
            **kwargs (dict): The keyword arguments.
        """
        index_current = tuple(list(state) + [action])
        q_current = self.Q[index_current]
        next_action = self._epsilon_greedy_policy(next_state, epsilon=epsilon)
        index_next = tuple(list(next_state) + [next_action])
        q_next = self.Q[index_next]

        delta = reward + self.gamma * q_next - q_current

        self.Q_num_samples[index_current] += 1

        self.e[index_current] += 1

        indices = self.e.data.keys()

        for index in indices:
            self.Q[index] += lr * delta * self.e[index]
            self.e[index] *= self.gamma * self.lambda_





