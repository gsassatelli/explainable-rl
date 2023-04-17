import numpy as np
import sparse
import random
from datetime import datetime
from src.foundation.utils import *
from src.agents.td import TD


class QLearningAgent(TD):
    """Agent class to store and update q-table.
    """

    def __init__(self, env, gamma):
        """Initialise the agent class.

        Args:
            env (MDP): MDP object.
            gamma (float): Discount factor.
        """

        super().__init__(env, gamma)

    def _update_q_values(self, state, action, next_state, reward, epsilon, lr):
        """Update the Q table using the Bellman equation and q_learning update.

        Args:
            state (list): current state of the agent.
            action (int): selected action.
            next_state (list): next state of the agent.
            reward (float): reward for the selected action.
            epsilon (float): the exploration parameter
            lr (float): learning rate.
        """
        index_current = tuple(list(state) + [action])
        q_current = self.Q[index_current]
        index_next = tuple(next_state)
        q_next = np.max(self.Q[index_next].todense())

        self.Q[index_current] = \
            q_current + lr * (reward + self.gamma * q_next - q_current)

        self.Q_num_samples[index_current] += 1
