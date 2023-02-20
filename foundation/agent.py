# Import packages
import numpy as np
from foundation.utils import *
import sparse
import random
from datetime import datetime



class Agent:
    """Agent class to store and update q-table.
    """

    def __init__(self, env, gamma=0.9):
        """Initialise the agent class.
        
        Args:
            env (MDP): MDP object.
            gamma (float): Discount factor.
        """
        self.env = env
        self.Q = None
        self.state_to_action = None
        self.gamma = gamma
        self.state = None
        self._initialize_agent()

    def fit(self, n_episodes, n_steps, lr=0.1, lr_decay=0.05, lr_min=0.01,
            epsilon=0.1, epsilon_decay=0.05, epsilon_min=0.01, verbose=False):

        """Fit agent to the dataset.

        Args:
            n_episodes (int): number of episodes.
            n_steps (int): number of steps per episode.
            lr (float): learning rate.
            lr_decay (float): learning rate decay.
            lr_min (float): minimum learning rate.
            epsilon (float): epsilon-greedy policy parameter.
            epsilon_decay (float): epsilon decay.
            epsilon_min (float): minimum epsilon.
            verbose (bool): print training information.
        """
        if verbose:
            print("Apply q-learning and update q-table")

        for _ in range(n_episodes):
            self.state = self.env.reset()
            for i in range(n_steps):
                done = self._step(epsilon=epsilon, lr=lr)
                if done:
                    break
            lr = decay_param(lr, lr_decay, lr_min)
            epsilon = decay_param(epsilon, epsilon_decay, epsilon_min)

        if verbose:
            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"{timestamp}: Finished training :) ! \n"
                  f"Example Q-table for state "
                  f"{[1, 9, 0]}: {self.Q[1, 9, 0].todense()}\n"
                  f"Example Q-table for state "
                  f"{[1, 0, 0]}: {self.Q[1, 0, 0].todense()}")

    def _initialize_agent(self,
                         verbose=False):
        """Initialize the agent.

        This resets the environment, creates the q-table and the state to
        action mapping.

        Args:
            verbose (bool): print information.
        """
        self.env.reset()
        if verbose:
            print("Create q-table")
        # create q-table
        self._init_q_table()
        self.state_to_action = self.env.state_to_action

    def _epsilon_greedy_policy(self,
                               state=None,
                               epsilon=0.1):
        """Get the epsilon greedy action.

        Args:
            state (list): current state of the agent.
            state_str (string): the state as a string.
            epsilon (float): the exploration parameter.

        Returns:
            action (int): selected action.
        """
        if state is None:
            state = self.state

        state_str = convert_to_string(state)

        q_values = self.Q[state[0], state[1], state[2], :].todense()
        if random.random() > epsilon:
            action = np.argmax(q_values)
        else:
            action = random.choice(list(self.state_to_action[state_str]))
        return action

    def _init_q_table(self):
        """Initialize the q-table.
        """
        coords = []
        for state_str, actions in self.env.state_to_action.items():
            state = convert_to_list(state_str)
            actions = list(actions)
            for action in actions:
                coords.append(state + [action])
        self._create_dok_q_table(coords)

    def _create_dok_q_table(self,
                            coords):
        """Create the q-table in DOK format.

        Args:
            coords (list): list of coordinates.
        """
        q_values = np.zeros(len(coords))
        coords = np.array(coords).T.tolist()

        # create COO (read only) matrix
        self.Q = sparse.COO(coords, q_values)

        # convert to DOK
        self.Q = sparse.DOK.from_numpy(self.Q.todense())

    def _step(self, epsilon, lr):
        """Perform a step in the environment.

        Args:
            epsilon (float): epsilon-greedy policy parameter.
            lr (float): learning rate.

        Returns:
            done: boolean indicating whether the episode is finished.
        """
        action = self._epsilon_greedy_policy(self.state,
                                             epsilon=epsilon)
        state, next_state, reward, done = self.env.step(self.state,
                                                        action)
        self._update_q_values(state, action, next_state, reward, lr)
        self.state = next_state
        return done

    def _update_q_values(self, state,
                         action,
                         next_state,
                         reward,
                         lr):
        """Update the Q table using the Bellman equation.

        Args:
            state (list): current state of the agent.
            action (int): selected action.
            next_state (list): next state of the agent.
            reward (float): reward for the selected action.
            lr (float): learning rate.
        """
        q_current = self.Q[state[0], state[1], state[2], action]
        q_next = np.max(self.Q[next_state[0], next_state[1],
                        next_state[2], :].todense())

        self.Q[state[0], state[1], state[2], action] = \
            q_current + lr * (reward + self.gamma * q_next - q_current)
