# Import packages
import numpy as np
import sparse
import random
from datetime import datetime

class Agent():
    def __init__(self, env, gamma: float = 0.9):
        """Initialise the agent class.
        
        Args:
            env: mdp.
            gamma: discount factor.
        
        TODO: fix circular import (we cannot import MDP into agent 
        and agent into MDP)
        """        
        self.env = env
        self.Q = None
        self.state_to_action = None
        self.gamma = 0.9
        # self.state = None
        # self.rewards_per_episode = []
        # self.total_episode_reward = 0

        self.initialize_agent()


    def fit(self, n_episodes: int, n_steps: int, lr: float = 1):
        """Fit agent to the dataset.

        Args:
            n_episodes: number of training episodes.
            n_steps: maximum number of steps within each episode.
            lr: learning rate.

        TODO: decay.
        """
        print("Apply q-learning and update q-table")
        for _ in range(n_episodes):
            # self.state = self.env.reset()
            # env reset is not working so implementing it here:
            self.state_str = random.choice(list(self.state_to_action.keys()))
            self.state = [int(s) for s in self.state_str.split(",")]
            for i in range(n_steps):
                action = self.epsilon_greedy_policy(self.state_str)
                state, next_state, reward, done = self.env.step(self.state,
                                                                action)
                self.update_q_values(state, action, next_state, reward, lr)
                self.state = next_state
                if done:
                    break

        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(timestamp + ": Finished training :) !")
        print(f"Example Q-table for state {[1,9,0]}: {self.Q[1,9,0].todense()}")
        print(f"Example Q-table for state {[1, 0, 0]}: {self.Q[1, 0, 0].todense()}")

    def update_q_values(self, state: list,
                        action: int,
                        next_state: list,
                        reward: float,
                        lr: float):
        """Update the Q table using the Bellman equation.

        Args:
            state: current state of the agent.
            action: selected action.
            next_state: state to which the agent transitions.
            reward: reward obtained with the selected action.
            lr: learning rate.
        
        TODO: implement part related to gamma (not necessary for now
              because we have a myopic env)
        """
        self.Q[state[0],state[1],state[2],action] = \
            (1-lr)*self.Q[state[0],state[1],state[2],action] + lr*(reward)

    def initialize_agent(self):
        """Initialize agent (called by agent when the episode starts)."""
        # reset environment
        self.env.reset()

        print("Create q-table")
        # create q-table
        coords = []
        for state_str, actions in self.env.state_to_action.items():
            state = [int(s) for s in state_str.split(",")]
            actions = list(actions)
            for action in actions:
                coords.append(state+[action])
        q_values = np.zeros(len(coords))
        coords = np.array(coords).T.tolist()
        
        # create COO (read only) matrix
        self.Q = sparse.COO(coords, q_values)
        
        # convert to DOK
        self.Q = sparse.DOK.from_numpy(self.Q.todense())
        
        # create state to action mapping
        self.state_to_action = self.env.state_to_action


    def epsilon_greedy_policy(self, state):
        """Returns the epsilon greedy action.

        Args:
            state: current state.
        
        TODO: implement the "epsilon part"
        """
        return random.choice(list(self.state_to_action[state]))
        
