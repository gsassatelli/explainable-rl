# Import packages
from typing import Tuple
import torch
import numpy as np

# Import environment and agent
from agent import Agent
from environment import MDP

class Engine:

    __slots__ = ["states", "actions", "rewards", "next_states", "agent_type", "env_type",
                 "agent", "env"]

    def __init__(self, 
                 dataset: np.array[np.array, np.array, np.arraynp.array],
                 agent_type: str, 
                 env_type: str,
                 episode_flag: bool = False):
        """Initialise the Engine superclass.
        Dataset is a multi-type np.array [state, action, reward, next_state].

        TODO: add hyperparameters for training wherever they need to go.
        """
        # Split dataset
        self.states = dataset[0]
        self.actions = dataset[1]
        self.rewards = dataset[2]
        # If episode is more than one step, save next states
        if episode_flag:
            self.next_states = dataset[3]
        # Initialize agent
        self.agent_type = agent_type
        self.agent = None
        # Initialize environment
        self.env_type = env_type
        self.env = None

    def create_world(self):
        """Create the Agent and MDP instances for the given task.

        """
        # Create chosen agent
        self.create_agent()
        # Create chosen environment
        self.create_env()

    def create_agent(self):
        """Create an agent and store it in Engine.

        """
        if self.agent_type == "q-learner":
            self.agent = Agent()
            # TODO: change if different name
        pass

    def create_env(self):
        """Create an env and store it in Engine.

        """
        self.env = MDP(self.states, self.actions, self.rewards)
        pass

    def train_agent(self, n_episodes, n_steps):
        """Train the agent for a chosen number of steps and episodes."""
        # TODO Giulia
        self.agent.fit(n_episodes, n_steps)
        pass

    def what_should_i_do_given_state(self):
        """Implement this."""
        # TODO Giulia
        pass

    def get_results(self):
        """Get the results of training.

        This could be the average return after convergence.
        """
        # TODO Giulia
        pass

    def save_parameters(self):
        """Save the parameters learned during training.

        This could be e.g. the q-values, the policy, or any other learned parameters.
        """
        # TODO Giulia
        pass

    def evaluate(self, state):
        """Evaluate the learned policy at a particular state.

        This method returns the action that should be taken given a state.
        """
        # TODO Giulia
        pass