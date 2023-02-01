from typing import Tuple
import torch
import numpy as np

class Engine:
    def __init__(self, dataset: np.array[np.array, np.array, np.array,
                                         np.array],
                 agent_type: str, env_type: str):
        """Initialise the Engine superclass.

        Dataset is a multi-type np.array [state, action, reward, next_state].

        TODO: keep slots here
        TODO: add hyperparameters for training wherever they need to go.
        """
        self.agent_type = agent_type
        pass

    def create_world(self):
        """Create the Agent and MDP instances for the given task.

        """
        self.create_agent()
        self.create_env()
        pass

    def create_agent(self):
        """Create an agent and store it in Engine.

        """
        # if self.agent_type == 'blah':
        #   self.agent = 'blah'
        pass

    def create_env(self):
        """Create an env and store it in Engine.

        Pass dataset to env.
        """
        # self.env = Environment(dataset)
        pass

    def train_agent(self, n_episodes, n_steps):
        """Train the agent for a chosen number of steps and episodes."""

        self.agent.fit(n_episodes, n_steps)
        pass

    def what_should_i_do_given_state(self):
        """Implement this."""
        pass

    def get_results(self):
        """Get the results of training.

        This could be the average return after convergence.
        """
        pass

    def save_parameters(self):
        """Save the parameters learned during training.

        This could be e.g. the q-values, the policy, or any other learned parameters.
        """
        pass

    def evaluate(self, state):
        """Evaluate the learned policy at a particular state.

        This method returns the action that should be taken given a state.
        """
        pass