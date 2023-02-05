# Import packages
from typing import Tuple
import torch
import numpy as np

# Import environment and agent
from agent import Agent
from environment import MDP

class Engine:

    __slots__ = ["dataset", "agent_type", "env_type", "agent", "env", "gamma",
                 "episode_flag", "num_episodes", "num_steps", "policy", "q_table"]

    def __init__(self, 
                 dataset: np.array[np.array, np.array, np.arraynp.array],
                 agent_type: str, 
                 env_type: str,
                 num_episodes: int,
                 num_steps: int):
        """Initialise the Engine superclass.
        Dataset is a multi-type np.array [state, action, reward, next_state].

        """
        # Save dataset to train
        self.dataset = dataset

        # Hyperparameters
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.gamma = gamma

        # Initialize agent
        self.agent_type = agent_type
        self.agent = None

        # Initialize environment
        self.env_type = env_type
        self.env = None

        # Parameters of the agent
        self.policy = None
        self.q_table = None

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
            # Initialize agent
            self.agent = Agent()
            # Create agent
            self.agent.create_agent()

    def create_env(self):
        """Create an env and store it in Engine.

        """
        # Initialize environment
        self.env = MDP(self.dataset, self.gamma)
        # Create environment
        self.env.initialise_env()

    def train_agent(self):
        """Train the agent for a chosen number of steps and episodes.

        """
        # Fit the agent
        self.agent.fit(self.num_episodes, self.num_episodes)

    def what_should_i_do_given_state(self):
        """Implement this."""
        # TODO: Not sure if this is needed
        pass

    def get_results(self):
        """Get the results of training.

        This could be the average return after convergence.
        """
        # TODO: Not sure if this is needed
        pass

    def save_parameters(self):
        """Save the parameters learned during training.

        This could be e.g. the q-values, the policy, or any other learned parameters.

        TODO: Not sure this function is needed, can call directly agent
        TODO: Epsilon greedy policy already contains q-values, remove it?
        """
        # Save parameters of the trained agent to predict
        self.policy = self.agent.policy
        self.q_table = self.agent.q_table


    def evaluate(self, state):
        """Evaluate the learned policy at a particular state.

        This method returns the action that should be taken given a state.
        """
        # TODO: ensure that here output is action with max q values (NO exploration)
        return self.agent.epsilon_greedy_policy(state)
