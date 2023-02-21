# Import packages
from typing import Tuple
import torch
import numpy as np
import pandas as pd

# Import environment and agent
from foundation.agent import Agent
from foundation.environment import MDP

class Engine:

    __slots__ = ["mdp_data", "agent_type", "env_type", "agent", "env", "gamma",
                 "episode_flag", "num_episodes", "num_steps", "policy", "q_table"]

    def __init__(self, 
                 mdp_data: pd.DataFrame,
                 agent_type: str, 
                 env_type: str,
                 num_episodes: int,
                 num_steps: int,
                 gamma: float = 0.9):
        """Initialise the Engine superclass.

        """
        # Save dataset to train
        self.mdp_data = mdp_data

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
        # Create chosen environment
        print("Initialize environment")
        self.create_env()
        
        # Create chosen agent
        print("Initialize agent")
        self.create_agent()

    def create_agent(self):
        """Create an agent and store it in Engine.

        """
        if self.agent_type == "q_learner":
            # Initialize agent
            self.agent = Agent(self.env)

    def create_env(self):
        """Create an env and store it in Engine.

        """
        # Initialize environment
        self.env = MDP(self.mdp_data)

    def train_agent(self):
        """Train the agent for a chosen number of steps and episodes.

        """
        # Fit the agent
        self.agent.fit(self.num_episodes, self.num_episodes)

    def get_results(self):
        """Get the results of training.

        TODO: Next sprint to compare 2 agents
              This could be the average return after convergence.
        """
        #
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

        Args:
            state: state for which an action needs to be predicted.
        Returns:
            action_reward: action and reward for a given state

        TODO: ensure that here output is action with max q values (NO exploration)
        """
        # Get both action & reward
        action_reward = self.agent._epsilon_greedy_policy(state)
        return action_reward

