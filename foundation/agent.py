import numpy as np

class Agent():
    def __init__(self, state, q_values, policy, env):
        """Initialise the agent superclass."""
        self.env = env
        self.state = None
        self.q = ...
        # self.rewards_per_episode = []
        # self.total_episode_reward = 0
        pass

    def fit(self, n_episodes, n_steps):
        """Fit agent to dataset.

        TODO: decay. """
        for _ in n_episodes:

            self.state = self.env.reset()
            for i in range(n_steps):
                action = self.epsilon_greedy_policy(self.state)
                state, next_state, reward, done = self.env.step(self.state,
                                                                action)
                self.update_q_values(state, action, next_state, reward)
                self.state = next_state
                if done:
                    break

        pass

    def update_q_values(self, state, action, next_state, reward):
        """Update the q table of the agent using Bellman."""

        pass

    def create_agent(self):
        """Reset the environment. Called by agent when the episode starts."""
        pass

    def epsilon_greedy_policy(self, state):
        """Returns the epsilon greedy action."""
        pass
    
