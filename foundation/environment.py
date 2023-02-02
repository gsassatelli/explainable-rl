import agent

class MDP:
    def _init_(self, dataset, discount_factor):
        """Initialise the MDP superclass."""
        self.average_rewards = None
        self.dataset = dataset
        self.initialise_env()
        pass

    def initialise_env(self):
        """Create the environment given the MDP information.

        TODO: tricky part"""
        self.average_rewards = self.make_rewards_from_data()
        pass

    def make_rewards_from_data(self):
        """Make the state-action reward table from dataset."""
        pass

    def reset_env(self):
        """Reset environment and return a randomised state."""

        return []
        pass

    def step(self, state, action):
        """Take a step in the environment. Done means is the env terminated.

        Returns state, next state, reward, done."""

