import agent

class MDP:
    def _init_(self, dataset, discount_factor):
        """Initialise the MDP superclass."""
        self.initaliase_env()
        pass

    def initaliase_env(self):
        """Create the environment given the MDP information.

        TODO: tricky part"""
        pass

    def reset_env(self):
        """Reset environment and return a randomised state."""

        return []
        pass

    def step(self, state, action):
        """Take a step in the environment. Done means is the env terminated.

        Returns state, next state, reward, done."""

