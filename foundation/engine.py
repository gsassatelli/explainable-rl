class Engine:
    def __init__(self, dataset, agent_type):
        """Initialise the Engine superclass."""
        pass

    def create_world(self):
        """Create the Agent and MDP instances for the given task."""
        pass

    def train_agent(self, n_episodes, n_steps):
        """Train the agent for a chosen number of steps and episodes."""
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