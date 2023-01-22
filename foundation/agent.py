class agent():
    def __init__(self, state, q_values, policy):
        """Initialise the agent superclass."""
        pass

    def create_agent(self):
        """Reset the environment. Called by agent when the episode starts."""
        pass

    def get_action(self, state):
        """Get the action that the agent will execute.

        This can be from e.g. the q-values or from the algorithm.
        """
        pass

    def update_learning(self, dataset):
        """Fit a step of learning to the dataset."""
        pass

    
