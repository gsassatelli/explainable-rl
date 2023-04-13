import numpy as np

class ShapValues:
    """ This class implements the SHAP values algorithm.
    """

    __slots__ = ["sample", "features", "env", "Q", "number_of_samples"]

    def __init__(self,
                 sample,
                 features,
                 env,
                 Q,
                 number_of_samples=1000):
        self.sample = sample
        self.features = features
        self.env = env
        self.Q = Q
        self.number_of_samples = number_of_samples

    def compute_shape_values(self):

        # Verify if sample length is correct
        if not self.verify_sample_length():
            raise ValueError("The sample length is not correct.")

        # Bin sample
        binned_sample = self.bin_sample()

        # Verify if cell has been visited
        if not self.verify_cell_availability(binned_sample):
            raise ValueError("The cell has not been visited by the agent.")

        # Loop over all state dimensions
        shap_values = []
        for shap_ft in range(len(self.features)):
            num_bins_per_shap_ft = self.env.bins[shap_ft]
            samples_plus = np.zeros(self.number_of_samples)
            samples_minus = np.zeros(self.number_of_samples)
            for sample in range(self.number_of_samples):
                # Sample once feature shap for random sample
                for ft in range(len(self.features)):
                    True
                    # Sample twice or once each feature and add to 2 sample
                    # Bin 2 samples
                    # Check if we have Q values for 2 sample
                    # If not repeat
                # Find best Q values for 2 samples
                # Find actions that Q
                # Append the 2 actions
            # Compute difference between arrays
            # Compute mean
            # Append shap value for that feature!
                

                    # Sample with dimension fixed until sample has been visited
                    # Get action for that sample
                    # Sample randomly until sample has been visited
                    # Get action for that sample


    #   - Calculate difference
    #   - Calculate average difference
    #   - Output q-value for that state dimension

        pass

    def verify_sample_length(self):
        """ This function verifies if the sample length is correct.
        """
        if len(self.sample) != len(self.features):
            return False
        return True

    def bin_sample(self):
        """ This function bins the sample.
        """
        binned_sample = self.env._bin_state(self.sample)
        return binned_sample

    def verify_cell_availability(self, binned_sample):
        """ This function verifies if the cell has been visited.
        """
        num_actions = self.env.bins[-1]
        # The last element in the bins list is the number of actions
        for a in range(num_actions):
            index_current = tuple(list(binned_sample) + [a])
            if self.Q[index_current] != 0:
                # At least one action has been visited for this state has been visited
                return True
        return False

