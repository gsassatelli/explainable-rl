class ShapValues:
    """ This class implements the SHAP values algorithm.
    """

    __slots__ = ["sample", "features"]

    def __init__(self,
                 sample,
                 features):
        self.sample = sample
        self.features = features

    def compute_shape_values(self):

        if not self.verify_sample_length():
            raise ValueError("The sample length is not correct.")

    # - Find cell for that sample
    # - Verify if cell has been visited
    # - Loop over all state dimensions:
    #   - Loop for n times
    #       - Sample with dimension fixed until sample has been visited
    #       - Get action for that sample
    #       - Sample randomly until sample has been visited
    #       - Get action for that sample
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
        pass

    def verify_cell_availability(self):
        """ This function verifies if the cell has been visited.
        """
        pass

