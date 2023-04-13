class ShapValues:
    """ This class implements the SHAP values algorithm.
    """

    __slots__ = []

    def __init__(self):
        pass

    # TODO:
    # - Get sample from user
    # - Find cell for that sample >> What if the sample is not in the dataset?
    # - Loop over all state dimensions:
    #   - Loop for n times
    #       - Sample with dimension fixed until sample is in the dataset
    #       - Get action for that sample
    #       - Sample randomly until sample is in the dataset
    #       - Get action for that sample
    #   - Calculate difference
    #   - Calculate average difference
    #   - Output q-value for that state dimension

    # Start code
