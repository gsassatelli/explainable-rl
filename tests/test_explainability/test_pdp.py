import unittest
from src.explainability.pdp import PDP

class TestPDP(unittest.TestCase):
    """Test PDP class.
    """

    def setUp(self):
        """Setup PDP class.
        """
        bins = [10, 10, 10, 10]
        minmax_scalers = {}
        action_labels = ['price']
        state_labels = ['competitorPrice', 'adFlag', 'availability']
        self.PDP = PDP(bins, minmax_scalers, action_labels, state_labels)
    def test_create_pdp(self):
        """Test creation of PDP object.
        """
        assert type(self.PDP) == PDP

    def test_get_digitized_pdp(self):
        """Test digitized pdp.
        """
        pass

    def test_get_denorm_actions(self):
        """Test denormalized actions.
        """
        pass

    def test_get_denorm_states(self):
        """Test denormalized states.
        """
        pass

    def test_build_data_for_plots(self):
        """Test build data for plots.
        """
        pass




