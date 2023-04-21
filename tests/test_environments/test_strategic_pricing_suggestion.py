from library import *

# Import functions
from src.data_handler.data_handler import DataHandler
from src.environments.strategic_pricing_suggestion import StrategicPricingSuggestionMDP
from tests.test_environments.test_strategic_pricing import TestStrategicPricing


class TestStrategicPricingSuggestionMDP(TestStrategicPricing):
    """Test StrategicPricingSuggestionMDP class."""

    def setUp(self) -> None:
        """Set up test fixtures, if any."""
        self.env = StrategicPricingSuggestionMDP(self.dh)

    def tearDown(self) -> None:
        """Tear down test fixtures, if any."""
        del self.env

    # TODO: From Giulia, maybe we need to add more tests here
        