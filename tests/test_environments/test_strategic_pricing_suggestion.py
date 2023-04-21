from library import *

# Import functions
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

    def test_get_counts_and_rewards_per_bin(self):
        """Test get_counts_and_rewards_per_bin method."""
        binned = np.array([[1, 2, 3, 1], [1, 2, 3, 2], [1, 2, 3, 1], [1, 2, 3, 2]])
        self.env._reward_mdp_data = np.array([[1], [2], [1], [2]])
        result = self.env._get_counts_and_rewards_per_bin(binned)
        target = {'1,2,3,1': [2, 2], '1,2,3,2': [2, 4]}
        assert isinstance(result, dict)
        assert result == target

        