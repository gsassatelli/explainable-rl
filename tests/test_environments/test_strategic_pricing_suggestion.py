# Import package
import unittest

# Import functions
from src.data_handler.data_handler import DataHandler
from src.environments.strategic_pricing_suggestion import StrategicPricingSuggestionMDP


class TestStrategicPricingSuggestionMDP(unittest.TestCase):

    dh = None

    @classmethod
    def setUpClass(cls) -> None:
        states = ['competitorPrice', 'adFlag', 'availability', 'price']
        actions = [price_bin/10 for price_bin in range(1, 11)]
        rewards = ['revenue']
        n_samples = 50
        cls.dh = DataHandler('tests/test_env_data.csv', states, actions, rewards, n_samples=n_samples)
        cls.dh.prepare_data_for_engine(col_delimiter=',',
                                       cols_to_normalise=states)

    def setUp(self) -> None:
        self.env = StrategicPricingSuggestionMDP(self.dh)

    def tearDown(self) -> None:
        del self.env

    def test_type(self):
        assert isinstance(self.env, StrategicPricingSuggestionMDP)
        