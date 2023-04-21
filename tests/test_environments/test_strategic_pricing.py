from src.data_handler.data_handler import DataHandler
from src.environments.strategic_pricing import StrategicPricing
from library import *


class TestStrategicPricing(unittest.TestCase):
    dh = None

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures, if any."""
        states = ['competitorPrice', 'adFlag', 'availability']
        actions = ['price']
        rewards = ['revenue']
        n_samples = 50
        cls.dh = DataHandler('tests/test_env_data.csv', states, actions, rewards, n_samples=n_samples)
        cls.dh.prepare_data_for_engine(col_delimiter=',',
                                       cols_to_normalise=states+actions)
    def setUp(self) -> None:
        """Set up test fixtures, if any."""
        self.env = StrategicPricing(self.dh)

    def tearDown(self) -> None:
        del self.env

    def test_join_state_action(self):
        """Test join_state_action method."""
        self.env._state_mdp_data = np.array([[1, 2, 3], [4, 5, 6]])
        self.env._action_mdp_data = np.array([[1], [2]])
        self.env._reward_mdp_data = np.array([[1], [2]])
        result = self.env._join_state_action()
        target = np.array([[1, 2, 3, 1], [4, 5, 6, 2]])
        assert np.array_equal(result, target)




