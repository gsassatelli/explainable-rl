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

    def test_bin_state_action_space(self):
        """Test bin_state_action_space method."""
        zipped = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        self.env.bins = [10, 10, 5, 5]
        result = self.env._bin_state_action_space(zipped)
        target = np.array([[1, 2, 1, 2], [5, 6, 3, 4]])
        assert np.array_equal(result, target)

    def test_bin_states(self):
        """Test bin_states method."""
        states = [[0.34, 0.19, 0.79], [0.35, 0.37, 0.44]]
        self.env.bins = [10, 10, 5]
        result = self.env.bin_states(states)
        target = [[3, 1, 3], [3, 3, 2]]
        assert result == target

    def test_debin_states(self):
        """Test debin_states method."""
        states = [[3, 1, 3], [3, 3, 2]]
        self.env.bins = [10, 10, 5]
        result = self.env.debin_states(states)
        target = [[0.35, 0.15, 0.7], [0.35, 0.35, 0.5]]
        assert result == target

    def test_bin_state(self):
        """Test bin_state method."""
        state = np.array([0.34, 0.19, 0.79])
        self.env.bins = [10, 10, 5]
        result = self.env.bin_state(state)
        target = [3, 1, 3]
        assert result == target

    def test_debin_state(self):
        """Test debin_state method."""
        state = np.array([3, 1, 3])
        self.env.bins = [10, 10, 5]
        result = self.env._debin_state(state)
        target = [0.35, 0.15, 0.7]
        assert result == target






