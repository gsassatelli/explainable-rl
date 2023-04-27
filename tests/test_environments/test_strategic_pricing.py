from explainableRL.foundation.library import *

# Import functions
from explainableRL.data_handler.data_handler import DataHandler
from explainableRL.environments.strategic_pricing import StrategicPricing
from tests.test_hyperparams import hyperparam_dict


class TestStrategicPricing(unittest.TestCase):
    """Test StrategicPricing class."""

    dh = None

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test class."""
        dataset = pd.read_csv(
            hyperparam_dict["dataset"]["data_path"],
            sep=hyperparam_dict["dataset"]["col_delimiter"],
        )
        cls.dh = DataHandler(
            hyperparam_dict=hyperparam_dict, dataset=dataset, test_dataset=dataset
        )

    def setUp(self) -> None:
        """Set up test objects."""
        self.env = StrategicPricing(self.dh)

    def tearDown(self) -> None:
        """Tear down test objects."""
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

    def test_reset(self):
        """Test reset method."""
        self.env._state_mdp_data = np.array([[0.1, 0.23, 0.4], [0.5, 0.6, 0.7]])
        result = self.env.reset()
        assert result in [[1, 2, 4], [5, 6, 7]]

    def test_get_state_to_action(self):
        binned = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 2]])
        target = {"1,2,3": {4, 2}, "5,6,7": {8}}
        result = self.env._get_state_to_action(binned)
        assert result == target
