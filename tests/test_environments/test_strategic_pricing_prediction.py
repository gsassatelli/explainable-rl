from library import *
from tests.test_environments.test_strategic_pricing import TestStrategicPricing
from src.environments.strategic_pricing_prediction import \
    StrategicPricingPredictionMDP


class TestStrategicPricingPredictionMDP(TestStrategicPricing):
    """Test StrategicPricingPredictionMDP class."""

    def setUp(self) -> None:
        """Set up test fixtures, if any."""
        self.env = StrategicPricingPredictionMDP(self.dh)

    def tearDown(self) -> None:
        """Tear down test fixtures, if any."""
        del self.env

    def test_transform_df_to_numpy(self):
        """Test transform_df_to_numpy method."""
        target_states = self.dh.get_states().to_numpy()
        target_actions = np.array(self.dh.get_actions())
        target_rewards = self.dh.get_rewards().to_numpy()
        self.env._transform_df_to_numpy()
        result_states = self.env._state_mdp_data
        result_actions = self.env._action_mdp_data
        result_reward = self.env._reward_mdp_data
        assert np.array_equal(target_states, result_states)
        assert np.array_equal(target_actions, result_actions)
        assert np.array_equal(target_rewards, result_reward)

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

    def test_get_counts_and_rewards_per_bin(self):
        """Test get_counts_and_rewards_per_bin method."""
        binned = np.array([[1, 2, 3, 1], [1, 2, 3, 2], [1, 2, 3, 1], [1, 2, 3, 2]])
        self.env._reward_mdp_data = np.array([[1], [2], [1], [2]])
        result = self.env._get_counts_and_rewards_per_bin(binned)
        target = {'1,2,3,1': [2, 2], '1,2,3,2': [2, 4]}
        assert result == target

    def test_create_average_reward_matrix(self):
        """Test create_average_reward_matrix method."""
        counts_rewards = {'1,2,3,1': [2, 2], '1,2,3,2': [2, 4]}
        result = self.env._create_average_reward_matrix(counts_rewards)
        target = np.zeros((10, 10, 10, 10))
        target[1, 2, 3, 1] = 1.0
        target[1, 2, 3, 2] = 2.0
        target = sparse.COO(target)
        assert result == target

    def test_make_rewards_from_data(self):
        """Test make_rewards_from_data method."""
        self.env._state_mdp_data = pd.DataFrame([[0.1, 0.23, 0.4], [0.5, 0.6, 0.7]])
        self.env._action_mdp_data = pd.DataFrame([[0.1], [0.2]])
        self.env._reward_mdp_data = pd.DataFrame([[1], [2]])
        self.env.bins = [10, 10, 10, 10]
        result = self.env._make_rewards_from_data()

        target = np.zeros((10, 10, 10, 10))
        target[1, 2, 3, 1] = 1
        target[5, 6, 7, 2] = 2
        target = sparse.COO(target)
        assert result == target

    def test_reset(self):
        """Test reset method."""
        self.env._state_mdp_data = np.array([[0.1, 0.23, 0.4], [0.5, 0.6, 0.7]])
        result = self.env.reset()
        assert result in [[1, 2, 4], [5, 6, 7]]

    def test_step(self):
        """Test step method."""
        self.env._state_mdp_data = pd.DataFrame(
            [[0.1, 0.23, 0.4], [0.5, 0.6, 0.7]])
        self.env._action_mdp_data = pd.DataFrame([[0.1], [0.2]])
        self.env._reward_mdp_data = pd.DataFrame([[1], [2]])
        self.env.bins = [10, 10, 10, 10]
        self.env._make_rewards_from_data()
        state = [1, 2, 3]
        action = [1]
        result = self.env.step(state, action)
        assert result == (state, state, 1, True)
