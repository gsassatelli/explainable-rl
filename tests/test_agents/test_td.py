from explainable_rl.foundation.library import *

# Import functions
from explainable_rl.agents.td import TD
from explainable_rl.environments.strategic_pricing_prediction import StrategicPricingPredictionMDP
from explainable_rl.data_handler.data_handler import DataHandler
from tests.test_hyperparams import hyperparam_dict


class TestTD(unittest.TestCase):
    """Test the TD class."""

    dh = None

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test class."""
        dataset = pd.read_csv(
            hyperparam_dict["dataset"]["data_path"],
            sep=hyperparam_dict["dataset"]["col_delimiter"],
        )
        cls.dh = DataHandler(
            hyperparam_dict=hyperparam_dict, dataset=dataset, test_dataset=dataset
        )

    def setUp(self) -> None:
        """Set up the test class."""
        self.env = StrategicPricingPredictionMDP(self.dh)
        self.agent = TD(self.env, gamma=0.9)

    def tearDown(self) -> None:
        """Tear down the test class."""
        del self.agent

    def test_update_q_values(self):
        """Implemented in tests for subclasses."""
        pass

    def test_step(self):
        """Implemented in tests for subclasses."""
        pass

    def test_init_q_table(self):
        """Test the init_q_table method."""
        self.agent.env.bins = [10, 5, 4, 6]
        self.agent._init_q_table()
        assert self.agent.Q.shape == (10, 5, 4, 6)

    def test_convert_to_string(self):
        """Test the convert_to_string method."""
        state = [0, 5, 3, 2]
        result = self.agent._convert_to_string(state)
        target = "0,5,3,2"
        assert result == target

    def test_epsilon_greedy_policy(self):
        """Test the epsilon_greedy_policy method."""
        epsilon = 0
        state = [0, 0, 0]
        self.agent._init_q_table()
        self.agent.Q[0, 0, 0, 2] = 1.5
        self.agent.state = [0, 0, 0]
        result = self.agent._epsilon_greedy_policy(state=state, epsilon=epsilon)
        assert result == 2

    def test_uncertainty_informed_policy(self):
        """Test the uncertainty_informed_policy method."""
        epsilon = 0
        state = [0, 0, 0]
        self.agent._init_q_table()
        self.agent.Q[0, 0, 0, 2] = 1.5
        self.env.bins_dict = {"0,0,0,2": [2, 2]}
        self.env.state_to_action = {"0,0,0": {2}}
        self.agent.state = [0, 0, 0]
        result = self.agent.uncertainty_informed_policy(
            state=state, epsilon=epsilon, use_uncertainty=True
        )
        assert result == 2

    def test_create_tables(self):
        """Test the create_tables method."""
        self.agent.env.bins = [10, 5, 4, 6]
        self.agent.create_tables()
        assert self.agent.Q.shape == (10, 5, 4, 6)
        assert self.agent.state_to_action is not None

    def test_fit(self):
        """Implemented in tests for subclasses."""
        pass
