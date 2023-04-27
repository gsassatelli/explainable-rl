from explainable_RL.foundation.library import *

# Import functions
from explainable_RL.foundation.engine import Engine
from explainable_RL.data_handler.data_handler import DataHandler
from explainable_RL.agents.q_learner import QLearningAgent
from explainable_RL.agents.sarsa import SarsaAgent
from explainable_RL.agents.sarsa_lambda import SarsaLambdaAgent
from explainable_RL.agents.double_q_learner import DoubleQLearner
from explainable_RL.environments.strategic_pricing_prediction import StrategicPricingPredictionMDP
from explainable_RL.environments.strategic_pricing_suggestion import StrategicPricingSuggestionMDP
from tests.test_hyperparams import hyperparam_dict


class TestEngine(unittest.TestCase):
    """Test the Engine class."""

    dh = None

    @classmethod
    def setUpClass(cls):
        """Set up the data handler for the tests."""
        dataset = pd.read_csv(
            hyperparam_dict["dataset"]["data_path"],
            sep=hyperparam_dict["dataset"]["col_delimiter"],
        )
        cls.dh = DataHandler(
            hyperparam_dict=hyperparam_dict, dataset=dataset, test_dataset=dataset
        )

    def setUp(self) -> None:
        """Set up the engine for the tests."""
        self.engine = Engine(self.dh, hyperparam_dict=hyperparam_dict)

    def tearDown(self) -> None:
        """Tear down the engine after the tests."""
        del self.engine

    def test_create_world_agents(self):
        """Test the create_world method with different agent types."""
        types = ["q_learner", "sarsa", "sarsa_lambda", "double_q_learner"]
        types_dict = {
            "q_learner": QLearningAgent,
            "sarsa": SarsaAgent,
            "sarsa_lambda": SarsaLambdaAgent,
            "double_q_learner": DoubleQLearner,
        }

        for agent_type in types:
            self.engine.agent_type = agent_type
            self.engine.create_world()
            assert isinstance(self.engine.agent, types_dict[agent_type])
            assert isinstance(self.engine.env, StrategicPricingPredictionMDP)

    def test_create_agent(self):
        """Test the create_agent method with different agent types."""
        types = ["q_learner", "sarsa", "sarsa_lambda", "double_q_learner"]
        types_dict = {
            "q_learner": QLearningAgent,
            "sarsa": SarsaAgent,
            "sarsa_lambda": SarsaLambdaAgent,
            "double_q_learner": DoubleQLearner,
        }

        for agent_type in types:
            self.engine.agent_type = agent_type
            self.engine.env = StrategicPricingPredictionMDP(self.dh, self.engine.bins)
            self.engine.create_agent()
            assert isinstance(self.engine.agent, types_dict[agent_type])
            assert isinstance(self.engine.env, StrategicPricingPredictionMDP)

    def test_create_env(self):
        """Test the create_env method with different env types."""
        types = ["strategic_pricing_predict", "strategic_pricing_suggest"]
        types_dict = {
            "strategic_pricing_predict": StrategicPricingPredictionMDP,
            "strategic_pricing_suggest": StrategicPricingSuggestionMDP,
        }

        for env_type in types:
            self.engine.env_type = env_type
            self.engine.create_env()
            self.engine.agent = QLearningAgent(self.engine.env, gamma=0.8)
            assert isinstance(self.engine.env, types_dict[env_type])
            assert isinstance(self.engine.agent, QLearningAgent)

    def test_train_agent(self):
        """Test the train_agent method."""
        self.engine.create_world()
        original_q = copy.deepcopy(self.engine.agent.Q)
        self.engine.train_agent()
        assert self.engine.agent.Q is not None
        assert self.engine.agent.Q is not original_q

    def test_get_bins(self):
        """Test the get_bins method."""
        bins = self.engine._get_bins()
        target = [10, 2, 2, 5]
        assert isinstance(bins, list)
        assert len(bins) == 4
        assert bins == target
