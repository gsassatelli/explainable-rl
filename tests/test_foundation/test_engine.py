# Import functions
from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler
from src.agents.q_learner import QLearningAgent
from src.agents.sarsa import SarsaAgent
from src.agents.sarsa_lambda import SarsaLambdaAgent
from src.agents.double_q_learner import DoubleQLearner
from src.environments.strategic_pricing_prediction import StrategicPricingPredictionMDP
from src.environments.strategic_pricing_suggestion import StrategicPricingSuggestionMDP

# Import packages
import copy
import unittest


class TestEngine(unittest.TestCase):
    """Test the Engine class."""

    dh = None

    @classmethod
    def setUpClass(cls):
        """Set up the data handler for the tests."""
        states = ['competitorPrice', 'adFlag', 'availability']
        actions = ['price']
        rewards = ['revenue']
        n_samples = 50
        cls.dh = DataHandler('tests/test_env_data.csv', states, actions, rewards, n_samples=n_samples)
        cls.dh.prepare_data_for_engine(col_delimiter=',', cols_to_normalise=states + actions)

    def setUp(self) -> None:
        """Set up the engine for the tests."""
        self.engine = Engine(self.dh,
                             agent_type="q_learner",
                             env_type="strategic_pricing_predict",
                             bins=[10, 10, 10, 10],
                             num_episodes=100,
                             num_steps=1)

    def tearDown(self) -> None:
        """Tear down the engine after the tests."""
        del self.engine

    def test_create_world_agents(self):
        """Test the create_world method with different agent types."""
        types = ["q_learner", "sarsa", "sarsa_lambda", "double_q_learner"]
        types_dict = {"q_learner": QLearningAgent,
                        "sarsa": SarsaAgent,
                        "sarsa_lambda": SarsaLambdaAgent,
                        "double_q_learner": DoubleQLearner}

        for agent_type in types:
            self.engine.agent_type = agent_type
            self.engine.create_world()
            assert isinstance(self.engine.agent, types_dict[agent_type])
            assert isinstance(self.engine.env, StrategicPricingPredictionMDP)

    def test_create_agent(self):
        """Test the create_agent method with different agent types."""
        types = ["q_learner", "sarsa", "sarsa_lambda", "double_q_learner"]
        types_dict = {"q_learner": QLearningAgent,
                      "sarsa": SarsaAgent,
                      "sarsa_lambda": SarsaLambdaAgent,
                      "double_q_learner": DoubleQLearner}

        for agent_type in types:
            self.engine.agent_type = agent_type
            self.engine.env = StrategicPricingPredictionMDP(self.dh, self.engine.bins)
            self.engine.create_agent()
            assert isinstance(self.engine.agent, types_dict[agent_type])
            assert isinstance(self.engine.env, StrategicPricingPredictionMDP)

    def test_create_env(self):
        """Test the create_env method with different env types."""
        types = ["strategic_pricing_predict", "strategic_pricing_suggest"]
        types_dict = {"strategic_pricing_predict": StrategicPricingPredictionMDP,
                      "strategic_pricing_suggest": StrategicPricingSuggestionMDP}

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

    def test_get_results(self):
        pass

    def test_save_parameters(self):
        pass

    def test_inverse_scale_features(self):
        pass

    def test_build_evaluation(self):
        pass

    def test_evaluate_total_agent_reward(self):
        pass

    def test_evaluate_total_hist_reward(self):
        pass

    def test_evaluate_agent(self):
        pass

    # TODO: from Giulia, do we need to test this?



