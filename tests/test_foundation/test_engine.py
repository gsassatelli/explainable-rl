import unittest
from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler
from src.agents.q_learner import QLearningAgent
from src.agents.sarsa import SarsaAgent
from src.agents.sarsa_lambda import SarsaLambdaAgent
from src.agents.double_q_learner import DoubleQLearner
from src.environments.strategic_pricing_suggestion import StrategicPricingSuggestionMDP
from src.environments.strategic_pricing_prediction import StrategicPricingPredictionMDP


class TestEngine(unittest.TestCase):

    dh = None

    @classmethod
    def setUpClass(cls):
        states = ['competitorPrice', 'adFlag', 'availability']
        actions = ['price']
        rewards = ['revenue']
        n_samples = 50
        cls.dh = DataHandler('tests/test_env_data.csv', states, actions, rewards, n_samples=n_samples)
        cls.dh.prepare_data_for_engine(col_delimiter=',', cols_to_normalise=states + actions)

    def setUp(self) -> None:
        self.engine = Engine(self.dh,
                             agent_type="q_learner",
                             env_type="strategic_pricing",
                             bins=[10, 10, 10, 10],
                             num_episodes=100,
                             num_steps=1)

    def tearDown(self) -> None:
        del self.engine

    def test_create_world(self):
        types = ["q_learner", "sarsa", "sarsa_lambda", "double_q_learner"]
        types_dict = {"q_learner": QLearningAgent,
                        "sarsa": SarsaAgent,
                        "sarsa_lambda": SarsaLambdaAgent,
                        "double_q_learner": DoubleQLearner}

        for agent_type in types:
            self.engine.agent_type = agent_type
            self.engine.create_world()
            assert isinstance(self.engine.agent, types_dict[agent_type])
            assert isinstance(self.engine.env, StrategicPricingMDP)

    def test_create_agent(self):
        types = ["q_learner", "sarsa", "sarsa_lambda", "double_q_learner"]
        types_dict = {"q_learner": QLearningAgent,
                      "sarsa": SarsaAgent,
                      "sarsa_lambda": SarsaLambdaAgent,
                      "double_q_learner": DoubleQLearner}

        for agent_type in types:
            self.engine.agent_type = agent_type
            self.engine.env = StrategicPricingMDP(self.dh, self.engine.bins)
            self.engine.create_agent()
            assert isinstance(self.engine.agent, types_dict[agent_type])

    def test_create_env(self):
        types = ["strategic_pricing"]
        types_dict = {"strategic_pricing": StrategicPricingMDP}

        for env_type in types:
            self.engine.env_type = env_type
            self.engine.create_env()
            assert isinstance(self.engine.env, types_dict[env_type])

    def test_train_agent(self):
        self.engine.create_world()
        original_q = self.engine.agent.Q
        self.engine.train_agent()
        assert self.engine.agent.q_table is not None
        assert self.engine.agent.q_table is not original_q


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



