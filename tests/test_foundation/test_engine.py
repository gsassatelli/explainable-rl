import unittest
from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler
from src.foundation.environment import MDP
from src.foundation.agent import Agent


class TestEngine(unittest.TestCase):

    dh = None

    @classmethod
    def setUpClass(cls):
        states = ['competitorPrice', 'adFlag', 'availability']
        actions = ['price']
        rewards = ['revenue']
        n_samples = 200
        cls.dh = DataHandler('tests/test_env_data.csv', states, actions, rewards, n_samples=n_samples)
        cls.dh.prepare_data_for_engine(col_delimiter=',', cols_to_normalise=states + actions)

    def setUp(self):
        self.engine = Engine(dh=self.dh,
                             agent_type="q_learner",
                             env_type="kaggle",
                             bins=[10, 10, 10, 10],
                             num_episodes=100,
                             num_steps=10)
        self.engine.create_world()

    def tearDown(self) -> None:
        del self.engine

    def test_env_type(self):
        assert isinstance(self.engine.env, MDP)

    def test_agent_type(self):
        assert isinstance(self.engine.agent, Agent)


