import unittest
from src.foundation.agent import QLearningAgent
from src.foundation.environment import StrategicPricingMDP
from src.data_handler.data_handler import DataHandler

class TestQLearningAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        states = ['competitorPrice', 'adFlag', 'availability']
        actions = ['price']
        rewards = ['revenue']
        n_samples = 50
        cls.dh = DataHandler('../test_env_data.csv', states, actions, rewards,
                              n_samples=n_samples)
        cls.dh.prepare_data_for_engine(col_delimiter=',',
                                        cols_to_normalise=states+actions)
        cls.env = StrategicPricingMDP(cls.dh)

    def setUp(self) -> None:
        self.agent = QLearningAgent(self.env, gamma=0.9)

    def tearDown(self) -> None:
        del self.agent

    def test_update_q_values(self):
        self.agent._init_q_table()
        self.agent.Q[0, 0, 0, 2] = 1.5
        self.agent.Q[0, 2, 0, 3] = 5
        state = [0, 0, 0]
        action = 2
        next_state = [0, 2, 0]
        reward = 10
        lr = 0.1

        self.agent._update_q_values(state, action, next_state, reward, lr)
        result = self.agent.Q[0, 0, 0, 2]
        target = 1.5 + lr * (10 + 0.9 * 5 - 1.5)
        assert result == target


    def test_step(self):
        pass

    def test_init_q_table(self):
        pass

    def test_convert_to_string(self):
        pass

    def test_epsilon_greedy_policy(self):
        pass

    def test_create_tables(self):
        pass

    def test_fit(self):
        pass

