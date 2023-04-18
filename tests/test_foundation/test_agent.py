import unittest
from src.foundation.agent import QLearningAgent
from src.foundation.environment import StrategicPricingMDP
from src.data_handler.data_handler import DataHandler
import copy

class TestQLearningAgent(unittest.TestCase):

    dh = None

    @classmethod
    def setUpClass(cls) -> None:
        states = ['competitorPrice', 'adFlag', 'availability']
        actions = ['price']
        rewards = ['revenue']
        n_samples = 50
        cls.dh = DataHandler('tests/test_env_data.csv', states, actions, rewards,
                              n_samples=n_samples)
        cls.dh.prepare_data_for_engine(col_delimiter=',',
                                        cols_to_normalise=states+actions)

    def setUp(self) -> None:
        self.env = StrategicPricingMDP(self.dh)
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
        epsilon = 0
        lr = 0.1
        self.agent.create_tables()
        self.agent.Q[0, 0, 0, 2] = 1.5
        self.agent.state = [0, 0, 0]

        self.agent._step(epsilon, lr)

        assert self.agent.state == [0, 0, 0]
        assert self.agent.Q[0, 0, 0, 2] == 1.5 + lr * (0 + 0.9 * 1.5 - 1.5)


    def test_init_q_table(self):
        self.agent.env.bins = [10, 5, 4, 6]
        self.agent._init_q_table()
        assert self.agent.Q.shape == (10, 5, 4, 6)

    def test_convert_to_string(self):
        state = [0, 5, 3, 2]
        result = self.agent._convert_to_string(state)
        target = '0,5,3,2'
        assert result == target

    def test_epsilon_greedy_policy(self):
        epsilon = 0
        state = [0, 0, 0]
        self.agent._init_q_table()
        self.agent.Q[0, 0, 0, 2] = 1.5
        self.agent.state = [0, 0, 0]
        result = self.agent._epsilon_greedy_policy(state=state, epsilon=epsilon)
        assert result == 2

    def test_create_tables(self):
        self.agent.env.bins = [10, 5, 4, 6]
        self.agent.create_tables()
        assert self.agent.Q.shape == (10, 5, 4, 6)
        assert self.agent.state_to_action is not None

    def test_fit(self):
        self.agent.create_tables()
        original_Q = copy.deepcopy(self.agent.Q)
        self.agent.fit(n_episodes=10, n_steps=1)
        assert self.agent.Q.shape == original_Q.shape
        assert self.agent.Q != original_Q

