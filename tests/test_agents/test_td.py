import unittest
from src.agents.td import TD
from src.environments.strategic_pricing import StrategicPricingMDP
from src.data_handler.data_handler import DataHandler


class TestTD(unittest.TestCase):
    dh = None

    @classmethod
    def setUpClass(cls) -> None:
        states = ['competitorPrice', 'adFlag', 'availability', 'price'],
        actions = [price_bin/10 for price_bin in range(1, 11)]
        rewards = ['revenue']
        n_samples = 50
        cls.dh = DataHandler('tests/test_env_data.csv', states, actions, rewards,
                              n_samples=n_samples)
        cls.dh.prepare_data_for_engine(col_delimiter=',',
                                       cols_to_normalise=states)

    def setUp(self) -> None:
        self.env = StrategicPricingMDP(self.dh)
        self.agent = TD(self.env, gamma=0.9)

    def tearDown(self) -> None:
        del self.agent

    def test_update_q_values(self):
        """Implemented in tests for subclasses."""
        pass

    def test_step(self):
        """Implemented in tests for subclasses."""
        pass

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
        """Implemented in tests for subclasses."""
        pass

