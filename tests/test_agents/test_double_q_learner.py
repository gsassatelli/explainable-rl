# Import functions
from tests.test_agents.test_td import TestTD
from src.environments.strategic_pricing_suggestion import StrategicPricingSuggestionMDP
from src.environments.strategic_pricing_prediction import StrategicPricingPredictionMDP
from src.agents.double_q_learner import DoubleQLearner

# Import packages
import copy
import numpy as np


class TestDoubleQLearner(TestTD):
    """Test the DoubleQLearner class."""

    def setUp(self) -> None:
        """Set up the test class."""
        self.env = StrategicPricingSuggestionMDP(self.dh)
        self.agent = DoubleQLearner(self.env, gamma=0.9)

    def test_update_q_values(self):
        """Test the update_q_values method."""
        self.agent.create_tables()
        self.agent.Q_a[0, 0, 0, 2] = 1.5
        self.agent.Q_a[3, 0, 0, 3] = 5
        state = [0, 0, 0]
        action = 2
        epsilon = 0.5
        next_state = [3, 0, 0]
        reward = 10
        lr = 0.1

        self.agent._update_q_values(state=state,
                                    action=action,
                                    next_state=next_state,
                                    reward=reward,
                                    epsilon=epsilon,
                                    lr=lr,
                                    Q_a=self.agent.Q_a,
                                    Q_b=self.agent.Q_b)
        result = self.agent.Q_a[0, 0, 0, 2]
        target = [1.5 + lr * (10 + 0.9 * 5 - 1.5), 1.5 + lr * (10 + 0.9 * 0 - 1.5)]
        assert result in target

    def test_step(self):
        """Test the step method."""
        epsilon = 0  # epsilon = 0 as this functionality is tested above.
        lr = 0.1
        self.agent.create_tables()
        self.agent.Q_a[0, 0, 0, :] = 1.5
        self.agent.state = [0, 0, 0]
        self.agent._step(epsilon, lr)

        assert self.agent.state == [0, 0, 0]
        assert type(self.agent.Q_a[0, 0, 0, 0]) is np.float64

    def test_fit(self):
        """Test the fit method."""
        self.agent.create_tables()
        original_Q = copy.deepcopy(self.agent.Q)
        self.agent.fit(n_episodes=1, n_steps=1)
        assert self.agent.Q.shape == original_Q.shape
        assert self.agent.Q != original_Q