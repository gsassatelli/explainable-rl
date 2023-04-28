from explainable_rl.foundation.library import *
from tests.test_agents.test_td import TestTD
from explainable_rl.environments.strategic_pricing_prediction import StrategicPricingPredictionMDP
from explainable_rl.agents.sarsa import SarsaAgent


class TestSarsa(TestTD):
    """Test the SarsaAgent class."""

    def setUp(self) -> None:
        """Set up the test class."""
        self.env = StrategicPricingPredictionMDP(self.dh)
        self.agent = SarsaAgent(self.env, gamma=0.9)

    def test_update_q_values(self):
        """Test the update_q_values method."""
        self.agent.create_tables()
        self.agent.Q[0, 0, 0, 2] = 1.5
        self.agent.Q[3, 0, 0, 3] = 5
        state = [0, 0, 0]
        action = 2
        epsilon = 0.5
        next_state = [3, 0, 0]
        reward = 10
        lr = 0.1

        self.agent._update_q_values(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            epsilon=epsilon,
            lr=lr,
        )
        result = self.agent.Q[0, 0, 0, 2]
        target = [1.5 + lr * (10 + 0.9 * 5 - 1.5), 1.5 + lr * (10 + 0.9 * 0 - 1.5)]
        assert result in target

    def test_step(self):
        """Test the step method."""
        epsilon = 0  # epsilon = 0 as this functionality is tested above.
        lr = 0.1
        self.agent.create_tables()
        self.agent.Q[0, 0, 0, 0] = 1.5
        self.agent.state = [0, 0, 0]
        self.agent._step(epsilon, lr, use_uncertainty=False)

        assert self.agent.state == [0, 0, 0]
        assert type(self.agent.Q[0, 0, 0, 0]) is np.float64

    def test_fit(self):
        """Test the fit method."""
        self.agent.create_tables()
        original_Q = copy.deepcopy(self.agent.Q)
        self.agent.fit(
            agent_hyperparams=self.dh.hyperparam_dict["agent"],
            training_hyperparams=self.dh.hyperparam_dict["training"],
        )
        assert self.agent.Q.shape == original_Q.shape
        assert self.agent.Q != original_Q
