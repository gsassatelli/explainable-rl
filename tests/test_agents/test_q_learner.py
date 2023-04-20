# Import functions
from tests.test_agents.test_td import TestTD
from src.environments.strategic_pricing_suggestion import StrategicPricingSuggestionMDP
from src.environments.strategic_pricing_prediction import StrategicPricingPredictionMDP
from src.agents.q_learner import QLearningAgent

# Import packages
import copy


class TestQLearningAgent(TestTD):
    """Test the QLearningAgent class."""

    def setUp(self) -> None:
        """Set up the test class."""
        self.env = StrategicPricingSuggestionMDP(self.dh)
        self.agent = QLearningAgent(self.env, gamma=0.9)


    def test_update_q_values(self):
        """Test the update_q_values method."""
        self.agent._init_q_table()
        self.agent.Q[0, 0, 0, 2] = 1.5
        self.agent.Q[0, 2, 0, 3] = 5
        state = [0, 0, 0]
        action = 2
        epsilon = 1
        next_state = [0, 2, 0]
        reward = 10
        lr = 0.1

        self.agent._update_q_values(state=state,
                                    action=action,
                                    next_state=next_state,
                                    reward=reward,
                                    epsilon=epsilon,
                                    lr=lr)
        result = self.agent.Q[0, 0, 0, 2]
        target = 1.5 + lr * (10 + 0.9 * 5 - 1.5)
        assert result == target

    def test_step(self):
        """Test the step method."""
        # TODO: from Giulia, can we remove the print?
        print("testing test_step")
        epsilon = 0
        lr = 0.1
        self.agent.create_tables()
        self.agent.Q[0, 0, 0, 2] = 1.5
        self.agent.state = [0, 0, 0]

        self.agent._step(epsilon, lr)

        print(f"self.agent.Q[0, 0, 0, 2]:{self.agent.Q[0, 0, 0, 2]}")

        assert self.agent.state == [0, 0, 0]
        assert self.agent.Q[0, 0, 0, 2] == 1.5 + lr * (0 + 0.9 * 1.5 - 1.5)

    def test_fit(self):
        """Test the fit method."""
        self.agent.create_tables()
        original_Q = copy.deepcopy(self.agent.Q)
        self.agent.fit(n_episodes=1, n_steps=1)
        assert self.agent.Q.shape == original_Q.shape
        assert self.agent.Q != original_Q

