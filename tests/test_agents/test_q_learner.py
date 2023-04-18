from tests.test_agents.test_td import TestTD
import copy
from src.environments.strategic_pricing import StrategicPricingMDP
from src.agents.q_learner import QLearningAgent

class TestQLearningAgent(TestTD):

    def setUp(self) -> None:
        self.env = StrategicPricingMDP(self.dh)
        self.agent = QLearningAgent(self.env, gamma=0.9)


    def test_update_q_values(self):
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
        epsilon = 0
        lr = 0.1
        self.agent.create_tables()
        self.agent.Q[0, 0, 0, 2] = 1.5
        self.agent.state = [0, 0, 0]

        self.agent._step(epsilon, lr)

        assert self.agent.state == [0, 0, 0]
        assert self.agent.Q[0, 0, 0, 2] == 1.5 + lr * (0 + 0.9 * 1.5 - 1.5)

    def test_fit(self):
        self.agent.create_tables()
        original_Q = copy.deepcopy(self.agent.Q)
        self.agent.fit(n_episodes=10, n_steps=1)
        assert self.agent.Q.shape == original_Q.shape
        assert self.agent.Q != original_Q
