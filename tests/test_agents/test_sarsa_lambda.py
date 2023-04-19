from tests.test_agents.test_td import TestTD
from src.agents.sarsa_lambda import SarsaLambdaAgent
from src.environments.strategic_pricing_suggestion import StrategicPricingSuggestionMDP
from src.environments.strategic_pricing_prediction import StrategicPricingPredictionMDP


class TestSarsaLambda(TestTD):

    def setUp(self) -> None:
        self.env = StrategicPricingMDP(self.dh)
        self.agent = SarsaLambdaAgent(self.env, gamma=0.9, lambda_=0.9)

    def test_update_q_values(self):
        self.agent.create_tables()
        self.agent.Q[0, 0, 0, 2] = 1.5
        self.agent.Q[3, 0, 0, 3] = 5
        state = [0, 0, 0]
        action = 2
        epsilon = 0.5
        next_state = [3, 0, 0]
        reward = 10
        lr = 0.1
        self.agent.e[0, 0, 0, 2] = 1

        self.agent._update_q_values(state=state,
                                    action=action,
                                    next_state=next_state,
                                    reward=reward,
                                    epsilon=epsilon,
                                    lr=lr)
        result_Q = self.agent.Q[0, 0, 0, 2]
        result_e = self.agent.e[0, 0, 0, 2]
        target_Q = [1.5 + lr * (10 + 0.9 * 5 - 1.5) * 2,
                  1.5 + lr * (10 + 0.9 * 0 - 1.5) * 2]
        target_e = 0.9 * 0.9 * 2
        assert result_Q in target_Q
        assert result_e == target_e