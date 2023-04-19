import unittest
from src.explainability.pdp import PDP
from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler


class TestPDP(unittest.TestCase):
    """Test PDP class.
    """

    dh = None
    pdp = None
    engine = None

    @classmethod
    def setUpClass(cls):
        """Setup PDP class.
        """
        states = ['competitorPrice', 'adFlag', 'availability']
        actions = ['price']
        rewards = ['revenue']
        n_samples = 50
        cls.dh = DataHandler('tests/test_env_data.csv', states, actions, rewards, n_samples=n_samples)
        cls.dh.prepare_data_for_engine(col_delimiter=',', cols_to_normalise=states + actions)
        cls.engine = Engine(cls.dh,
                            agent_type="q_learner",
                            env_type="strategic_pricing",
                            bins=[10, 10, 10, 10],
                            num_episodes=100,
                            num_steps=1)
        cls.engine.create_world()
        cls.engine.train_agent()
        cls.pdp = PDP(bins=cls.engine.env.bins,
                      minmax_scalers=cls.dh.minmax_scalars,
                      action_labels=actions,
                      state_labels=states)
        cls.pdp.build_data_for_plots(cls.engine.agent.Q,
                                     cls.engine.agent.Q_num_samples)

    def test_create_pdp(self):
        """Test creation of PDP object.
        """
        assert isinstance(self.pdp, PDP)

    def test_get_digitized_pdp(self):
        """Test digitized pdp.
        """
        assert isinstance(self.pdp._dig_state_actions, list)
        assert isinstance(self.pdp._dig_state_actions_std, list)
        assert isinstance(self.pdp._dig_state_actions_samples, list)
        assert len(self.pdp._dig_state_actions) == len(self.dh._state_labels) - 1
        assert len(self.pdp._dig_state_actions_std) == len(self.dh._state_labels) - 1
        assert len(self.pdp._dig_state_actions_samples) == len(self.dh._state_labels) - 1

    def test_get_denorm_actions(self):
        """Test denormalized actions.
        """
        assert isinstance(self.pdp._denorm_actions, list)

    def test_get_denorm_states(self):
        """Test denormalized states.
        """
        assert isinstance(self.pdp._denorm_states, list)

    def test_build_data_for_plots(self):
        """Test build data for plots.
        """
        pass

    def test_pdp_plot(self):
        """Test pdp plot.
        """
        pass
