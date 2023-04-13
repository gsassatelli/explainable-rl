import unittest
from src.explainability.pdp import PDP
from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler


class TestPDP(unittest.TestCase):
    """Test PDP class.
    """

    def setUp(self):
        # """Setup PDP class.
        # """
        # states = ['competitorPrice', 'adFlag', 'availability']
        # actions = ['price']
        # rewards = ['revenue']
        # n_samples = 50
        # self.dh = DataHandler('../../kaggle-dummy-dataset/train.csv', states, actions, rewards, n_samples=n_samples)
        # self.dh.prepare_data_for_engine(col_delimiter='|', cols_to_normalise=states + actions)
        # self.engine = Engine(self.dh, "q_learner", "kaggle", num_episodes=5, num_steps=1)
        # self.engine.create_world()
        # self.engine.train_agent()
        # self.pdp = PDP(bins=self.engine.env.bins,
        #           minmax_scalers=self.dh._minmax_scalars,
        #           action_labels=actions,
        #           state_labels=states)
        pass

    def test_create_pdp(self):
        """Test creation of PDP object.
        """
        # assert type(self.pdp) == PDP
        pass
    
    def test_get_digitized_pdp(self):
        """Test digitized pdp.
        """
        pass

    def test_get_denorm_actions(self):
        """Test denormalized actions.
        """
        pass

    def test_get_denorm_states(self):
        """Test denormalized states.
        """
        pass

    def test_build_data_for_plots(self):
        """Test build data for plots.
        """
        pass

if __name__ == '__main__':
    unittest.main()

