from library import *

# Import functions
from src.explainability.pdp import PDP
from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler
from tests.test_hyperparams import hyperparam_dict


class TestPDP(unittest.TestCase):
    """Test PDP class.
    """

    dh = None
    pdp = None
    engine = None

    @classmethod
    def setUpClass(cls):
        """Setup TestPDP class.
        """
        dataset = pd.read_csv(hyperparam_dict['dataset']['data_path'],
                              sep=hyperparam_dict['dataset']['col_delimiter'])
        cls.dh = DataHandler(hyperparam_dict=hyperparam_dict, dataset=dataset, test_dataset=dataset)
        cls.engine = Engine(dh=cls.dh,
                            hyperparam_dict=hyperparam_dict)
        cls.engine.create_world()
        cls.engine.train_agent()
        cls.pdp = PDP(engine=cls.engine)
        cls.pdp.build_data_for_plots()

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
        assert len(self.pdp._dig_state_actions) == len(self.dh.state_labels)
        assert len(self.pdp._dig_state_actions_std) == len(self.dh.state_labels)
        assert len(self.pdp._dig_state_actions_samples) == len(self.dh.state_labels)

    def test_get_denorm_actions(self):
        """Test denormalized actions.
        """
        assert isinstance(self.pdp._denorm_actions, list)

    def test_get_denorm_states(self):
        """Test denormalized states.
        """
        assert isinstance(self.pdp._denorm_states, list)
