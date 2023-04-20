import unittest
import numpy as np
from src.explainability.shap_values import ShapValues
from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler


class TestShapValues(unittest.TestCase):
    """Test ShapValues class.
    """

    dh = None
    shap_values = None
    engine = None

    @classmethod
    def setUpClass(cls):
        """Setup PDP class.
        """
        states = ['competitorPrice', 'adFlag', 'availability']
        actions = ['price']
        rewards = ['revenue']
        n_samples = 100
        cls.dh = DataHandler('tests/test_env_data.csv', states, actions, rewards, n_samples=n_samples)
        cls.dh.prepare_data_for_engine(col_delimiter=',', cols_to_normalise=states + actions)
        cls.engine = Engine(cls.dh,
                            agent_type="q_learner",
                            env_type="strategic_pricing_predict",
                            bins=[10, 10, 10, 10],
                            num_episodes=5000,
                            num_steps=1)
        cls.engine.create_world()
        cls.engine.train_agent()
        cls.shap_values = ShapValues(sample=[9, 1, 1], features=states, env=cls.engine.env,
                                     Q=cls.engine.agent.Q, minmax_scalars=cls.dh.minmax_scalars, action=actions,
                                     number_of_samples=10)

    def test_create_shap_values(self):
        """Test creation of ShapValues object.
        """
        assert isinstance(self.shap_values, ShapValues)

    # TODO not possible to add test since the sample may not be visited by the agent (ValueError)
    # def test_compute_shap_values(self):
    #     """Test compute_shap_values method.
    #     """
    #     shaps, predicted_action = self.shap_values.compute_shap_values()
    #
    #     assert isinstance(shaps, dict)
    #     assert isinstance(predicted_action, float)
    #     assert len(shaps) == len(self.shap_values.features)

    def test_verify_sample_length(self):
        """Test verify_sample_length method.
        """
        result = self.shap_values.verify_sample_length()
        assert isinstance(result, bool)

    def test_bin_sample(self):
        """Test bin_sample method.
        """
        result = self.shap_values.bin_sample()
        assert isinstance(result, list)
        assert len(result) == len(self.shap_values.sample)

    def test_verify_cell_availability(self):
        """Test verify_cell_availability method.
        """
        binned_sample = [0, 0, 0]
        result = self.shap_values.verify_cell_availability(binned_sample)
        assert isinstance(result, bool)

    def test_sample_plus_minus_samples(self):
        """Test sample_plus_minus_samples method.
        """
        self.shap_values.sample = self.shap_values.normalize_sample()
        self.shap_values.binned_sample = self.shap_values.bin_sample()
        shap_ft = 0
        num_bins_per_shap_ft = 10
        result_plus, result_minus = self.shap_values.sample_plus_minus_samples(shap_ft, num_bins_per_shap_ft)
        assert isinstance(result_plus, list)
        assert isinstance(result_minus, list)
        assert len(result_plus) == len(result_minus)
        assert len(result_plus) == len(self.shap_values.sample)

    def test_get_denorm_actions(self):
        """Test get_denorm_actions method.
        """
        actions = np.array([0, 0, 0, 0, 0])
        result = self.shap_values.get_denorm_actions(actions)
        assert isinstance(result, list)
        assert len(result) == len(actions)

    def test_normalize_sample(self):
        """Test normalize_sample method.
        """
        result = self.shap_values.normalize_sample()
        assert isinstance(result, list)
        assert len(result) == len(self.shap_values.sample)

    def test_predict_action(self):
        """Test predict_action method.
        """
        self.shap_values.sample = self.shap_values.normalize_sample()
        self.shap_values.binned_sample = self.shap_values.bin_sample()
        result = self.shap_values.predict_action()
        assert isinstance(result, list)
        assert isinstance(result[0], float)

