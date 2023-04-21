from library import *

# Import functions
from src.explainability.shap_values import ShapValues
from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler
from tests.test_hyperparams import hyperparam_dict

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
        dataset = pd.read_csv(hyperparam_dict['dataset']['data_path'], sep=hyperparam_dict['dataset']['col_delimiter'])
        cls.dh = DataHandler(hyperparam_dict=hyperparam_dict, dataset=dataset)
        cls.dh.prepare_data_for_engine()
        cls.engine = Engine(cls.dh,
                            hyperparam_dict=hyperparam_dict)
        cls.engine.create_world()
        cls.engine.train_agent()
        cls.shap_values = ShapValues(sample=[9, 1, 1],
                                     engine=cls.engine)

    def test_create_shap_values(self):
        """Test creation of ShapValues object.
        """
        assert isinstance(self.shap_values, ShapValues)

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

    def test_verify_outliers(self):
        """Test verify_outliers method.
        """
        binned_sample_correct = [0, 0, 0]
        binned_sample_wrong = [0, 0, 20]
        result_correct = self.shap_values.verify_outliers(binned_sample_correct)
        result_wrong = self.shap_values.verify_outliers(binned_sample_wrong)
        assert isinstance(result_correct, bool)
        assert isinstance(result_wrong, bool)
        assert result_correct == False
        assert result_wrong == True
