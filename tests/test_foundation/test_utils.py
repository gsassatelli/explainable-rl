# Import functions
from src.foundation.utils import *


class TestUtils(unittest.TestCase):
    """Test the utils functions."""

    def test_convert_to_string(self):
        """Test convert_to_string function.
        """
        state = [1, 2, 3]
        state_str = convert_to_string(state)
        assert state_str == "1,2,3"

    def test_convert_to_list(self):
        """Test convert_to_list function.
        """
        state_str = "1,2,3"
        state = convert_to_list(state_str)
        assert state == [1, 2, 3]

    def test_decay_param(self):
        """Test decay_param function.
        """
        param = 1
        decay = 0.1
        min_param = 0.1
        param = decay_param(param, decay, min_param)
        assert param == 0.9

    def test_load_dataset(self):
        """Test load_dataset function.
        """
        dataset_50 = load_data(data_path="tests/test_env_data.csv", n_samples=50)
        dataset_25 = load_data(data_path="tests/test_env_data.csv", n_samples=25)
        assert len(dataset_50) == 50
        assert len(dataset_25) == 25
        assert isinstance(dataset_50, pd.DataFrame)
        assert isinstance(dataset_25, pd.DataFrame)

    def test_split_train_test(self):
        """Test split_train_test function.
        """
        dataset = load_data(data_path="tests/test_env_data.csv", n_samples=50)
        train_dataset, test_dataset = split_train_test(dataset, train_test_split=0.2)
        assert len(train_dataset) == 40
        assert len(test_dataset) == 10
        assert isinstance(train_dataset, pd.DataFrame)
        assert isinstance(test_dataset, pd.DataFrame)
