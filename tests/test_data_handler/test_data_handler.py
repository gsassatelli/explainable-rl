from explainable_RL.foundation.library import *

# Import functions
from explainable_RL.data_handler.data_handler import DataHandler
from sklearn.preprocessing import MinMaxScaler
from tests.test_hyperparams import hyperparam_dict


class TestDataHandler(unittest.TestCase):
    """Test DataHandler class."""

    dh = None

    def setUp(self) -> None:
        """Set up test fixtures, if any."""
        dataset = pd.read_csv(hyperparam_dict["dataset"]["data_path"])
        self.dh = DataHandler(
            hyperparam_dict=hyperparam_dict, dataset=dataset, test_dataset=dataset
        )
        self.target = pd.read_csv("tests/test_env_data.csv").dropna()

    def tearDown(self) -> None:
        """Tear down test fixtures, if any."""
        del self.dh
        del self.target

    def test_type_get_actions(self):
        """Test get_actions method."""
        result = self.dh.get_actions()
        assert isinstance(result, pd.DataFrame)

    def test_len_get_actions(self):
        """Test get_actions method."""
        result = self.dh.get_actions()
        assert len(result) != 0

    def test_type_get_action_labels(self):
        """Test get_action_labels method."""
        result = self.dh.get_action_labels()
        assert isinstance(result, list)

    def test_len_get_action_labels(self):
        """Test get_action_labels method."""
        result = self.dh.get_action_labels()
        assert len(result) != 0

    def test_type_get_rewards(self):
        """Test get_rewards method."""
        result = self.dh.get_rewards()
        assert isinstance(result, pd.DataFrame)

    def test_len_get_rewards(self):
        """Test get_rewards method."""
        result = self.dh.get_rewards()
        assert len(result) != 0

    def test_type_get_states(self):
        """Test get_states method."""
        result = self.dh.get_states()
        assert isinstance(result, pd.DataFrame)

    def test_len_get_states(self):
        """Test get_states method."""
        result = self.dh.get_states()
        assert len(result) != 0

    def test_filter_data(self):
        """Test filter_data method."""
        self.dh._filter_data()
        assert self.dh.dataset.isnull().values.any() == False

    def test_transform_col(self):
        """Test transform_col method."""
        col_name = "price"
        scalar = MinMaxScaler()
        target = self.target
        target.loc[:, col_name] = scalar.fit_transform(
            pd.DataFrame(self.target[col_name])
        )
        target = target.round(decimals=2).astype("float64")[col_name]
        self.dh._fit_standard_scalars()
        self.dh._transform_col(col_name="price")
        result = (
            self.dh.dataset.sort_index().round(decimals=2).astype("float64")[col_name]
        )
        assert target.equals(result)

    def test_inverse_transform_col(self):
        """Test inverse_transform_col method."""
        target = self.target["price"].astype("float64")
        self.dh._inverse_transform_col(col_name="price")
        result = (
            self.dh.dataset["price"].sort_index().round(decimals=2).astype("float64")
        )
        assert target.equals(result)

    def test_fit_standard_scalars(self):
        """Test fit_standard_scalars method."""
        self.dh._fit_standard_scalars()
        assert len(self.dh.minmax_scalars) == 12

    def test_prepare_data_for_engine(self):
        """Test prepare_data_for_engine method."""
        self.dh.prepare_data_for_engine(
            cols_to_normalise=["competitorPrice", "adFlag", "availability", "price"]
        )
        target = self.target[["competitorPrice", "adFlag", "availability", "price"]]
        for col in target.columns:
            scalar = MinMaxScaler()
            target[col] = scalar.fit_transform(pd.DataFrame(target[col]))
        assert (
            self.dh.dataset[["competitorPrice", "adFlag", "availability", "price"]]
            .sort_index()
            .equals(target.sort_index())
        )
        assert len(self.dh.dataset) == 48
        assert len(self.dh.dataset.columns) == 12

    def test_reverse_norm(self):
        """Test reverse_norm method."""
        target = self.target.round(decimals=2).astype("float64")
        self.dh.reverse_norm()
        result = self.dh.dataset.round(decimals=2).astype("float64").sort_index()
        assert result.equals(target)
