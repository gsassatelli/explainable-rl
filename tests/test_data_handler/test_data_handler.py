from src.data_handler.data_handler import DataHandler
import unittest
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class TestDataHandler(unittest.TestCase):

    dh = None

    def setUp(self) -> None:
        states = ['competitorPrice', 'adFlag', 'availability']
        actions = ['price']
        rewards = ['revenue']
        n_samples = 50
        self.dh = DataHandler('tests/test_env_data.csv', states, actions, rewards, n_samples=n_samples)
        self.dh.prepare_data_for_engine(col_delimiter=',', cols_to_normalise=states + actions)
        self.target = pd.read_csv('tests/test_env_data.csv').dropna()

    def tearDown(self) -> None:
        del self.dh
        del self.target

    def test_load_data(self):
        self.dh.load_data(delimiter=',')
        assert isinstance(self.dh.dataset, pd.DataFrame)

    def test_type_get_actions(self):
        result = self.dh.get_actions()
        assert isinstance(result, pd.DataFrame)

    def test_len_get_actions(self):
        result = self.dh.get_actions()
        assert len(result) != 0

    def test_type_get_action_labels(self):
        result = self.dh.get_action_labels()
        assert isinstance(result, list)

    def test_len_get_action_labels(self):
        result = self.dh.get_action_labels()
        assert len(result) != 0

    def test_type_get_rewards(self):
        result = self.dh.get_rewards()
        assert isinstance(result, pd.DataFrame)

    def test_len_get_rewards(self):
        result = self.dh.get_rewards()
        assert len(result) != 0

    def test_type_get_states(self):
        result = self.dh.get_states()
        assert isinstance(result, pd.DataFrame)

    def test_len_get_states(self):
        result = self.dh.get_states()
        assert len(result) != 0

    def test_filter_data(self):
        self.dh._filter_data()
        assert self.dh.dataset.isnull().values.any() == False
    
    def test_transform_col(self):
        col_name = 'price'
        scalar = MinMaxScaler()
        target = self.target
        target[col_name] = scalar.fit_transform(pd.DataFrame(self.target[col_name]))
        target = target.round(decimals=2).astype('float64')
        self.dh._fit_standard_scalars()
        self.dh._transform_col(col_name='price')
        result = self.dh.dataset.sort_index().round(decimals=2).astype('float64')
        assert target.equals(result)

    def test_inverse_transform_col(self):
        target = self.target['price'].astype('float64')
        self.dh._fit_standard_scalars()
        self.dh._transform_col(col_name='price')
        self.dh._inverse_transform_col(col_name='price')
        result = self.dh.dataset['price'].sort_index().round(
            decimals=2).astype('float64')
        assert target.equals(result)

    def test_fit_standard_scalars(self):
        self.dh._fit_standard_scalars()
        assert len(self.dh.minmax_scalars) == 12


    def test_prepare_data_for_engine(self):
        self.dh.prepare_data_for_engine(col_delimiter=',',
                                        cols_to_normalise=['competitorPrice',
                                                           'adFlag',
                                                           'availability',
                                                           'price'])
        target = self.target[
            ['competitorPrice', 'adFlag', 'availability', 'price']]
        for col in target.columns:
            scalar = MinMaxScaler()
            target[col] = scalar.fit_transform(pd.DataFrame(target[col]))
        assert self.dh.dataset[['competitorPrice', 'adFlag', 'availability',
                                'price']].sort_index(). \
            equals(target.sort_index())
        assert len(self.dh.dataset) == 48
        assert len(self.dh.dataset.columns) == 12

    def test_reverse_norm(self):
        target = self.target.round(decimals=2).astype('float64')
        self.dh.prepare_data_for_engine(col_delimiter=',',
                                        cols_to_normalise=['competitorPrice',
                                                           'adFlag',
                                                           'availability',
                                                           'price'])
        self.dh.reverse_norm()
        result = self.dh.dataset.round(decimals=2).astype('float64').sort_index()
        assert result.equals(target)