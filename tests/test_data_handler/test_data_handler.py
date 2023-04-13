import unittest
from src.data_handler.data_handler import DataHandler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class TestDataHandler(unittest.TestCase):

    def setUp(self):
        states = ['competitorPrice', 'adFlag', 'availability']
        actions = ['price']
        rewards = ['revenue']
        n_samples = 50
        self.dh = DataHandler('tests/test_env_data.csv', states, actions, rewards,
                              n_samples=n_samples)
        self.target = pd.read_csv('tests/test_env_data.csv').dropna()
        self.dh.load_data(delimiter=',')
        self.dh.preprocess_data(normalisation=False)

    def tearDown(self) -> None:
        del self.dh
        del self.target

    def test_get_actions(self):
        target = pd.DataFrame(self.target['price']).astype('float64').\
                    round(decimals=2)
        result = self.dh.get_actions().sort_index()
        assert target.equals(result)

    def test_get_rewards(self):
        target = pd.DataFrame(self.target['revenue']).astype('float64').\
                    round(decimals=2)
        result = self.dh.get_rewards().sort_index()
        assert target.equals(result)

    def test_get_states(self):
        target = self.target[['competitorPrice', 'adFlag', 'availability']]
        result = self.dh.get_states().sort_index()
        assert target.equals(result)

    def test_filter_data(self):
        self.dh._filter_data()
        assert self.dh.dataset.sort_index().equals(self.target)

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

    def test_load_data(self):
        assert type(self.dh.dataset) == pd.core.frame.DataFrame
        assert len(self.dh.dataset) == 48

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
