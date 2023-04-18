from src.data_handler.data_handler import DataHandler
import unittest
import pandas as pd


class TestDataHandler(unittest.TestCase):

    dh = None

    @classmethod
    def setUpClass(cls) -> None:
        states = ['competitorPrice', 'adFlag', 'availability']
        actions = ['price']
        rewards = ['revenue']
        n_samples = 200
        cls.dh = DataHandler('tests/test_env_data.csv', states, actions, rewards, n_samples=n_samples)
        cls.dh.prepare_data_for_engine(col_delimiter=',', cols_to_normalise=states + actions)

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_load_data(self):
        self.dh.load_data(delimiter=',')
        assert isinstance(self.dh.dataset, pd.DataFrame)

    def test_type_get_actions(self):
        result = self.dh.get_actions()
        assert isinstance(result, list)

    def test_len_get_actions(self):
        result = self.dh.get_actions()
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
