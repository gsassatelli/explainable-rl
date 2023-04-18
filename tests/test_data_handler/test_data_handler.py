# import ipdb 
import numpy as np
from src.data_handler.data_handler import DataHandler
import unittest 
import pandas as pd

# Load data
states = ['competitorPrice', 'adFlag', 'availability']
actions = ['price']
rewards = ['revenue']
n_samples = 2000
dh = DataHandler('tests/test_env_data.csv', states, actions, rewards, n_samples=n_samples)
dh.prepare_data_for_engine(col_delimiter=',', cols_to_normalise=states+actions)

def test_load_data():
    dh.load_data(delimiter=',')
    assert type(dh.dataset) == pd.core.frame.DataFrame

def test_type_get_actions():
    result = dh.get_actions()
    assert isinstance(result, list)

def test_len_get_actions():
    result = dh.get_actions()
    assert len(result) != 0

def test_type_get_rewards():
    result = dh.get_rewards()
    assert type(result) == pd.core.frame.DataFrame

def test_len_get_rewards():
    result = dh.get_rewards()
    assert len(result) != 0

def test_type_get_states():
    result = dh.get_states()
    assert type(result) == pd.core.frame.DataFrame

def test_len_get_states():
    result = dh.get_states()
    assert len(result) != 0

def test_filter_data():
    dh._filter_data()
    assert dh.dataset.isnull().values.any() == False

if __name__ == '__main__':
    unittest.main()
