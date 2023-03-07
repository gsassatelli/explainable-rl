# import ipdb 
import numpy as np
from src.foundation.environment import MDP
from src.data_handler.data_handler import DataHandler
from src.foundation.engine import Engine
import unittest 
import sparse


# Load data
states = ['competitorPrice', 'adFlag', 'availability']
actions = ['price']
rewards = ['revenue']
n_samples = 200
dh = DataHandler('kaggle-dummy-dataset/train.csv', states, actions, rewards, n_samples=n_samples)

# Preprocess the data
dh.prepare_data_for_engine(col_delimiter='|', cols_to_normalise=states+actions)

# Create engine
engine = Engine(dh, "q_learner", "kaggle", num_episodes=100, num_steps=10)

# Create world
engine.create_world()

env = MDP(dh)

def test_state_transform_df_to_numpy():
    env._transform_df_to_numpy()
    result = env.dh.get_states().to_numpy()
    assert type(result) == np.ndarray

def test_action_transform_df_to_numpy():
    env._transform_df_to_numpy()
    result = env.dh.get_actions().to_numpy()
    assert type(result) == np.ndarray

def test_reward_transform_df_to_numpy():
    env._transform_df_to_numpy()
    result = env.dh.get_rewards().to_numpy()
    assert type(result) == np.ndarray

def test_join_state_action():
    env._reward_mdp_data = np.array([0,0,0,0])
    env._state_mdp_data = np.array([1,2,3,0])
    env._action_mdp_data = np.array([0,0,1,1])
    result = env._join_state_action()
    expected = [1, 2, 4, 1]
    assert result == expected

def test_bin_state_action_space():
    zipped = env._join_state_action()
    result = env._bin_state_action_space(zipped).tolist()
    expected = [100, 101, 101, 100]
    assert expected == result

def test_get_counts_and_rewards_per_bin():
    zipped = env._join_state_action()
    binned = env._bin_state_action_space(zipped)
    result = env._get_counts_and_rewards_per_bin(binned)
    assert type(result) == dict

def test_make_rewards_from_data():
    result = env._make_rewards_from_data()
    assert type(result) == sparse._coo.core.COO

def test_type_reset():
    result = env.reset()
    assert type(result) == np.ndarray

def test_args_reset():
    result = env.reset()
    assert len(result) == 3

def test_step():
    reward_matrix = env._make_rewards_from_data()
    state = [1,0,1]
    action = [1]
    result = reward_matrix[state[0], state[1], state[2], action]
    assert type(result) == sparse._coo.core.COO

if __name__ == '__main__':
    unittest.main()
