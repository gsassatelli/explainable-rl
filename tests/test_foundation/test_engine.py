def test_engine():
    pass


# import ipdb
# from src.foundation.engine import Engine
# from src.data_handler.data_handler import DataHandler
# from src.explainability.pdp import PDP
# from src.foundation.environment import MDP
# from src.foundation.agent import QLearningAgent
#
#
# def test_create_env_class():
#     """ Test creation of environment.
#     """
#
#     states = ['competitorPrice', 'adFlag', 'availability']
#     actions = ['price']
#     rewards = ['revenue']
#     n_samples = 2000
#     dh = DataHandler('kaggle-dummy-dataset/train.csv', states, actions, rewards, n_samples=n_samples)
#
#     # Preprocess the data
#     dh.prepare_data_for_engine(col_delimiter='|', cols_to_normalise=states+actions)
#
#     # Create engine
#     engine = Engine(dh, "q_learner", "kaggle", num_episodes=100, num_steps=10)
#
#     # Create world
#     engine.create_world()
#
#     # Assert it creates an environment of type MDP
#     assert type(engine.env) == MDP
#
#
# def test_create_agent_class():
#     """ Test creation of environment.
#     """
#
#     states = ['competitorPrice', 'adFlag', 'availability']
#     actions = ['price']
#     rewards = ['revenue']
#     n_samples = 2000
#     dh = DataHandler('kaggle-dummy-dataset/train.csv', states, actions, rewards, n_samples=n_samples)
#
#     # Preprocess the data
#     dh.prepare_data_for_engine(col_delimiter='|', cols_to_normalise=states+actions)
#
#     # Create engine
#     engine = Engine(dh, "q_learner", "kaggle", num_episodes=100, num_steps=10)
#
#     # Create world
#     engine.create_world()
#
#     # Assert it creates an environment of type MDP
#     assert type(engine.agent) == QLearningAgent
#
