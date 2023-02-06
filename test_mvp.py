from foundation.engine import Engine
from data_handler.data_handler import DataHandler


if __name__ == "__main__":
    # Load data
    states = ['competitorPrice', 'adFlag', 'availability']
    actions = ['price']
    rewards = ['revenue']
    dh = DataHandler('dataset_kaggle/train.csv', states, actions, rewards)
    dh.prepare_data_for_engine(col_delimiter='|', cols_to_normalise=states+actions)
    # Create engine
    engine = Engine(dh.mdp_data[:100], "q_learner", "kaggle", 
                    num_episodes = 100, num_steps = 10)
    
    # Create world
    engine.create_world()
    
    # Train agent
    engine.train_agent()
    
    breakpoint()
