import ipdb 
from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler
from src.explainability.pdp import PDP


def test_q_table():
    """Test shape of Q table.
    """
    # Load data
    states = ['competitorPrice', 'adFlag', 'availability']
    actions = ['price']
    rewards = ['revenue']
    n_samples = 2000
    dh = DataHandler('kaggle-dummy-dataset/train.csv', states, actions, rewards, n_samples=n_samples)

    # Preprocess the data
    dh.prepare_data_for_engine(col_delimiter='|', cols_to_normalise=states+actions)

    # Create engine
    engine = Engine(dh, "q_learner", "kaggle", num_episodes=100, num_steps=10)
    
    # Create world
    engine.create_world()
    
    # Agent
    agent = engine.agent
    
    # Create tables
    agent.create_tables()
    
    # Assert Q-table has right dimensions
    Q = agent.Q
    
    # state dim
    n_bins = engine.env.num_bins
    state_dim = engine.env.state_dim
    
    assert Q.shape == tuple([n_bins+1]*(state_dim+1))
    
    
