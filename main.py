from foundation.engine import Engine
from data_handler.data_handler import DataHandler
from datetime import datetime
from explainability.pdp import PDP


if __name__ == "__main__":
    # Load data
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(timestamp + ": Load data")
    states = ['competitorPrice', 'adFlag', 'availability']
    actions = ['price']
    rewards = ['revenue']
    dh = DataHandler('kaggle-dummy-dataset/train.csv', states, actions, rewards)

    # Preprocess the data
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(timestamp + ": Preprocess data")
    dh.prepare_data_for_engine(col_delimiter='|', cols_to_normalise=states+actions)

    # Create engine
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(timestamp + ": Initialize Engine")
    engine = Engine(dh.mdp_data[:100], "q_learner", "kaggle", num_episodes=10000, num_steps=10)
    
    # Create world
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(timestamp + ": Create the world")
    engine.create_world()
    
    # Train agent
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(timestamp + f": Train the agent on {n_samples} samples")
    engine.train_agent()

    # Plot PDPs
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(timestamp + ": Show PDPs plots")
    pdp = PDP(bins = engine.env.bins,
              minmax_scalers = dh._minmax_scalars,
              action_labels = actions,
              state_labels = states)
    pdp. build_pdp_plots(engine.agent.Q, states, savefig=True)


