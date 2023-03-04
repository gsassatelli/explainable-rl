from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler
from src.explainability.pdp import PDP

from datetime import datetime


if __name__ == "__main__":
    # Load data
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{timestamp}: Load data")
    states = ['competitorPrice', 'adFlag', 'availability']
    actions = ['price']
    rewards = ['revenue']
    n_samples = 2000
    dh = DataHandler('kaggle-dummy-dataset/train.csv', states, actions, rewards, n_samples=n_samples)

    # Preprocess the data
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{timestamp}: Preprocess data")
    dh.prepare_data_for_engine(col_delimiter='|', cols_to_normalise=states+actions)

    # Create engine
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{timestamp}: Initialize Engine")
    n_samples = 1000
    dh.mdp_data = dh.mdp_data[:n_samples]
    engine = Engine(dh, "q_learner", "kaggle", num_episodes=100, num_steps=10)

    
    # Create world
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{timestamp}: Create the world")
    engine.create_world()
    
    # Train agent
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{timestamp}: Train the agent on {n_samples} samples")
    engine.train_agent()

    # Plot PDPs
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{timestamp}: Show PDPs plots")
    pdp = PDP(bins=engine.env.bins,
              minmax_scalers=dh._minmax_scalars,
              action_labels=actions,
              state_labels=states)
    pdp.build_data_for_plots(engine.agent.Q)
    type_features = {'competitorPrice': "continuous",
                     'adFlag': "discrete",
                     'availability': "discrete"}
    fig_name = "PDP plots - All states"
    pdp.plot_pdp(states_names=states, fig_name=fig_name, type_features=type_features, savefig=True, all_states=True)
    fig_name = "PDP plots - Visited states"
    pdp.plot_pdp(states_names=states, fig_name=fig_name, type_features=type_features, savefig=True, all_states=False)


    # Plot PDPs
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{timestamp}: Show PDPs plots")
    pdp = PDP(bins=engine.env.bins,
              minmax_scalers=dh._minmax_scalars,
              action_labels=actions,
              state_labels=states)
    pdp.build_data_for_plots(engine.agent.Q)
    type_features = {'competitorPrice': "continuous",
                     'adFlag': "discrete",
                     'availability': "discrete"}
    fig_name = "PDP plots - All states"
    pdp.plot_pdp(states_names=states, fig_name=fig_name, type_features=type_features, savefig=True, all_states=True)
    fig_name = "PDP plots - Visited states"
    pdp.plot_pdp(states_names=states, fig_name=fig_name, type_features=type_features, savefig=True, all_states=False)


