from foundation.engine import Engine
from data_handler.data_handler import DataHandler
from explainability.pdp import PDP
from datetime import datetime


def run_all(hyperparam_dict):
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{timestamp}: Load data")
    states = hyperparam_dict['states']
    actions = hyperparam_dict['actions']
    rewards = hyperparam_dict['rewards']
    n_samples = hyperparam_dict['n_samples']
    dh = DataHandler(data_path=hyperparam_dict['data_path'],
                     state_labels=states,
                     action_labels=actions,
                     reward_labels=rewards,
                     n_samples=n_samples)

    # Preprocess the data
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{timestamp}: Preprocess data")
    dh.prepare_data_for_engine(col_delimiter=hyperparam_dict['col_delimiter'],
                               cols_to_normalise=hyperparam_dict[
                                   'cols_to_normalise'])

    # Create engine
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    print(f"{timestamp}: Initialize Engine")
    engine = Engine(dh,
                    agent_type=hyperparam_dict['agent_type'],
                    env_type=hyperparam_dict['env_type'],
                    num_episodes=hyperparam_dict['num_episodes'],
                    num_steps=hyperparam_dict['num_steps']
                    )
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
              minmax_scalars=dh.minmax_scalars,
              action_labels=actions,
              state_labels=states)
    pdp.build_data_for_plots(engine.agent.Q, engine.agent.Q_num_samples)
    type_features = hyperparam_dict['feature_types']
    fig_name = "PDP plots - All states"
    pdp.plot_pdp(states_names=states, fig_name=fig_name,
                 type_features=type_features, savefig=True, all_states=True)
    fig_name = "PDP plots - Visited states"
    pdp.plot_pdp(states_names=states, fig_name=fig_name,
                 type_features=type_features, savefig=True, all_states=False)


if __name__ == "__main__":
    hyperparam_dict_ds_data = {
        'states': ['lead_time', 'length_of_stay',
                   'competitor_price_difference_bin', 'demand_bin'],
        'actions': ['price'],
        'rewards': ['reward'],
        'feature_types': {
            'lead_time': "continuous",
            'length_of_stay': "continuous",
            'competitor_price_difference_bin': "discrete",
            'demand_bin': "discrete",
            'price': "continuous",
            'reward' : "continuous"
        },
        'n_samples': 500000,
        'data_path': '../data/ds-data/my_example_data.parquet',
        'col_delimiter': '|',
        'cols_to_normalise': ['lead_time', 'length_of_stay',
                   'competitor_price_difference_bin', 'demand_bin', 'price', 'reward'],
        'agent_type': 'q_learner',
        'env_type': 'kaggle',
        'num_episodes': 15000,
        'num_steps': 10
    }

    hyperparam_dict_kaggle_data = {
        'states': ['competitorPrice', 'adFlag', 'availability'],
        'actions': ['price'],
        'rewards': ['revenue'],
        'feature_types': {
            'competitorPrice': "continuous",
            'adFlag': "discrete",
            'availability': "discrete",
            'price': "continuous",
            'revenue': "continuous"
        },
        'n_samples': 500000,
        'data_path': '../data/kaggle-dummy-dataset/train.csv',
        'col_delimiter': '|',
        'cols_to_normalise': ['competitorPrice', 'adFlag', 'availability', 'price'],
        'agent_type': 'q_learner',
        'env_type': 'kaggle',
        'num_episodes': 1000,
        'num_steps': 10
    }
    for i in range(1):
        run_all(hyperparam_dict_ds_data)
        # ran this 10 times to check everything was fine.