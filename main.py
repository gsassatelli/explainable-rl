from library import *

# Import functions
from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler
from src.explainability.pdp import PDP
from src.explainability.shap_values import ShapValues


def run_all(hyperparam_dict, verbose=True, show_plots=True):
    # Load data
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{timestamp}: Load data")
    state_labels = hyperparam_dict['states']
    action_labels = hyperparam_dict['actions']
    reward_labels = hyperparam_dict['rewards']
    n_samples = hyperparam_dict['n_samples']
    n_episodes = hyperparam_dict['num_episodes']
    shap_num_samples = hyperparam_dict['shap_num_samples']
    dh = DataHandler(data_path=hyperparam_dict['data_path'],
                     state_labels=state_labels,
                     action_labels=action_labels,
                     reward_labels=reward_labels,
                     n_samples=n_samples)

    # Preprocess the data
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if verbose:
        print(f"{timestamp}: Preprocess data")
    cols_to_normalise = list(set(hyperparam_dict['states'] + hyperparam_dict['actions'] + hyperparam_dict['rewards']))
    dh.prepare_data_for_engine(col_delimiter=hyperparam_dict['col_delimiter'],
                               cols_to_normalise=cols_to_normalise)

    # Create engine
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if verbose:
        print(f"{timestamp}: Initialize Engine")
    engine = Engine(dh,
                    agent_type=hyperparam_dict['agent_type'],
                    env_type=hyperparam_dict['env_type'],
                    num_episodes=hyperparam_dict['num_episodes'],
                    num_steps=hyperparam_dict['num_steps'],
                    bins=hyperparam_dict['bins'])

    # Create world
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if verbose:
        print(f"{timestamp}: Create the world")
    engine.create_world()

    # Train agent
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if verbose:
        print(f"{timestamp}: Train the agent on {n_samples} samples and {n_episodes} episodes")
    engine.train_agent()

if __name__ == "__main__":

    # Define hyperparameters
    hyperparam_dict_ds_data_suggest = {
        'states': ['lead_time', 'length_of_stay',
                   'competitor_price_difference_bin', 'demand_bin', 'price'],
        'actions': ['price'],
        'bins': [10, 10, 4, 4, 4, 4],
        'rewards': ['reward'],
        'feature_types': {
            'lead_time': "continuous",
            'length_of_stay': "continuous",
            'competitor_price_difference_bin': "discrete",
            'demand_bin': "discrete",
            'price': "continuous",
            'reward': "continuous"
        },
        'n_samples': 1000,
        'data_path': 'data/ds-data/my_example_data.parquet',
        'col_delimiter': '|',
        'cols_to_normalise': ['lead_time', 'length_of_stay',
                              'competitor_price_difference_bin', 'demand_bin', 'price', 'reward'],
        'agent_type': 'q_learner',
        'env_type': 'strategic_pricing_suggest',
        'num_episodes': 500,
        'num_steps': 1,
        'train_test_split': 0.2,
        'shap_num_samples': 1
    }


    for i in range(1):
        run_all(hyperparam_dict_ds_data_suggest)
        # Run this 10 times to check everything was fine.
