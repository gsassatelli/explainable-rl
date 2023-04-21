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

    ###########################################################
    ################ Evaluate agent ###########################
    ###########################################################

    # TODO: denorm states, actions and rewards (using datahandler's inverse scaling)

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if verbose:
        print(f"{timestamp}: Evaluate agent")

    states, actions, b_actions, rewards_hist, actions_agent, b_actions_agent, rewards_agent = \
        engine.evaluate_agent()
    # Sum obtained reward optimal vs historical policy
    print(f"Return based on historical data: {np.sum(rewards_hist)}")
    print(f"Return based on agent policy: {np.sum(rewards_agent)}")

    import matplotlib.pyplot as plt
    plt.scatter(actions, actions_agent)
    plt.savefig('policy.png')

    ###########################################################
    ################# End of evaluation #######################
    ###########################################################

    # Plot PDPs
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if verbose:
        print(f"{timestamp}: Show PDPs plots")
    pdp = PDP(engine=engine)
    pdp.build_data_for_plots(engine.agent.Q, engine.agent.Q_num_samples)
    type_features = hyperparam_dict['feature_types']
    fig_name = "PDP plots - All states"
    pdp.plot_pdp(states_names=state_labels, fig_name=fig_name,
                 savefig=True, all_states=True)

    # Compute SHAP values
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{timestamp}: Show SHAP values plots")
    shap_values = ShapValues(sample=[8, 1, 1, 1],
                             engine=engine,
                             number_of_samples=shap_num_samples)
    shaps, predicted_action = shap_values.compute_shap_values()
    print(shaps)
    print(predicted_action)


if __name__ == "__main__":

    # Define hyperparameters
    hyperparam_dict_ds_data_suggest = {
        'states': ['lead_time', 'length_of_stay',
                   'competitor_price_difference_bin', 'demand_bin', 'price'],
        'actions': [price_bin / 10 for price_bin in range(1, 11)],
        'bins': [10, 10, 4, 4, 10, 10],
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

    hyperparam_dict_kaggle_data_suggest = {
        'states': ['competitorPrice', 'adFlag', 'availability', 'price'],
        'actions': [price_bin / 10 for price_bin in range(1, 11)],
        'rewards': ['revenue'],
        'feature_types': {
            'competitorPrice': "continuous",
            'adFlag': "discrete",
            'availability': "discrete",
            'price': "continuous",
            'revenue': "continuous"
        },
        'bins': [10, 2, 2, 10, 10],
        'n_samples': 2000,
        'data_path': 'data/kaggle-dummy-dataset/train.csv',
        'col_delimiter': '|',
        'cols_to_normalise': ['competitorPrice', 'adFlag', 'availability', 'price'],
        'agent_type': 'q_learner',
        'env_type': 'strategic_pricing_suggest',
        'num_episodes': 1000,
        'num_steps': 1,
        'train_test_split': 0.2,
        'shap_num_samples': 1
    }

    hyperparam_dict_ds_data_predict = {
        'states': ['lead_time', 'length_of_stay',
                   'competitor_price_difference_bin', 'demand_bin'],
        'actions': ['price'],
        'rewards': ['reward'],
        'bins': [10, 10, 4, 4, 10],
        'feature_types': {
            'lead_time': "continuous",
            'length_of_stay': "continuous",
            'competitor_price_difference_bin': "discrete",
            'demand_bin': "discrete",
            'price': "continuous",
            'reward': "continuous"
        },
        'n_samples': 100000,
        'data_path': 'data/ds-data/my_example_data.parquet',
        'col_delimiter': '|',
        'cols_to_normalise': ['lead_time', 'length_of_stay',
                              'competitor_price_difference_bin', 'demand_bin', 'price', 'reward'],
        'agent_type': 'q_learner',
        'env_type': 'strategic_pricing_predict',
        'num_episodes': 5000,
        'num_steps': 1,
        'train_test_split': 0.2,
        'shap_num_samples': 10
    }

    hyperparam_dict_kaggle_data_predict = {
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
        'bins': [10, 2, 2, 10],
        'n_samples': 20000,
        'data_path': 'data/kaggle-dummy-dataset/train.csv',
        'col_delimiter': '|',
        'cols_to_normalise': ['competitorPrice', 'adFlag', 'availability', 'price'],
        'agent_type': 'q_learner',
        'env_type': 'strategic_pricing_predict',
        'num_episodes': 1000,
        'num_steps': 1,
        'train_test_split': 0.2,
        'shap_num_samples': 1
    }

    for i in range(1):
        run_all(hyperparam_dict_ds_data_predict)
        # Run this 10 times to check everything was fine.
