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

    dh = DataHandler(hyperparam_dict=hyperparam_dict)

    # Preprocess the data
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if verbose:
        print(f"{timestamp}: Preprocess data")
    dh.prepare_data_for_engine()

    # Create engine
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if verbose:
        print(f"{timestamp}: Initialize Engine")
    engine = Engine(dh, hyperparam_dict=hyperparam_dict)

    # Create world
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if verbose:
        print(f"{timestamp}: Create the world")
    engine.create_world()

    # Train agent
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if verbose:
        print(f"{timestamp}: Train the agent")
    engine.train_agent()

    # PDP plots
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if verbose:
        print(f"{timestamp}: PDP plots")
    pdp = PDP(engine)
    pdp.build_data_for_plots()
    pdp.plot_pdp(fig_name="PDP plots", savefig=True)

    # SHAP values
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if verbose:
        print(f"{timestamp}: SHAP values")
    shap_values = ShapValues(sample=[8, 1, 1, 1, 10], engine=engine)
    shap_values.compute_shap_values()


if __name__ == "__main__":

    hyperparam_dict_ds_data_suggest = {
        "dimensions": {'states': {'lead_time': 10,
                                  'length_of_stay': 10,
                                  'competitor_price_difference_bin': 4,
                                  'demand_bin': 4,
                                  'price': 4},
                       'actions': {'price': 10},
                       'rewards': ['reward']
                       },

        "dataset": {'data_path': 'data/ds-data/my_example_data.parquet',
                    'col_delimiter': '|',
                    'n_samples': 1000,
                    'normalisation': True},

        "training": {'env_type': 'strategic_pricing_suggest',
                     'num_episodes': 500,
                     'num_steps': 1,
                     'train_test_split': 0.2},

        "agent": {'agent_type': 'q_learner',
                  "gamma": 0.3,
                  "epsilon": 0.4,
                  "epsilon_decay": 0.1,
                  "epsilon_minimum": 0.1,
                  "learning_rate": 0.1,
                  "learning_rate_decay": 0.1,
                  "learning_rate_minimum": 0.1,
                  "lambda": 0.2,
                  "use_uncertainty": True,
                  "q_importance": 0.7,
                  },

        "explainability": {'shap_num_samples': 1},

        "program_flow": {"verbose": True}
    }

    for i in range(1):
        run_all(hyperparam_dict_ds_data_suggest)
        # Run this 10 times to check everything was fine.
