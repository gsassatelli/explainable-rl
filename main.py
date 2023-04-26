from library import *

# Import functions
from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler
from src.explainability.pdp import PDP
from src.explainability.shap_values import ShapValues


def load_data(path, delimiter=','):
    """Load data from file.

    Args:
        delimiter (str): Which separates columns.
    """
    file_type = path.split('.')[-1]
    if file_type == 'csv':
        dataset = pd.read_csv(path, sep=delimiter)
    elif file_type == 'xlsx':
        dataset = pd.read_excel(path)
    elif file_type == 'parquet':
        dataset = pd.read_parquet(path)
    else:
        raise ValueError("File type not supported")

    return dataset


def run_all(hyperparam_dict, verbose=True, show_plots=True):
    # Load data
    dataset = load_data(hyperparam_dict['dataset']['data_path'],
                        delimiter=hyperparam_dict['dataset']['col_delimiter'])
    dataset=dataset.sample(n=1000)
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{timestamp}: Load data")

    dh = DataHandler(hyperparam_dict=hyperparam_dict, dataset=dataset,
                     test_dataset=dataset)

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
    pdp.plot_pdp(feature="length_of_stay", fig_name="PDP plots", savefig=True)

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
                    'n_test_samples': 100,
                    'normalisation': True},

        "training": {'env_type': 'strategic_pricing_predict',
                     'num_episodes': 100,
                     'num_steps': 1,
                     'train_test_split': 0.2,
                     'evaluate': False,
                     'num_eval_steps': 1},

        "agent": {'agent_type': 'sarsa_lambda',
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

        "program_flow": {"verbose": False}
    }

    for i in range(1):
        run_all(hyperparam_dict_ds_data_suggest)
        # Run this 10 times to check everything was fine.
