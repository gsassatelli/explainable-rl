import os

import time
import tracemalloc

from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler
from src.explainability.pdp import PDP
from datetime import datetime

class PerformanceEvaluator:
    def __init__(self):
        # Directory containing all results for this evaluator
        self.init_time = time.strftime("%Y%m%d-%H:%M:%S")
        os.mkdir(f"evaluations/performance-{self.init_time}")

    def get_all_performances(self):
        self.get_benchmark_perf()

    def get_benchmark_perf(self):
        """ Get performance (space and time) for constant benchmark settings.

        """
        # Benchmark settings
        NUM_EP = int(1e+1)
        NUM_BINS = 10
        NUM_SAMPLES = int(1e+5)

        # Halt any ongoing memory tracing not to get memory results from previous code runs
        tracemalloc.stop()

        # Start memory tracing
        tracemalloc.start()

        start_time = time.time()

        # Run the code to be evaluated
        self.run_training_loop(num_episodes=NUM_EP,
                               num_bins=NUM_BINS,
                               num_samples=NUM_SAMPLES)

        end_time = time.time()

        # Memory usage results returned in bytes (num. Bytes / 1024 = num. KiB ; num. KiB / 1024 = num. MiB)
        _, first_peak = tracemalloc.get_traced_memory()

        # Peak in MiB
        peak = first_peak/(1024*1024)

        time_summary = f"Time spent (s): {end_time-start_time}"
        space_summary = f"Peak memory usage (MiB): {peak}"

        with open(f"evaluations/performance-{self.init_time}/benchmark-report.txt", "w") as report_file:
            report_file.write(f'BENCHMARK SETTINGS\nNumber of episodes: {NUM_EP}\n'
                              f'Number of bins: {NUM_BINS}\nNumber of samples: {NUM_SAMPLES}\n\n'
                              f'RESULTS\n{time_summary}\n{space_summary}')

    def run_training_loop(self, num_episodes, num_bins, num_samples):
        """ Run an example main.py.

        Eventually, may load the hyperparameter data from a user-facing file.

        """

        # This is the function call to potentially be replaced
        hyperparam_dict = self.get_hyperparam_dict_ds_data(num_episodes, num_bins, num_samples)

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
                        num_steps=hyperparam_dict['num_steps'],
                        bins=hyperparam_dict['bins']
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

    def get_hyperparam_dict_ds_data(self, num_episodes, num_bins, num_samples):
        hyperparam_dict_ds_data = {
            'states': ['lead_time', 'length_of_stay',
                       'competitor_price_difference_bin', 'demand_bin'],
            'actions': ['price'],
            'bins': [num_bins] * 5,
            'rewards': ['reward'],
            'feature_types': {
                'lead_time': "continuous",
                'length_of_stay': "continuous",
                'competitor_price_difference_bin': "discrete",
                'demand_bin': "discrete",
                'price': "continuous",
                'reward': "continuous"
            },
            'n_samples': num_samples,
            'data_path': '../../data/ds-data/my_example_data.parquet',
            'col_delimiter': '|',
            'cols_to_normalise': ['lead_time', 'length_of_stay',
                                  'competitor_price_difference_bin', 'demand_bin', 'price', 'reward'],
            'agent_type': 'q_learner',
            'env_type': 'strategic_pricing',
            'num_episodes': num_episodes,
            'num_steps': 1
        }
        return hyperparam_dict_ds_data


if __name__ == "__main__":
    performance_evaluator = PerformanceEvaluator()
    performance_evaluator.get_benchmark_perf()