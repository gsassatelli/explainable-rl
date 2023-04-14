import os
import io
import time
import tracemalloc
import cProfile, pstats
import matplotlib.pyplot as plt

from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler
from src.explainability.pdp import PDP
from datetime import datetime

class PerformanceEvaluator:
    def __init__(self):
        """Initialise a PerformanceEvaluator.

        """
        # Define the directory containing all the evaluations of this performance evaluator
        self.init_time = time.strftime("%Y%m%d-%H:%M:%S")
        os.mkdir(f"evaluations/performance-{self.init_time}")

        # Benchmark settings
        self.NUM_EP = int(1e+1)
        self.NUM_BINS = 10
        self.NUM_SAMPLES = int(1e+5)

        # Graph colors
        self.time_color = "cornflowerblue"
        self.space_color = "sienna"

        # Print statements
        self.verbose = True

        # How many of the top lines of memory allocation to record
        self.num_memalloc_lines_to_keep = 5000

    def get_all_performance_evaluations(self):
        """Perform and save all desired performance evaluations."""
        self.get_benchmark_performance()
        self.get_performance_graphs()
        self.get_time_breakdown_per_function()
        self.get_space_breakdown_per_function()

    def get_benchmark_performance(self):
        """Get performance (space and time) for constant benchmark settings.

        Note, the reason the timeit module is not used is because the piece of code being profiled
        is lengthy (whole training flow); timeit is designed to test small snippets of code
        by running them many times.

        """
        if self.verbose:
            print("\nGETTING BENCHMARK PERFORMANCE")
        # Halt any ongoing memory tracing not to get memory results from previous code runs
        tracemalloc.stop()

        # Start memory tracing
        tracemalloc.start()

        start_time = time.time()

        # Run the code to be evaluated
        self.run_training_loop(num_episodes=self.NUM_EP,
                               num_bins=self.NUM_BINS,
                               num_samples=self.NUM_SAMPLES)

        end_time = time.time()

        # Memory usage results returned in bytes (num. Bytes / 1024 = num. KiB ; num. KiB / 1024 = num. MiB)
        _, first_peak = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        # Peak in MiB
        peak = first_peak / (1024 * 1024)

        time_summary = f"Time spent (s): {end_time - start_time}"
        space_summary = f"Peak memory usage (MiB): {peak}"

        with open(f"evaluations/performance-{self.init_time}/benchmark-report.txt", "w") as report_file:
            report_file.write(f'BENCHMARK SETTINGS\nNumber of episodes: {self.NUM_EP}\n'
                              f'Number of bins: {self.NUM_BINS}\nNumber of samples: {self.NUM_SAMPLES}\n\n'
                              f'RESULTS\n{time_summary}\n{space_summary}')

    def do_pre_benchmark_run_configuration(self):
        """Prepare the benchmark performance run."""
        tracemalloc.stop()
        tracemalloc.start()
        start_time = time.time()
        return start_time

    def get_post_benchmark_run_results(self):
        """Get the results of the benchmark performance run."""
        end_time = time.time()
        _, first_peak = tracemalloc.get_traced_memory()
        peak = first_peak / (1024 * 1024)
        return end_time, peak

    def get_performance_graphs(self):
        """ Plot performance (time and space) against chosen varying parameters.

            Parameters:
            - Number of samples
            - Number of episodes
            - Number of bins (held constant for all dimensions)

        """
        if self.verbose:
            print("\nGETTING PERFORMANCE GRAPHS")

        if self.verbose:
            print("* Plot of performance vs number of samples")
        # Plot of performance vs number of samples
        num_sample_range = [int(1e2), int(1e3), int(1e4), int(1e5)]
        times, memory = self.get_times_and_memory_from_parameter_range(parameter_name="num_samples", x=num_sample_range)
        self.plot_performance_graph(x_label="samples", x=num_sample_range, times=times, memory=memory)

        if self.verbose:
            print("* Plot of performance vs number of episodes")
        # Plot of performance vs number of episodes
        num_ep_range = [int(1e1), int(1e2), int(1e3), int(1e4)]
        times, memory = self.get_times_and_memory_from_parameter_range(parameter_name="num_episodes", x=num_ep_range)
        self.plot_performance_graph(x_label="episodes", x=num_ep_range, times=times, memory=memory)

        if self.verbose:
            print("* Plot of performance vs number of bins")
        # Plot of performance vs number of bins
        num_bin_range = [2, 5, 10, 20, 30, 40, 50]
        times, memory = self.get_times_and_memory_from_parameter_range(parameter_name="num_bins", x=num_bin_range)
        self.plot_performance_graph(x_label="bins", x=num_bin_range, times=times, memory=memory)

    def get_times_and_memory_from_parameter_range(self, parameter_name, x):
        """Get time and memory complexities for varying values of an inputted parameter name.

        Args:
            parameter_name (str): Name of the parameter against which time and memory complexities are recorded.
            x (list[int]): Range of values to vary the parameter.

        Returns:
            (list[float], list[float]): Time and memory recordings for the different parameter values in x.

        """
        times = []
        memory = []

        for val in x:

            start_time = self.do_pre_benchmark_run_configuration()

            if parameter_name == "num_samples":
                self.run_training_loop(num_episodes=self.NUM_EP, num_bins=self.NUM_BINS, num_samples=val)

            elif parameter_name == "num_bins":
                self.run_training_loop(num_episodes=self.NUM_EP, num_bins=val, num_samples=self.NUM_SAMPLES)

            elif parameter_name == "num_episodes":
                self.run_training_loop(num_episodes=val, num_bins=self.NUM_BINS, num_samples=self.NUM_SAMPLES)

            end_time, peak = self.get_post_benchmark_run_results()

            times += [end_time - start_time]
            memory += [peak]

        return times, memory

    def plot_performance_graph(self, x_label, x, times, memory):
        """Plot and save a performance graph.

        Args:
            x_label (str): Label of the horizontal axis (e.g. "episodes", "samples", or "bins").
            x (list[int]): Range of the x parameter across which time and memory have been recorded.
            times (list[int]): Results of time complexity recording for different values of x.
            memory (list[int]): Results of memory complexity recording for different values of x.

        """
        plt.style.use("seaborn-v0_8-darkgrid")

        fig, ax1 = plt.subplots()
        ax1.plot(x, times, color=self.time_color, alpha=0.6)
        ax2 = ax1.twinx()
        ax2.plot(x, memory, color=self.space_color, alpha=0.6)

        ax1.set_ylabel("Time (s)", color=self.time_color)
        ax2.set_ylabel("Space (MiB)", color=self.space_color)
        plt.xlabel(f"Number of {x_label}")
        plt.title(f"Time and space complexities vs number of {x_label}")
        plt.grid()
        plt.tight_layout()

        plt.savefig(f"evaluations/performance-{self.init_time}/perf_vs_{x_label}.png")

    def get_time_breakdown_per_function(self):
        """Get a per-function breakdown of time complexity.

            See: https://www.machinelearningplus.com/python/cprofile-how-to-profile-your-python-code/.
            And: https://stackoverflow.com/questions/51536411/saving-cprofile-results-to-readable-external-file.

        """
        if self.verbose:
            print("\nGETTING TIME BREAKDOWN PER FUNCTION")

        profiler = cProfile.Profile()
        profiler.enable()
        self.run_training_loop(num_episodes=self.NUM_EP,
                               num_bins=self.NUM_BINS,
                               num_samples=self.NUM_SAMPLES)
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumtime')
        stats.print_stats()

        with open(f"evaluations/performance-{self.init_time}/per_function_time.txt", "w") as outfile:
            outfile.write(stream.getvalue())

    def get_space_breakdown_per_function(self):
        """Get a per-function breakdown of space complexity.

        """
        if self.verbose:
            print("\nGETTING SPACE BREAKDOWN PER FUNCTION")

        tracemalloc.start()

        self.run_training_loop(num_episodes=self.NUM_EP,
                               num_bins=self.NUM_BINS,
                               num_samples=self.NUM_SAMPLES)

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        with open(f"evaluations/performance-{self.init_time}/per_function_space.txt", "w") as outfile:
            outfile.write(f"TOP {self.num_memalloc_lines_to_keep} MEMORY ALLOCATION LINES")
            for stat in top_stats[:self.num_memalloc_lines_to_keep]:
                outfile.write(f"\n{stat}")

        tracemalloc.stop()

    def run_training_loop(self, num_episodes, num_bins, num_samples):
        """Run an example main.py.

            Eventually, may load the hyperparameter data from a user-facing file.

        Args:
            num_episodes (int): Number of episodes in the training loop.
            num_bins (int): Number of bins used for digitisation (equal number of bins for all dimensions).
            num_samples (int): Number of datapoint samples used for training.

        """
        # TODO: call the outside function.
        if self.verbose:
            print(f"-> Running training loop for {num_episodes} episodes, {num_bins} bins, {num_samples} samples")

        # Get the hyperparameter dictionary for the specified parameters
        hyperparam_dict = self.get_hyperparam_dict_ds_data(num_episodes, num_bins, num_samples)

        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        #print(f"{timestamp}: Load data")
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
        #print(f"{timestamp}: Preprocess data")
        dh.prepare_data_for_engine(col_delimiter=hyperparam_dict['col_delimiter'],
                                   cols_to_normalise=hyperparam_dict[
                                       'cols_to_normalise'])

        # Create engine
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        #print(f"{timestamp}: Initialize Engine")
        engine = Engine(dh,
                        agent_type=hyperparam_dict['agent_type'],
                        env_type=hyperparam_dict['env_type'],
                        num_episodes=hyperparam_dict['num_episodes'],
                        num_steps=hyperparam_dict['num_steps'],
                        bins=hyperparam_dict['bins']
                        )
        # Create world
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        #print(f"{timestamp}: Create the world")
        engine.create_world()

        # Train agent
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        #print(f"{timestamp}: Train the agent on {n_samples} samples")
        engine.train_agent()

        # TODO: removed PDP plotting for now because having many plots successively hinders flow of performance check
        #  this should not be a problem considering the build_data_for_plots has been kept (it is space/time consuming)
        # Plot PDPs
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        #print(f"{timestamp}: Show PDPs plots")
        pdp = PDP(bins=engine.env.bins,
                  minmax_scalars=dh.minmax_scalars,
                  action_labels=actions,
                  state_labels=states)
        pdp.build_data_for_plots(engine.agent.Q, engine.agent.Q_num_samples)
        type_features = hyperparam_dict['feature_types']
        # fig_name = "PDP plots - All states"
        # pdp.plot_pdp(states_names=states, fig_name=fig_name,
        #              type_features=type_features, savefig=True, all_states=True)
        # fig_name = "PDP plots - Visited states"
        # pdp.plot_pdp(states_names=states, fig_name=fig_name,
        #              type_features=type_features, savefig=True, all_states=False)

    def get_hyperparam_dict_ds_data(self, num_episodes, num_bins, num_samples):
        """Load the hyperparameter dictionaries for the Datasparq data.

        Args:
            num_episodes (int): Number of episodes in the training loop.
            num_bins (int): Number of bins used for digitisation (equal number of bins for all dimensions).
            num_samples (int): Number of datapoint samples used for training.

        Returns:
            (dict): Hyperparameter dictionary specific to the Datasparq dataset.

        """
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
    performance_evaluator.get_all_performance_evaluations()


# TODO: now that all the content has been obtained, use html to create pdf from results:
#  https://towardsdatascience.com/how-to-easily-create-a-pdf-file-with-python-in-3-steps-a70faaf5bed5

# TODO: remove all the print statements from the other classes (environment, agent, etc...) because it clouds the output
#  here.
