# Note that you need to decorate all the functions being called with the "@profile" decorator for full memory analysis
# To view the memory usage without saving to the results, run: python3 -m memory_profiler performance/full_flow.py
# To save the memory data to a data file, run: mprof run performance/full_flow.py
# To plot the memory usage file just created with the standard title, run: mprof plot
# To plot the memory usage file just created with a custom title, run: mprof plot -t "DESIRED_TITLE"


from foundation.engine import Engine
from data_handler.data_handler import DataHandler
from datetime import datetime
from memory_profiler import profile

@profile
def evaluate_full_flow_performance():
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
    dh.prepare_data_for_engine(col_delimiter='|', cols_to_normalise=states + actions)

    # Create engine
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(timestamp + ": Initialize Engine")
    engine = Engine(dh.mdp_data[:100000], "q_learner", "kaggle", num_episodes=100, num_steps=1)

    # Create world
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(timestamp + ": Create the world")
    engine.create_world()

    # Train agent
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(timestamp + ": Train the agent on 100 samples")
    engine.train_agent()


if __name__ == "__main__":
    evaluate_full_flow_performance()
