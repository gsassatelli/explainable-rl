from library import *


def save_engine(engine, path_name=None):
    """ Save engine.
    
    Args:
        engine (Engine): Engine to save.
        path_name (str): path to save the engine
    """
    engine.dh.dataset = None  # Used to remove training set
    with open(path_name, 'wb') as f:
        pickle.dump(engine, f)


def load_engine(path_name=None):
    """ Load engine.
    
    Args:
        path_name (str): path to save the engine
    """
    with open(path_name, 'rb') as f:
        engine = pickle.load(f)
    return engine


def load_data(data_path, n_samples, delimiter=','):
    """Load data from file.

    Args:
        data_path (str): Path to data file.
        n_samples (int): Number of samples to load.
        delimiter (str): Which separates columns.
    """
    file_type = data_path.split('.')[-1]
    if file_type == 'csv':
        dataset = pd.read_csv(data_path, sep=delimiter)
    elif file_type == 'xlsx':
        dataset = pd.read_excel(data_path)
    elif file_type == 'parquet':
        dataset = pd.read_parquet(data_path)
    else:
        raise ValueError("File type not supported")
    dataset.sample(frac=1)
    return dataset[:n_samples]


def split_train_test(dataset, train_test_split=0.2):
    """Split dataset into train and test.

    Args:
        dataset (pd.DataFrame): Dataset.
        train_test_split (float): Proportion of test data.
    
    Returns:
        train_dataset (pd.DataFrame): Train dataset.
        test_dataset (pd.DataFrame): Test dataset.
    """
    dataset = dataset.sample(frac=1)
    split = int(train_test_split*len(dataset))
    return dataset[split:], dataset[:split]


def convert_to_string(state):
    """Convert a state to a string.

    Args:
        state (list): State to convert.

    Returns:
        str: State as a string.
    """
    return ",".join([str(s) for s in state])


def convert_to_list(state_str):
    """Convert a state string to a list.

    Args:
        state_str (str): State as a string.

    Returns:
        list: State as a list.
    """
    return [int(s) for s in state_str.split(",")]


def decay_param(param,
                decay,
                min_param):
    """Decay a parameter.

    Args:
        param (float): Parameter to decay.
        decay (float): Decay rate.
        min_param (float): Minimum value of the parameter.

    Returns:
        float: Updated parameter.
    """
    return max(param * (1 - decay), min_param)
