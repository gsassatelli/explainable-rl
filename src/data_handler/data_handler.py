# Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Union
import copy

# TODO: change normalisation to [0, 1]
# TODO: add flag for next states
class DataHandler:
    """Data handler class to store and preprocess data needed for training.

    """

    __slots__ = ["data_path", "dataset", "_normalised_cols", "_minmax_scalars",
                 "_state_labels", "_action_labels", "_reward_labels", "mdp_data"]

    def __init__(self, data_path,
                 state_labels,
                 action_labels,
                 reward_labels):
        """Initialize the DataHandler class.

        Args:
            data_path (str): path to the data file.
            state_labels (list): list of state labels.
            action_labels (list): list of action labels.
            reward_labels (list): list of reward labels.
        """

        self.data_path = data_path
        self.dataset = None
        self._normalised_cols = []
        self._minmax_scalars = {}
        self._state_labels = state_labels
        self._action_labels = action_labels
        self._reward_labels = reward_labels
        self.mdp_data = None

    def prepare_data_for_engine(self, col_delimiter=',',
                                cols_to_normalise=None):
        """Prepare data for engine.

        Args:
            col_delimiter (str): column delimiter.
            cols_to_normalise (list): list of columns to normalise.
        """
        self.load_data(delimiter=col_delimiter)
        self.preprocess_data(normalisation=True,
                             columns_to_normalise=cols_to_normalise)


    def load_data(self, delimiter=','):
        """Load data from file.

        Args:
            delimiter (str): column
        """
        self.dataset = pd.read_csv(self.data_path, sep=delimiter)

    def preprocess_data(self,
                        normalisation=True,
                        columns_to_normalise=None):
        """Preprocess data into state, action and reward spaces.

        Preprocessing applies shuffling, normalisation (if selected) and
        splits the dataset into states, actions and rewards.

        Args: normalisation (bool): True if normalisation is to be applied.
        columns_to_normalise (list): Columns on which to apply
        normalisation. if left empty all columns will be normalised.

        TODO: Extension - aggregate over a time period
        """
        np.random.seed = 1
        self._filter_data()
        self.dataset = self.dataset.sample(frac=1)
        if normalisation:
            self.normalise_dataset(cols_to_norm=columns_to_normalise)

        s = self.dataset[self._state_labels]
        a = self.dataset[self._action_labels]
        r = self.dataset[self._reward_labels]

        self.mdp_data = pd.concat({'s': s, 'a': a, 'r': r}, axis=1)

    def normalise_dataset(self, cols_to_norm=None):
        """Normalise the dataset to centre with mean zero and variance one.

        Args:
            cols_to_norm (list): the column names that need normalising
        """
        self._fit_standard_scalars()
        if cols_to_norm is None:
            cols_to_norm = self.dataset.columns
        for col in cols_to_norm:
            self._transform_col(col_name=col)
            self._normalised_cols.append(col)

    def reverse_norm(self):
        """Reverse the normalising of the dataset.

        """
        for col in self._normalised_cols:
            self._inverse_transform_col(col_name=col)

    def get_actions(self):
        """Get the actions taken in the dataset.

        Returns:
            pd.DataFrame of the actions.
        """
        return self.dataset['s']

    def get_rewards(self):
        """Get the rewards taken in the dataset.

        Returns:
            pd.DataFrame of the rewards.
        """
        return self.dataset['r']

    def get_states(self):
        """Get the states taken in the dataset.

        Returns:
            pd.DataFrame of the states.
        """
        return self.dataset['s']

    def _filter_data(self):
        """Filter the dataset.

        """
        self.dataset = self.dataset.dropna()

    def _transform_col(self, col_name: str):
        """Normalise one column of the dataset.

        """
        scalar = self._minmax_scalars[col_name]
        self.dataset[col_name] = \
            scalar.transform(pd.DataFrame(self.dataset[col_name]))

    def _inverse_transform_col(self, col_name: str):
        """Reverse the normalisation of one column of the dataset.

        """
        scalar = self._minmax_scalars[col_name]
        self.dataset[col_name] = scalar.inverse_transform(
            pd.DataFrame(self.dataset[col_name]))

    def _fit_standard_scalars(self):
        """Train the sklearn MinMaxScaler and store one per column.

        """
        for col in self.dataset:
            scalar = MinMaxScaler()
            scalar = scalar.fit(pd.DataFrame(self.dataset[col]))
            self._minmax_scalars[col] = scalar


# if __name__ == "__main__":
#     states = ['competitorPrice', 'adFlag', 'availability']
#     actions = ['price']
#     rewards = ['revenue']
#     dh = DataHandler('../kaggle-dummy-dataset/train.csv', states, actions,
#                      rewards)
#     dh.prepare_data_for_engine(col_delimiter='|', cols_to_normalise=actions)