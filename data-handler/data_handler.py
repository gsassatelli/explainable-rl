import pandas as pd
import numpy as np
from typing import List, Tuple


class DataHandler:
    """Data handler class to store and preprocess data needed for training.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.dataset = None

    def load_data(self, delimiter: str = ','):
        """Load and store a csv dataset."""
        self.dataset = pd.read_csv(self.data_path, sep=delimiter)

    def filter_data(self):
        """Filter the dataset."""
        self.dataset = self.dataset.dropna()

    def preprocess_data(self, state_labels: List[str],
                        action_labels: List[str],
                        reward_labels: List[str]) \
            -> Tuple[np.ndarray]:
        # TODO: Make output np.ndarray
        # TODO: Extension - aggregate over a time period
        """Preprocess data into state, action and reward spaces.
        Parameters
        ----------
        state_labels
            Dataset labels that make up the state space.
        action_labels
            Dataset labels from which the action space is derived.
        reward_labels
            Dataset labels from which the reward is derived.

        Returns
        -------
        np.ndarray
            np.ndarray of state, action, reward space.
        """
        np.random.seed = 1
        self.dataset = self.dataset.sample(frac=1)
        states = np.array(self.dataset[state_labels])
        actions = np.array(self.dataset[action_labels])
        rewards = np.array(self.dataset[reward_labels])
        mdp = (states, actions, rewards)
        return mdp


if __name__ == "__main__":
    dh = DataHandler('../kaggle-dummy-dataset/train.csv')
    dh.load_data(delimiter='|')
    print(dh.dataset.columns)
    state_labels = ['competitorPrice', 'adFlag', 'availability']
    action_labels = ['price']
    reward_labels = ['revenue']
    dh.filter_data()
    out = dh.preprocess_data(state_labels=state_labels,
                       action_labels=action_labels,
                       reward_labels=reward_labels)
    print(out[0].shape, out[1].shape, out[2].shape)
