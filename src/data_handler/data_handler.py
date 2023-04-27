from src.foundation.library import *


class DataHandler:
    """Data Handler which stores and preprocesses data needed for training."""

    def __init__(self, hyperparam_dict, dataset, test_dataset=None):
        """Initialise the DataHandler.

        Args:
            hyperparam_dict (dict): Dictionary of hyperparameters.
            dataset (pd.DataFrame): Training dataset.
            test_dataset (pd.DataFrame): Test dataset.
        """
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.hyperparam_dict = hyperparam_dict
        self.data_path = hyperparam_dict["dataset"]["data_path"]
        self._n_samples = hyperparam_dict["dataset"]["n_samples"]
        self._normalised_cols = []
        self.minmax_scalars = {}
        self.state_labels = self._get_labels(hyperparam_dict["dimensions"]["states"])
        self.action_labels = self._get_labels(hyperparam_dict["dimensions"]["actions"])
        self.reward_labels = self._get_labels(hyperparam_dict["dimensions"]["rewards"])
        self.mdp_data = None
        self.test_mdp_data = None

        self.prepare_data_for_engine()

    def prepare_data_for_engine(self, cols_to_normalise=None):
        """Prepare the data to be given to the engine.

        Args:
            cols_to_normalise (list): List of columns to normalise.
        """
        if cols_to_normalise is None:
            cols_to_normalise = list(
                set(self.state_labels + self.action_labels + self.reward_labels)
            )

        self.preprocess_data(
            normalisation=self.hyperparam_dict["dataset"]["normalisation"],
            columns_to_normalise=cols_to_normalise,
        )

    def preprocess_data(self, normalisation=True, columns_to_normalise=None):
        """Preprocess data into state, action and reward spaces.

        Preprocessing applies shuffling, normalisation (if selected) and
        splits the dataset into states, actions and rewards.

        Args:
            normalisation (bool): True if normalisation is to be applied.
            columns_to_normalise (list): Columns on which to apply
                normalisation. If left empty all columns will be normalised.
        """
        np.random.seed = 1
        self._filter_data()
        self.dataset = self.dataset.sample(frac=1)
        self.test_dataset = self.test_dataset.sample(frac=1)

        if normalisation:
            self.normalise_dataset(cols_to_norm=columns_to_normalise)

        s = self.dataset[self.state_labels]
        r = self.dataset[self.reward_labels]
        try:
            a = self.dataset[self.action_labels]
            self.mdp_data = pd.concat({"s": s, "a": a, "r": r}, axis=1)
        except KeyError:
            self.mdp_data = pd.concat({"s": s, "r": r}, axis=1)

        # Apply preprocessing to test data
        if not self.test_dataset is None:
            test_s = self.test_dataset[self.state_labels]
            test_r = self.test_dataset[self.reward_labels]
            test_a = self.test_dataset[self.action_labels]
            self.test_mdp_data = pd.concat(
                {"s": test_s, "a": test_a, "r": test_r}, axis=1
            )

    def normalise_dataset(self, cols_to_norm=None):
        """Normalise the dataset to centre with mean zero and variance one.

        Args:
            cols_to_norm (list): The column names that need normalising.
        """
        self._fit_standard_scalars()
        if cols_to_norm is None:
            cols_to_norm = self.dataset.columns
        for col in cols_to_norm:
            self._transform_col(col_name=col)
            self._normalised_cols.append(col)

    def reverse_norm(self):
        """Reverse the normalising of the dataset."""
        for col in self._normalised_cols:
            self._inverse_transform_col(col_name=col)

    def get_actions(self, split="train"):
        """Get the actions taken in the dataset.

        Args:
            split (str): Specifies train or test split.

        Returns:
            pd.DataFrame: Actions.
        """
        if split == "train":
            return self.mdp_data["a"]
        return self.test_mdp_data["a"]

    def get_action_labels(self):
        """Get the action labels.

        Returns:
            list: Action labels.
        """
        return self.action_labels

    def get_rewards(self, split="train"):
        """Get the rewards taken in the dataset.

        Args:
            split (str): Specifies train or test split.
        
        Returns:
            pd.DataFrame: The rewards.
        """
        if split == "train":
            return self.mdp_data["r"]
        return self.test_mdp_data["r"]

    def get_states(self, split="train"):
        """Get the states taken in the dataset.

        Args:
            split (str): Specifies train or test split.
        
        Returns:
            pd.DataFrame: The states.
        """
        if split == "train":
            return self.mdp_data["s"]
        return self.test_mdp_data["s"]

    def _filter_data(self):
        """Filter the dataset."""
        self.dataset = self.dataset.dropna()

    def _transform_col(self, col_name: str):
        """Normalise one column of the dataset.

        Args:
            col_name (str): The column name.
        """
        scalar = self.minmax_scalars[col_name]
        self.dataset[col_name] = scalar.transform(pd.DataFrame(self.dataset[col_name]))
        self.test_dataset[col_name] = scalar.transform(
            pd.DataFrame(self.test_dataset[col_name])
        ).clip(0, 1)

    def _inverse_transform_col(self, col_name: str):
        """Reverse the normalisation of one column of the dataset.

        Args:
            col_name (str): The column name.
        """
        scalar = self.minmax_scalars[col_name]
        self.dataset[col_name] = scalar.inverse_transform(
            pd.DataFrame(self.dataset[col_name])
        )

    def _fit_standard_scalars(self):
        """Train the sklearn MinMaxScaler and store one per column."""
        for col in self.dataset:
            scalar = MinMaxScaler()
            scalar = scalar.fit(pd.DataFrame(self.dataset[col]))
            self.minmax_scalars[col] = scalar

    def _get_labels(self, label_dict):
        """Get the labels from the label dictionary.

        Args:
            label_dict (dict): The label dictionary.

        Returns:
            list: The labels.
        """
        labels = []
        for key in label_dict:
            labels.append(key)
        return labels
