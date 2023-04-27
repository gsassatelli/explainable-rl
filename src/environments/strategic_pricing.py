from src.foundation.library import *

# Import functions
from src.foundation.environment import MDP


class StrategicPricing(MDP):
    """Environment for Strategic Pricing."""

    def __init__(self, dh, bins=None):
        """Initialise the Strategic Pricing MDP class.

        Args:
            dh (DataHandler): Data handler object.
        """
        super().__init__(dh)

        if bins is None:
            bins = [10]
        self.state_to_action = {}

        self._state_mdp_data = None
        self._action_mdp_data = None
        self._reward_mdp_data = None
        self._average_rewards = None
        self.bins_dict = None

        self.state_dim = self.dh.get_states().shape[1]
        self.action_dim = len(self.dh.get_action_labels())

        if len(bins) != self.state_dim + 1:
            print(
                "Warning: bins not equal to state_dim + 1. "
                "Setting bins to [10] * (state_dim + 1)"
            )
            self.bins = [10] * (self.state_dim + 1)
        else:
            self.bins = bins

    def initialise_env(self):
        """Create the environment given the MDP information."""
        self._average_rewards = self._make_rewards_from_data()

    def _transform_df_to_numpy(self):
        """Transform the MDP data from a dataframe to a numpy array."""
        raise NotImplementedError

    def _join_state_action(self):
        """Join the state and action pairs together.

        Returns:
            list: Group of states and actions per datapoint.
        """
        zipped = []
        for i in range(len(self._reward_mdp_data)):
            state_array = self._state_mdp_data[i].tolist()
            action_array = self._action_mdp_data[i].tolist()
            zipped.append(state_array + action_array)

        return zipped

    def _bin_state_action_space(self, zipped):
        """Bin the state-action pairs.

        Args:
            zipped (list): Group of states and actions per datapoint.

        Returns:
            np.array: Binned state-action pairs.
        """
        return np.array(self.bin_states(zipped))

    def bin_states(self, states, idxs=None):
        """ Bin a list of states.

        Args:
            states (list[list]): State to bin.
            idxs (list): indexes of the state dimensions. This argument can be used if the state list contains only
                certain features (e.g. only actions).

        Returns:
            b_states (list): Binned state.
        """
        b_states = []
        for state in states:
            b_states.append(self.bin_state(state, idxs=idxs))
        return b_states

    def debin_states(self, b_states, idxs=None):
        """ Debin a list of binned states.

        Args:
            b_states (list[list]): Binned states to debin.
            idxs (list): indexes of the state dimensions. This argument can be used
                if the state list contains only certain features (e.g. only actions)

        Returns:
            states (list): Binned state.
        """
        states = []
        for b_state in b_states:
            states.append(self._debin_state(b_state, idxs=idxs))
        return states

    def bin_state(self, state, idxs=None):
        """Bin a singular state.

        The states are binned according to the number of bins
        of each feature.

        Args:
            state (list): State to bin.
            idxs (list): indexes of the state dimensions.
                This argument can be used if the state list contains
                only certain features (e.g. only actions).

        Returns:
            binned (list): Binned state.

        """
        if idxs == None:
            idxs = range(len(state))

        binned = []
        for i, value in zip(idxs, state):
            binned.append(
                np.digitize(
                    value,
                    [
                        n / self.bins[i] if n < self.bins[i] else 1.01
                        for n in range(1, self.bins[i] + 1)
                    ],
                )
            )
        return binned

    def _debin_state(self, b_state, idxs=None):
        """Debin a singular states.

        Args:
            b_state (list): Binned state to de-bin.

        Returns:
            list: Debinned state.
        """
        if idxs == None:
            idxs = range(len(b_state))

        state = []
        for i, value in zip(idxs, b_state):
            # Append middle point of the state bin
            try:
                state.append((value + 0.5) / self.bins[i])
            except:
                ipdb.set_trace()
        return state

    def _get_counts_and_rewards_per_bin(self, binned):
        """Create a dictionary of counts of datapoints per bin and sum the associated rewards.

        Args:
            binned (np.array): Binned state-action pairs.
        Returns:
            dict: Counts of datapoints per bin and sums the associated rewards.
        """
        raise NotImplementedError

    def _create_average_reward_matrix(self, bins_dict):
        """Create a sparse matrix of the state-action pairs and associated rewards from the inputted dataset.

        Args:
            bins_dict (dict): Dictionary of counts of datapoints per bin and sum of the associated rewards.

        Returns:
            sparse.COO: Sparse matrix of binned state-action pairs and their associated average reward.
        """

        raise NotImplementedError

    def _make_rewards_from_data(self):
        """Create sparse matrix of the state-action pairs and associated rewards from the inputted dataset.

        Returns:
            sparse.COO: Sparse matrix of binned state-action pairs and their associate average reward.
        """
        raise NotImplementedError

    def reset(self):
        """Reset environment.

        Returns:
            list: Randomised initial state.
        """
        sample_ix_point = np.random.choice(np.arange(len(self._state_mdp_data)))
        state = self._state_mdp_data[sample_ix_point].tolist()
        binned_state = self.bin_state(state)
        return binned_state

    def _get_state_to_action(self, binned):
        """Create a dictionary of states and their associated actions.

        Args:
            binned (np.array): Binned state-action pairs.
        Returns:
            state_to_action (dict): States and their associated actions.
        """
        state_to_action = {}
        final_dim = binned.shape[1] - 1
        binned_df = pd.DataFrame(binned)
        binned_df[final_dim] = binned_df[final_dim].apply(lambda x: [x])
        group_by_inds = [i for i in range(final_dim)]
        binned_df = (
            binned_df.groupby(group_by_inds).sum(numeric_only=False).reset_index()
        )
        binned_df[final_dim] = binned_df[final_dim].apply(lambda x: set(x))
        binned = np.array(binned_df)
        for ix, bin in enumerate(binned):
            state = ",".join(str(e) for e in bin[:-1])
            state_to_action[state] = bin[-1]

        return state_to_action

    def step(self, state, action):
        """Take a step in the environment.

        Args:
            state (list): Current state values of the agent.
            action (int): Action for agent to take.
        """
        raise NotImplementedError
