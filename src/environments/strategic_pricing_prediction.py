from library import *

# Import Environment
from src.foundation.environment import MDP


class StrategicPricingPredictionMDP(MDP):
    """Defines and instantiates the MDP object for Strategic Pricing.
    """

    __slots__ = ["dh", "_average_rewards", "num_bins", "state_to_action", "bins_dict", "ix", "_state_mdp_data",
                 "_action_mdp_data", "_reward_mdp_data", "bins", 'state_dim', 'action_dim']

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
        self.state_dim = self.dh.get_states().shape[1]
        self.action_dim = self.dh.get_actions().shape[1]
        if len(bins) != self.state_dim + self.action_dim:
            self.bins = [10] * (self.state_dim + self.action_dim)
        else:
            self.bins = bins

        self.initialise_env()

    def initialise_env(self):
        """Create the environment given the MDP information."""
        self._average_rewards = self._make_rewards_from_data()

    def _transform_df_to_numpy(self):
        """Transform the MDP data from a dataframe to a numpy array
        """
        self._state_mdp_data = self.dh.get_states().to_numpy()
        self._action_mdp_data = self.dh.get_actions().to_numpy()
        self._reward_mdp_data = self.dh.get_rewards().to_numpy()

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
            certain features (e.g. only actions)

        Returns:
            b_states (list): Binned state

        """
        b_states = []
        for state in states:
            b_states.append(
                self._bin_state(state, idxs=idxs)
            )
        return b_states

    def debin_states(self, b_states, idxs=None):
        """ Debin a list of binned states.

        Args:
            b_states (list[list]): Binned states to debin.
            idxs (list): indexes of the state dimensions
            This argument can be used if the state list contains
            only certain features (e.g. only actions)
        Returns:
            states (list): Binned state
        """
        states = []
        for b_state in b_states:
            states.append(
                self._debin_state(b_state, idxs=idxs)
            )
        return states

    def _bin_state(self, state, idxs=None):
        """Bin a singular state.

        The states are binned according to the number of bins
        of each feature.

        Args:
            state (list): State to bin.
            idxs (list): indexes of the state dimensions
                This argument can be used if the state list contains
                only certain features (e.g. only actions)
        
        Returns:
            binned (list): Binned state

        """
        if idxs == None:
            idxs = range(len(state))

        binned = []
        for i, value in zip(idxs, state):
            binned.append(
                np.digitize(
                    value,
                    [n / self.bins[i] if n < self.bins[i] else 1.01 \
                     for n in range(1, self.bins[i] + 1)]
                )
            )
        return binned

    def _debin_state(self, b_state, idxs=None):
        """ Debin a singular states.
        Returns middle point of the bin.
        
        Args:
            b_state (list): Binned state to de-bin
        """
        if idxs == None:
            idxs = range(len(b_state))

        state = []
        for i, value in zip(idxs, b_state):
            # append middle point of the state bin
            try:
                state.append((value + 0.5) / self.bins[i])
            except:
                ipdb.set_trace()
        return state

    def _debin_state(self, b_state, idxs=None):
        """ Debin a singular states.
        Returns middle point of the bin.

        Args:
            b_state (list): Binned state to de-bin
        """
        if idxs == None:
            idxs = range(len(b_state))

        state = []
        for i, value in zip(idxs, b_state):
            # append middle point of the state bin
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
            dict: dictionary of counts of datapoints per bin and sums the associated rewards.
        """

        bins_dict = {}
        for ix, bin in enumerate(binned):
            state_str = ",".join(str(e) for e in bin.tolist()[:-1])
            action = bin[-1]
            # update number of data points in the bin
            state_action_str = state_str + ',' + str(action)
            bins_dict[state_action_str][0] = \
                bins_dict.setdefault(state_action_str, [0, 0])[0] + 1
            # update total reward in the bin
            reward = self._reward_mdp_data[ix]
            bins_dict[state_action_str][1] += reward[0]
        return bins_dict

    def _create_average_reward_matrix(self, bins_dict):
        """Create a sparse matrix of the state-action pairs and associated rewards from the inputted dataset.

        Args:
            bins_dict (dict): dictionary of counts of datapoints per bin and sums the associated rewards.
        Returns:
            sparse.COO: sparse matrix of binned state-action pairs and their associate average reward.
        """

        coords = []
        data = []

        for key, value in bins_dict.items():
            d = [int(i) for i in key.split(",")]
            coords.append(d)
            data.extend([value[1] / value[0]])

        coords = np.array(coords).T.tolist()

        return sparse.COO(coords, data, shape=tuple(self.bins))

    def _make_rewards_from_data(self):
        """Create sparse matrix of the state-action pairs and associated rewards from the inputted dataset.

        Returns:
            sparse.COO: sparse matrix of binned state-action pairs and their associate average reward.
        """
        print("Create average rewards matrix")

        # Transform data for efficiency
        self._transform_df_to_numpy()

        zipped = self._join_state_action()

        # Create the bins
        binned = self._bin_state_action_space(zipped)

        bins_dict = self._get_counts_and_rewards_per_bin(binned)
        self.state_to_action = self._get_state_to_action(binned)
        average_reward_matrix = self._create_average_reward_matrix(bins_dict)

        return average_reward_matrix

    def reset(self):
        """Reset environment.

        Returns:
            list: Randomised initial state.
        """
        sample_ix_point = np.random.choice(np.arange(len(self._state_mdp_data)))
        state = self._state_mdp_data[sample_ix_point].tolist()
        binned_state = self._bin_state(state)
        return binned_state

    def step(self, state, action):
        """Take a step in the environment.
        Done flags means the environment terminated.

        Args:
            state (list): Current state values of agent.
            action (int): Action for agent to take.
        Returns:
            tuple: current state, action, next state, done flag.
        """
        index = tuple(list(state) + [action])
        reward = self._average_rewards[index]

        return state, state, reward, True

    def _get_state_to_action(self, binned):
        """Create a dictionary of states and their associated actions.

        Args:
            binned (np.array): Binned state-action pairs.
        Returns:
            state_to_action (dict): dictionary of states and their associated actions.
        """
        state_to_action = {}
        final_dim = binned.shape[1] - 1
        binned_df = pd.DataFrame(binned)
        binned_df[final_dim] = binned_df[final_dim].apply(lambda x: [x])
        group_by_inds = [i for i in range(final_dim)]
        binned_df = binned_df.groupby(group_by_inds).sum(numeric_only=False). \
            reset_index()
        binned_df[final_dim] = binned_df[final_dim].apply(lambda x: set(x))
        binned = np.array(binned_df)
        for ix, bin in enumerate(binned):
            state = ",".join(str(e) for e in bin[:-1])
            state_to_action[state] = bin[-1]

        return state_to_action
