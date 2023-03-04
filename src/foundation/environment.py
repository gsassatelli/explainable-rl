# Import packages
import sparse
import numpy as np


class MDP:
    """Defines and instantiates an MDP object.
    """
    __slots__ = ["dh", "_average_rewards", "_num_bins", "state_to_action", "bins_dict", "ix", "_state_mdp_data",
                 "_action_mdp_data", "_reward_mdp_data", "bins"]

    def __init__(self, dh):
        """Initialises the MDP superclass.
        Args:
            dh (DataHandler): Data handler object.
        """
        self._average_rewards = None
        self.state_to_action = {}
        self.dh = dh

        self._state_mdp_data = None
        self._action_mdp_data = None
        self._reward_mdp_data = None
        self._num_bins = 100
        self.bins = None

    def initialise_env(self):
        """Creates the environment given the MDP information."""
        self._average_rewards = self._make_rewards_from_data()

    def _transform_df_to_numpy(self):
        """Transforms the MDP data from a dataframe to a numpy array
        """
        self._state_mdp_data = self.dh.get_states().to_numpy()
        self._action_mdp_data = self.dh.get_actions().to_numpy()
        self._reward_mdp_data = self.dh.get_rewards().to_numpy()

    def _join_state_action(self):
        """Joins the state and action pairs together.
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
        """Bins the state-action pairs.
        Args:
            zipped (list): Group of states and actions per datapoint.
        Returns:
            np.array: Binned state-action pairs.
        """
        self.bins = np.arange(0, 1 + 1 / self._num_bins, step=1 / self._num_bins).tolist()
        return np.digitize(zipped, self.bins, right=True)

    def _get_counts_and_rewards_per_bin(self, binned):
        """Creates a dictionary of counts of datapoints per bin and sums the associated rewards. 
        Args:
            binned (np.array): Binned state-action pairs.
        Returns:
            dict: dictionary of counts of datapoints per bin and sums the associated rewards.
        """
        bins_dict = {}
        self.state_to_action = {}
        for ix, bin in enumerate(binned):
            bin = ",".join(str(e) for e in bin.tolist())
            # Update state to action
            self.state_to_action.setdefault(bin[:-2], set()).add(int(bin[-1]))

            # Update bins_dict
            bins_dict[bin][0] = bins_dict.setdefault(bin, [0, 0])[0] + 1
            reward = self._reward_mdp_data[ix]
            bins_dict[bin][1] += reward[0]
        return bins_dict

    def _create_average_reward_matrix(self, bins_dict):
        """Generates a sparse matrix of average rewards for each bin in the dataset.
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
        return sparse.COO(coords, data)

    def _make_rewards_from_data(self):
        """Creates sparse matrix of the state-action pairs and associated rewards from the inputted dataset.
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

        average_reward_matrix = self._create_average_reward_matrix(bins_dict)

        return average_reward_matrix

    def reset(self):
        """Resets environment.
        Returns:
            list: Randomised initial state.
        """
        sample_ix_point = np.random.choice(np.arange(len(self._state_mdp_data) + 1))
        state = self._state_mdp_data[sample_ix_point].tolist()
        binned_state = self._bin_state_action_space(state)
        return binned_state[0]

    def step(self, state, action):
        """Takes a step in the environment.
        Done flags means the environment terminated.
        Args:
            state (list): Current state values of agent.
            action (int): Action for agent to take.

        Returns:
            tuple: current state, action, next state, done flag.
        """
        reward = self._average_rewards[state[0], state[1], state[2], action]

        return state, state, reward, True
