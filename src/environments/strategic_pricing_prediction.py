from library import *

# Import functions
from src.environments.strategic_pricing import StrategicPricing


class StrategicPricingPredictionMDP(StrategicPricing):
    """Environment for Strategic Pricing (prediction task)."""

    def __init__(self, dh, bins=None, verbose=False):
        """Initialise Strategic Pricing Environment.

        Args:
            dh (DataHandler): Data Handler instance.
            verbose (bool): Whether print statements about the program flow should be displayed.
        """
        super().__init__(dh=dh, bins=bins)
        self._verbose = verbose
        self.initialise_env()

    def _get_counts_and_rewards_per_bin(self, binned):
        """Create a dictionary of counts of datapoints per bin and sum the associated rewards.

        Args:
            binned (np.array): Binned state-action pairs.

        Returns:
            dict: Dictionary of counts of datapoints per bin and sums the associated rewards.
        """
        bins_dict = {}
        for ix, bin in enumerate(binned):
            state_str = ",".join(str(e) for e in bin.tolist()[:-1])
            action = bin[-1]
            state_action_str = state_str + "," + str(action)
            bins_dict[state_action_str][0] = (
                bins_dict.setdefault(state_action_str, [0, 0])[0] + 1
            )
            reward = self._reward_mdp_data[ix]
            bins_dict[state_action_str][1] += reward[0]
        return bins_dict

    def _make_rewards_from_data(self):
        """Create sparse matrix of the state-action pairs and associated rewards from the inputted dataset.

        Returns:
            sparse.COO: Sparse matrix of binned state-action pairs and their associate average reward.
        """
        if self._verbose:
            print("Create average rewards matrix")

        # Transform data for efficiency
        self._transform_df_to_numpy()

        zipped = self._join_state_action()

        # Create the bins
        binned = self._bin_state_action_space(zipped)

        self.bins_dict = self._get_counts_and_rewards_per_bin(binned)
        self.state_to_action = self._get_state_to_action(binned)
        average_reward_matrix = self._create_average_reward_matrix(self.bins_dict)

        return average_reward_matrix

    def step(self, state, action):
        """Take a step in the environment.

        Note that the last element (the Done flag) of the return tuple is always True
        as the prediction problem requires single step episodes for which the Done flag
        is always True.

        Args:
            state (list): Current state values of agent.
            action (int): Action for agent to take.

        Returns:
            tuple: Current state, action, next state, done flag.
        """
        index = tuple(list(state) + [action])
        reward = self._average_rewards[index]

        return state, state, reward, True

    def _create_average_reward_matrix(self, bins_dict):
        """Create a sparse matrix of the state-action pairs and associated rewards from the inputted dataset.

        Args:
            bins_dict (dict): Counts of datapoints per bin and sum of the associated rewards.

        Returns:
            sparse.COO: Sparse matrix of binned state-action pairs and their associated average reward.
        """
        coords = []
        data = []

        for key, value in bins_dict.items():
            d = [int(i) for i in key.split(",")]
            coords.append(d)
            data.extend([value[1] / value[0]])

        coords = np.array(coords).T.tolist()

        return sparse.COO(coords, data, shape=tuple(self.bins))

    def _transform_df_to_numpy(self):
        self._state_mdp_data = self.dh.get_states().to_numpy()
        self._action_mdp_data = self.dh.get_actions().to_numpy()
        self._reward_mdp_data = self.dh.get_rewards().to_numpy()
