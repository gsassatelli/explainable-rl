from library import *

# Import functions
from src.environments.strategic_pricing import StrategicPricing


class StrategicPricingSuggestionMDP(StrategicPricing):
    """Environment for Strategic Pricing (suggestion task)."""

    def __init__(self, dh, bins=None, verbose=False):
        """Initialise Strategic Pricing Environment."""
        super().__init__(dh=dh, bins=bins)
        self._verbose = verbose
        self.initialise_env()

    def _get_counts_and_rewards_per_bin(self, binned):
        """Create a dictionary of counts of datapoints per bin and sum the associated rewards.

        Args:
            binned (np.array): Binned state-action pairs.
        Returns:
            dict: Counts of datapoints per bin and sums the associated rewards.
        """

        bins_dict = {}
        for ix, bin in enumerate(binned):
            state_str = ",".join(str(e) for e in bin.tolist())
            bins_dict[state_str][0] = \
                bins_dict.setdefault(state_str, [0, 0])[0] + 1
            reward = self._reward_mdp_data[ix]
            bins_dict[state_str][1] += reward[0]

        return bins_dict

    def _make_rewards_from_data(self):
        """Create sparse matrix of the state-action pairs and associated rewards from the inputted dataset.

        Returns:
            sparse.COO: Binned state-action pairs and their associate average reward.
        """
        if self._verbose:
            print("Create average rewards matrix")

        # Transform data for efficiency
        self._transform_df_to_numpy()

        state_data = self._state_mdp_data.tolist()

        # Create the bins
        binned = self._bin_state_action_space(state_data)

        self.bins_dict = self._get_counts_and_rewards_per_bin(binned)
        self.state_to_action = self._get_state_to_action(binned)
        average_reward_matrix = self._create_average_reward_matrix(self.bins_dict)

        return average_reward_matrix

    def step(self, state, action):
        """Take a step in the environment.

        Done flag set to True means that the environment terminated.

        Args:
            state (list): Current state values of agent.
            action (int): Action for agent to take.

        Returns:
            tuple: Current state, action, next state, done flag.
        """
        # TODO: input index to find_next_state
        index = tuple(list(state)[:-1] + [action])

        reward = self._average_rewards[index]
        next_state, done = self._find_next_state(state, action)
        return state, next_state, reward, done

    def _find_next_state(self, state, action):
        """Lookup whether the next state exists in the state-action space matrix.

        Args:
            state (list): Current state values of agent.
            action (int): Action for agent to take.

        Returns:
            list: Next state for the agent to visit.
            bool: Whether the environment has terminated.
        """
        index = list(state) + [action]
        state_action_str = ",".join(str(e) for e in index)
        if state_action_str in self.bins_dict:
            next_state = state[:-1] + [action]
            done = False
        else:
            next_state = state
            done = True

        return next_state, done

    def _create_average_reward_matrix(self, bins_dict):
        """Create a sparse matrix of the state-action pairs and associated rewards from the inputted dataset.

        Args:
            bins_dict (dict): Counts of datapoints per bin and sum of the associated rewards.

        Returns:
            sparse.COO: Binned state-action pairs and their associated average reward.
        """

        coords = []
        data = []

        for key, value in bins_dict.items():
            d = [int(i) for i in key.split(",")]
            coords.append(d)
            data.extend([value[1] / value[0]])

        coords = np.array(coords).T.tolist()

        return sparse.COO(coords, data, shape=tuple(self.bins[:-1]))
        # TODO: make the bins dimensions work

    def _transform_df_to_numpy(self):
        self._state_mdp_data = self.dh.get_states().to_numpy()
        self._action_mdp_data = np.array(self.dh.get_action_labels())
        self._reward_mdp_data = self.dh.get_rewards().to_numpy()
