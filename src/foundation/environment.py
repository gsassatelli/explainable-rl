# Import packages
import sparse
import numpy as np

from src.foundation.super_classes import MDP

class StrategicPricingMDP(MDP):
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
        self._average_rewards = None
        self.state_to_action = {}

        self._state_mdp_data = None
        self._action_mdp_data = None
        self._reward_mdp_data = None
        
        self.state_dim = self.dh.get_states().shape[1]
        ####### NEW
        # self.action_dim = self.dh.get_actions().shape[1]
        self.action_dim = len(self.dh.get_actions())
        #######

        if len(bins) != self.state_dim + 1:
            self.bins = [10] * (self.state_dim + 1)
        else:
            self.bins = bins

    def initialise_env(self):
        """Create the environment given the MDP information."""
        self._average_rewards = self._make_rewards_from_data()

    def _transform_df_to_numpy(self):
        """Transform the MDP data from a dataframe to a numpy array
        """
        self._state_mdp_data = self.dh.get_states().to_numpy()
        ######## NEW
        self._action_mdp_data = np.array(self.dh.get_actions())
        # self._action_mdp_data = self.dh.get_actions().to_numpy()
        ########
        self._reward_mdp_data = self.dh.get_rewards().to_numpy()

    def _join_state_action(self):
        """Join the state and action pairs together.
        Returns:
            list: Group of states and actions per datapoint.
        """
        zipped = []
        for i in range(len(self._reward_mdp_data)):
            state_array = self._state_mdp_data[i].tolist()
            for j in range(self.action_dim):
                action_array = self._action_mdp_data[j].tolist()
                zipped.append(state_array + [action_array])
        return zipped
    
    def _bin_state_action_space(self, zipped):
        """Bin the state-action pairs.
        Args:
            zipped (list): Group of states and actions per datapoint.
        Returns:
            np.array: Binned state-action pairs.
        """
        binned = []
        zipped = np.array(zipped)
        for i in range(zipped.shape[1]):
            binned.append(
                np.digitize(zipped[:, i], np.linspace(0, 1, self.bins[i])))
        return np.array(binned).T
    

    def _bin_state(self, state):
        """Bin a singular state.

        Args:
            state (list): State to bin.
        """
        binned = []
        for i in range(len(state)):
            binned.append(np.digitize(state[i], np.linspace(0, 1 + 1 / self.bins[i], self.bins[i])))
        return binned

    def _get_counts_and_rewards_per_bin(self, binned):
        """Create a dictionary of counts of datapoints per bin and sums the associated rewards.
        Args:
            binned (np.array): Binned state-action pairs.
        Returns:
            dict: dictionary of counts of datapoints per bin and sums the associated rewards.
        """

        bins_dict = {}
        self.state_to_action = {}

        # I want to go through binned with step_size = 10
        for ix, bin in enumerate(binned):
            if (ix % 9 == 0 and ix != 0) or 1 <= ix < 9 or ix >= len(binned)/10:
                continue
            # Go through each bin
            state_str = ",".join(str(e) for e in bin.tolist()[:-1])
            action = bin[-1]
            # Update state to action
            self.state_to_action.setdefault(state_str, set()).add(action)

            # update number of data points in the bin
            state_action_str = state_str+','+str(action)
            bins_dict[state_action_str][0] =\
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
        return sparse.COO(coords, data)

    def _make_rewards_from_data(self):
        """Create sparse matrix of the state-action pairs and associated rewards from the inputted dataset.
        Returns:
            sparse.COO: sparse matrix of binned state-action pairs and their associate average reward.
        """
        print("Create average rewards matrix")

        # Transform data for efficiency
        self._transform_df_to_numpy()

        # state_data = self._state_mdp_data.tolist()

        zipped = self._join_state_action()

        # Create the bins
        binned = self._bin_state_action_space(zipped)

        bins_dict = self._get_counts_and_rewards_per_bin(binned)

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
        next_state = self._find_next_state(state, action)
        done = False
        if next_state == None:
            done = True
        return state, next_state, reward, done

    def _find_next_state(self, state, action):
        """Lookup whether the next state exists in the state-action space matrix

        Args:
            state (list): Current state values of agent.
            action (int): Action for agent to take.

        Returns:
            list: next state for the agent to visit
        """
        index_next_state = tuple(state + [action])
        next_state_reward = self._average_rewards[index_next_state]
        if next_state_reward > 0.00001:
            next_state = state + [action]
        else:
            next_state = None
        
        return next_state

