# Import packages
import sparse
import numpy as np
from line_profiler_pycharm import profile


class MDP:
    """Defines and instantiates an MDP object.
    """
    def __init__(self, mdp_data, discount_factor=0):
        """Initialises the MDP superclass.

        Args:
            mdp_data (pd.DataFrame): Dataframe containing states, actions and rewards.
            discount_factor (int, optional): Gamma value for the MDP. Defaults to 0.
        """
        self.average_rewards = None
        self.state_to_action = {}
        self.mdp_data = mdp_data
        self.num_bins = 100
        self.initialise_env()
        

    def initialise_env(self):
        """Creates the environment given the MDP information."""
        self.average_rewards = self.make_rewards_from_data()

    @profile
    def join_state_action(self):
        """Joins the state and action pairs together.

        Returns:
            list: Group of states and actions per datapoint.
        """
        zipped = []
        for i in range(len(self.mdp_data)):
            # The following 2 lines taking up most time
            state_array = self.mdp_data["s"].iloc[i].tolist()
            action_array =  self.mdp_data["a"].iloc[i].tolist()
            zipped.append(state_array + action_array)
        return zipped

    def bin_state_action_space(self, zipped):
        """Bins the state-action pairs.

        Args:
            zipped (list): Group of states and actions per datapoint.

        Returns:
            np.array: Binned state-action pairs.
        """
        self.bins = np.arange(0, 1 + 1/self.num_bins, step=1/self.num_bins).tolist()
        return np.digitize(zipped, self.bins, right=True)

    @profile
    def get_counts_and_rewards_per_bin(self, binned):
        """Creates a dictionary of counts of datapoints per bin and sums the associated rewards. 

        Args:
            binned (np.array): Binned state-action pairs.

        Returns:
            dict: dictionary of counts of datapoints per bin and sums the associated rewards.
        """
        bins_dict = {}
        self.state_to_action = {}
        for ix, bin in enumerate(binned):
            # Following line taking up a lot time
            reward = self.mdp_data["r"].iloc[ix]
            # update state to action
            state = ",".join([str(s) for s in bin[:-1]])
            self.state_to_action.setdefault(state, set()).add(bin[-1])
            
            # update bin_dict
            # Following line taking up a little time. Must be changed anyway.
            bin = str(bin).replace("[", "").replace("]", "")
            bins_dict[bin][0] = bins_dict.setdefault(bin, [0, 0])[0] + 1
            bins_dict[bin][1] += reward[0]
        return bins_dict

    def create_average_reward_matrix(self, bins_dict):
        """Generates a sparse matrix of average rewards for each bin in the dataset.

        Args:
            bins_dict (dict): dictionary of counts of datapoints per bin and sums the associated rewards.

        Returns:
            sparse.COO: sparse matrix of binned state-action pairs and their associate average reward.
        """
        coords = []
        data = []

        # TODO change this for loop for efficiency
        for key, value in bins_dict.items():
            d = []
            for i in key.split(" "):
                if len(i) > 0:
                    d+= [int(i)]
 
            coords.append(d)
            data.extend([value[1] / value[0]])

        coords = np.array(coords).T.tolist()
        return sparse.COO(coords, data)

    @profile
    def make_rewards_from_data(self):
        """Creates sparse matrix of the state-action pairs and associated rewards from the inputted dataset.

        Returns:
            sparse.COO: sparse matrix of binned state-action pairs and their associate average reward.
        """
        print("Create average rewards matrix")
        zipped = self.join_state_action()

        # Create the bins
        binned = self.bin_state_action_space(zipped)
        # np.digitize(zipped, np.arange(0, 1 + 1/num_bins, step=1/num_bins).tolist(), right=True)

        bins_dict = self.get_counts_and_rewards_per_bin(binned)

        average_reward_matrix = self.create_average_reward_matrix(bins_dict)

        return average_reward_matrix
  
    def reset(self):
        """Resets environment.

        Returns:
            list: Randomised initial state.
        """
        state = self.mdp_data['s'].sample().values.tolist()
        binned_state = self.bin_state_action_space(state)
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
        reward = self.average_rewards[state[0],state[1],state[2],action]
        return state, state, reward, True
