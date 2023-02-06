from foundation.agent import Agent
import sparse
import numpy as np


class MDP:
    def __init__(self, mdp_data, discount_factor=0):
        """Initialise the MDP superclass."""
        self.average_rewards = None
        self.state_to_action = {}
        self.mdp_data = mdp_data
        self.num_bins = 9
        self.initialise_env()
        

    def initialise_env(self):
        """Create the environment given the MDP information.
        """
        self.average_rewards = self.make_rewards_from_data()

    def join_state_action(self):
        zipped = []
        for i in range(len(self.mdp_data)):
            state_array = self.mdp_data["s"].iloc[i].tolist()
            action_array =  self.mdp_data["a"].iloc[i].tolist()
            zipped.append(state_array + action_array)
        return zipped

    def bin_state_action_space(self, zipped):
        return np.digitize(zipped, np.arange(0, 1 + 1/self.num_bins, step=1/self.num_bins).tolist(), right=True)

    def get_counts_and_rewards_per_bin(self, binned):
        bins_dict = {}
        self.state_to_action = {}
        for ix, bin in enumerate(binned):
            reward = self.mdp_data["r"].iloc[ix]
            # update state to action
            state = ",".join([str(s) for s in bin[:-1]])
            self.state_to_action.setdefault(state, set()).add(bin[-1])
            
            # update bin_dict
            bin = str(bin).replace("[", "").replace("]", "")
            bins_dict[bin][0] = bins_dict.setdefault(bin, [0, 0])[0] + 1
            bins_dict[bin][1] += reward[0]
        return bins_dict

    def create_average_reward_matrix(self, bins_dict):
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

    def make_rewards_from_data(self):
        
        """Make the state-action reward table from dataset."""

        zipped = self.join_state_action()

        # Create the bins
        binned = self.bin_state_action_space(zipped)
        # np.digitize(zipped, np.arange(0, 1 + 1/num_bins, step=1/num_bins).tolist(), right=True)

        bins_dict = self.get_counts_and_rewards_per_bin(binned)

        average_reward_matrix = self.create_average_reward_matrix(bins_dict)

        return average_reward_matrix
  
    def reset(self):
        """Reset environment and return a randomised state.
        
        TODO: return only states that have happened
        """
        state = self.mdp_data['s'].sample().values.tolist()
 
        return state

    def step(self, state, action):
        """Take a step in the environment. Done means is the env terminated.
        Returns state, next state, reward, done.
        
        TODO: fix the way to query average rewards
        """
        reward = self.average_rewards[state[0],state[1],state[2],action]
        return state, state, reward, True

