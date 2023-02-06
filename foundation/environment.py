from foundation.agent import Agent
import sparse
import numpy as np


class MDP:
    def __init__(self, mdp_data, discount_factor=0):
        """Initialise the MDP superclass."""
        self.average_rewards = None
        self.state_to_action = {}
        self.mdp_data = mdp_data
        self.initialise_env()

    def initialise_env(self):
        """Create the environment given the MDP information.
        """
        self.average_rewards = self.make_rewards_from_data()

    def make_rewards_from_data(self, num_bins=9):
        """Make the state-action reward table from dataset."""

        # Group state and action from the dataset
        zipped = []
        for i in range(len(self.mdp_data)):
            state_array = self.mdp_data["s"].iloc[i].tolist()
            action_array =  self.mdp_data["a"].iloc[i].tolist()
            zipped.append(state_array + action_array)

        # Create the bins
        binned = np.digitize(zipped, np.arange(0, 1 + 1/num_bins, step=1/num_bins).tolist(), right=True)

        #breakpoint()
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

        coords = []
        data = []

        # TODO change this for loop for efficiency
        for key, value in bins_dict.items():
            # print(key)
            d = []
            for i in key.split(" "):
                if len(i) > 0:
                    d+= [int(i)]
            # d = [eval(i) for i in key]
            coords.append(d)
            data.extend([value[1] / value[0]])

        coords = np.array(coords).T.tolist()
        return sparse.COO(coords, data)


    def reset(self):
        """Reset environment and return a randomised state.
        
        TODO: return only states that have happened
        """
        # Tested
        state = []
        for dim in self.average_rewards.shape:
            # TODO: make it clear to everyone why low=1 and not 0
            state += [np.random.randint(low=1, high=dim)]
        return state

    def step(self, state, action):
        """Take a step in the environment. Done means is the env terminated.
        Returns state, next state, reward, done.
        
        TODO: fix the way to query average rewards
        """
        reward = self.average_rewards[state[0],state[1],state[2],action]
        return state, state, reward, True


# # """Understanding sparse matrices"""
# #
# #
#
# coords = [[0, 0, 0, 5, 5, 5],
#           [0, 0, 0, 0, 0, 0]]
#
# data = [10, 20, 30, 40, 50, 60]
#
# input_sparse_matrix = sparse.COO(coords, data, shape=(6, 6))

#
# def reset_env(input_sparse_matrix):
#     """Reset environment and return a randomised state."""
#     state = []
#     for dim in input_sparse_matrix.shape:
#         # TODO: make it clear to everyone why low=1 and not 0
#         state += [np.random.randint(low=1, high=dim)]
#     return state
#
# print(reset_env(input_sparse_matrix))
#

# # RUN THE FOLLOWING:
#
# dummy_array = np.array([
#                        [[0.75, 0.25], [0.1], [0.1]],
#                        [[0.25, 0.75], [0.3], [0.2]],
#                        [[0.85, 0.1], [0.5], [0.6]],
#                        [[0.7, 1], [0.9], [1]],
#                        [[0.5, 0.4], [0.8], [0.7]]
#                         ], dtype=object)
#
# env = MDP(dummy_array)
#
# env.initialise_env()
#
# print(env.average_rewards.todense())


# # print(s.todense())
#
# """Understanding binning"""
#
#
# # np.array([[dim1], [dim2], [dim3]])
# arr1 = np.array([[0.75, 0.25], [0.25, 0.75], [0.85, 0.1], [0.26, 0.80]])
#
# dummy_array = np.array([
#                        [[0.75, 0.25], [0.1], 0.1],
#                        [[0.25, 0.75], [0.3], 0.2],
#                        [[0.85, 0.1], [0.5], 0.6],
#                        [[0.26, 0.80], [1], 0.9]], dtype=object)
#
# zipped = []
#
# for i in range(len(dummy_array)):
#     zipped.append(dummy_array[i, 0] + dummy_array[i, 1])
#
# # print(zipped)
#
# binned = np.digitize(zipped, [0, 0.5, 1], right=True)
#
# """count the number of occurences of each bin"""
# print(binned)
# bins_dict = {}
# for ix, bin in enumerate(binned):
#     reward = dummy_array[ix, 2]
#     bin = str(bin)
#     bins_dict[bin][0] = bins_dict.setdefault(bin, [0, 0])[0] + 1
#     bins_dict[bin][1] = bins_dict.setdefault(bin, [0, 0])[1] + reward
#
# print(bins_dict)
#
# coords = []
# data = []
#
# for key, value in bins_dict.items():
#     d = []
#     for char in key:
#         if char.isdigit():
#             d += [int(char)]
#
#     coords.append(d)
#     data.extend([value[1]/value[0]])
#
# print(coords)
# coords = np.array(coords).T.tolist()
#
# print(coords)
# print(data)
#
# s = sparse.COO(coords, data)
# print(s.todense())
#
#
# rewards_lists = np.array(list(bins_dict.values()), dtype=object)
#
# for element in rewards_lists:
#     avg_reward = np.mean(element)
#
# # print(rewards_lists)
# mean_rewards = np.mean(rewards_lists)
# # print(mean_rewards)
#
#     # bins_dict[bin] = bins_dict.get(bin, []) + [reward]
#
# # print(bins_dict)
# # print(bins_dict.items())
#
#
# """Sum with digitize"""
#
# coords = binned.T
# # print(coords)
# data = (dummy_array[:, 2])
# # print(data.shape)
#
# s = sparse.COO(coords, data)
# # print(s.todense())
#
# """Average the rewards"""
# for key, value in bins_dict.items():
#     d = []
#
#     for char in key:
#         if char.isdigit():
#             d += [int(char)]
#
# # print(d)
# # print(bins_dict)
#
#
# blocks = list(bins_dict.keys())
#
# # for block in blocks:
#
# # s[tuple(d)] /= value
#
# # print(s.todense())
#
#
#
#     # #d1, d2, d3 = tuple(map(tuple, arr))
#     # # print(np.array(key))
#     # print(s[d1, d2, d3])
#
#     # print(s[list(key)])
#     # print(s[2,1,1])
#
#
# # unique, counts = np.unique(binned, return_counts=True)
# # print(dict(zip(unique, counts)))
#
# # print(np.count_nonzero(binned == [2, 1, 1]))
#
# """Average the rewards"""
#
# # print((arr1.shape))
# # print((dummy_array[0:1].shape))
# # assert arr1 == dummy_array[0:1]
#
# # zipped = list(zip(dummy_array[:, 0],dummy_array[:, 1]))
#
# # print(sum(zipped, ()))
#
# # zipped = itertools.zip_longest(dummy_array[:, 0], dummy_array[:, 1])
#
# # zipped = list(map(list, zip(dummy_array[:, 0], dummy_array[:, 1])))
# #
# # print(sum(zipped[0], []))
#
# # zipped = list(zip(dummy_array[:, 0],
# #              dummy_array[:, 1]))
#
# # print(list(zipped.flatten(order='C')))
# # print(f"nack: {zipped}")
#
# # print(np.digitize(dummy_array[0:1][0], [0, 0.5, 1], right=True))
