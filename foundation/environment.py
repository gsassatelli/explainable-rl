import agent

class MDP:
    def _init_(self, dataset, discount_factor):
        """Initialise the MDP superclass."""
        self.average_rewards = None
        self.dataset = dataset
        self.initialise_env()
        pass

    def initialise_env(self):
        """Create the environment given the MDP information.

        TODO: tricky part"""
        self.average_rewards = self.make_rewards_from_data()
        pass

    def make_rewards_from_data(self):
        """Make the state-action reward table from dataset."""
        pass

    def reset_env(self):
        """Reset environment and return a randomised state."""

        return []
        pass

    def step(self, state, action):
        """Take a step in the environment. Done means is the env terminated.

        Returns state, next state, reward, done."""


"""Understanding sparse matrices"""

import sparse

coords = [[0, 0, 0, 5, 5, 5],
          [0, 0, 0, 0, 0, 0]]

data = [10, 20, 30, 40, 50, 60]

s = sparse.COO(coords, data, shape=(6, 6))

# print(s.todense())

"""Understanding binning"""
import numpy as np
import itertools

# np.array([[dim1], [dim2], [dim3]])
arr1 = np.array([[0.75, 0.25], [0.25, 0.75], [0.85, 0.1], [0.26, 0.80]])

dummy_array = np.array([
                       [[0.75, 0.25], [0.1], 0.1],
                       [[0.25, 0.75], [0.3], 0.2],
                       [[0.85, 0.1], [0.5], 0.6],
                       [[0.26, 0.80], [1], 0.9]])

zipped = []

for i in range(len(dummy_array)):
    zipped.append(dummy_array[i, 0] + dummy_array[i, 1])

# print(zipped)

binned = np.digitize(zipped, [0, 0.5, 1], right=True)

"""count the number of occurences of each bin"""
bins_dict = {}
for ix, bin in enumerate(binned):
    bin = str(bin)
    bins_dict[bin] = bins_dict.get(bin, 0) + 1

# print(bins_dict.items())


"""Sum with digitize"""

coords = binned.T
# print(coords)
data = (dummy_array[:, 2])
# print(data.shape)

s = sparse.COO(coords, data)
print(s.todense())

"""Average the rewards"""
for key, value in bins_dict.items():
    # s[np.array(key)] = s[np.array(key)] / value

    d = []

    for char in key:
        if char.isdigit():
            d += [int(char)]

    s[tuple(d)] /= value

print(s.todense())

    # #d1, d2, d3 = tuple(map(tuple, arr))
    # # print(np.array(key))
    # print(s[d1, d2, d3])

    # print(s[list(key)])
    # print(s[2,1,1])


# unique, counts = np.unique(binned, return_counts=True)
# print(dict(zip(unique, counts)))

# print(np.count_nonzero(binned == [2, 1, 1]))

"""Average the rewards"""

# print((arr1.shape))
# print((dummy_array[0:1].shape))
# assert arr1 == dummy_array[0:1]

# zipped = list(zip(dummy_array[:, 0],dummy_array[:, 1]))

# print(sum(zipped, ()))

# zipped = itertools.zip_longest(dummy_array[:, 0], dummy_array[:, 1])

# zipped = list(map(list, zip(dummy_array[:, 0], dummy_array[:, 1])))
#
# print(sum(zipped[0], []))

# zipped = list(zip(dummy_array[:, 0],
#              dummy_array[:, 1]))

# print(list(zipped.flatten(order='C')))
# print(f"nack: {zipped}")

# print(np.digitize(dummy_array[0:1][0], [0, 0.5, 1], right=True))