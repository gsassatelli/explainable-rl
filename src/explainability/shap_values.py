import numpy as np
import random
import pandas as pd


class ShapValues:
    """ This class implements the SHAP values algorithm.
    """

    __slots__ = ["sample", "features", "env", "Q", "minmax_scalars",
                 "action", "number_of_samples", "binned_sample"]

    def __init__(self,
                 sample,
                 features,
                 env,
                 Q,
                 minmax_scalars,
                 action,
                 number_of_samples=500):
        """ Initialise the ShapValues class.

        Args:
            sample (np.array): Sample to explain.
            features (list): List of features.
            env (MDP): MDP object.
            Q (np.array): Q-table.
            minmax_scalars (list): List of minmax scalars.
            action (int): Action.
            number_of_samples (int): Number of samples to use.
        """
        self.sample = sample
        self.features = features
        self.env = env
        self.Q = Q
        self.minmax_scalars = minmax_scalars
        self.action = action
        self.number_of_samples = number_of_samples
        self.binned_sample = None

        # TODO: unit testing
        # TODO: parametrize number of samples
        # TODO: Check number of bins problem - should be 0-1?
        # TODO: how to use shap value function better, no re run of the algorithm
        # TODO: check warning in shap values
        # TODO: improve random sampling - time consuming

    def compute_shap_values(self):
        """ Compute the SHAP values for a given sample.
        """

        # Verify if sample length is correct
        print("Verify if sample length is correct")
        if not self.verify_sample_length():
            raise ValueError("The sample length is not correct.")

        # Normalize sample
        self.sample = self.normalize_sample()

        # Bin sample
        self.binned_sample = self.bin_sample()

        # Verify if cell has been visited
        print("Verify if selected cell has been visited")
        if not self.verify_cell_availability(self.binned_sample):
            raise ValueError("The cell has not been visited by the agent.")

        # Predict action
        print("Predict action")
        predicted_action = self.predict_action()

        # Loop over all state dimensions
        print("Compute shap values")
        shap_values = {}
        for shap_ft in range(len(self.features)):
            print("Compute shap values for feature: ", self.features[shap_ft])
            num_bins_per_shap_ft = self.env.bins[shap_ft]
            action_samples_plus = np.zeros(self.number_of_samples, dtype=int)
            action_samples_minus = np.zeros(self.number_of_samples, dtype=int)
            for sample in range(self.number_of_samples):
                verified_samples = False
                # Sample plus and minus samples
                while not verified_samples:
                    s_plus, s_minus = self.sample_plus_minus_samples(shap_ft, num_bins_per_shap_ft)
                    if not self.verify_cell_availability(s_plus) or not self.verify_cell_availability(s_minus):
                        verified_samples = False
                    else:
                        verified_samples = True
                # Find best Q values for 2 samples
                Q_state_plus = np.zeros(self.env.bins[-1])
                Q_state_minus = np.zeros(self.env.bins[-1])
                for a in range(self.env.bins[-1]):
                    index_plus = tuple(list(s_plus) + [a])
                    current_plus = self.Q[index_plus]
                    Q_state_plus[a] = current_plus
                    index_minus = tuple(list(s_minus) + [a])
                    current_minus = self.Q[index_minus]
                    Q_state_minus[a] = current_minus
                action_samples_plus[sample] = np.argmax(np.array(Q_state_plus))
                action_samples_minus[sample] = np.argmax(np.array(Q_state_minus))

            # Denorm actions
            denorm_action_samples_plus = self.get_denorm_actions(action_samples_plus)
            denorm_action_samples_minus = self.get_denorm_actions(action_samples_minus)

            # Compute difference between arrays
            difference = np.array(denorm_action_samples_plus) - np.array(denorm_action_samples_minus)

            # Compute mean
            mean_difference = round(np.mean(difference, axis=0), 4)

            # Append shap value for that feature
            shap_values.update({self.features[shap_ft]: mean_difference})

        return shap_values, predicted_action

    def verify_sample_length(self):
        """ This function verifies if the sample length is correct.
        """
        if len(self.sample) != len(self.features):
            return False
        return True

    def bin_sample(self):
        """ This function bins the sample.
        """
        state_dims = list(range(len(self.features)))
        binned_sample = self.env._bin_state(self.sample, idxs=state_dims)
        # binned_sample = [i-1 for i in binned_sample]
        return binned_sample

    def verify_cell_availability(self, binned_sample):
        """ This function verifies if the cell has been visited.

        Args:
            binned_sample (np.array): Binned sample.
        """
        num_actions = self.env.bins[-1]
        # The last element in the bins list is the number of actions
        for a in range(num_actions):
            index_current = tuple(list(binned_sample) + [a])
            if self.Q[index_current] != 0:
                # At least one action has been visited for this state has been visited
                return True
        return False

    def sample_plus_minus_samples(self, shap_ft, num_bins_per_shap_ft):
        """ This function samples the plus and minus samples.

        Args:
            shap_ft (int): Feature to explain.
            num_bins_per_shap_ft (int): Number of bins for the feature to explain.
        """
        shap_ft_random = random.randrange(num_bins_per_shap_ft)
        s_plus = np.zeros(len(self.sample))
        s_minus = np.zeros(len(self.sample))
        s_plus[shap_ft] = self.binned_sample[shap_ft]
        s_minus[shap_ft] = shap_ft_random
        for ft in range(len(self.features)):
            if shap_ft != ft:
                num_bins_ft = self.env.bins[ft]
                ft_random = random.randrange(num_bins_ft)
                s_plus[ft] = ft_random
                s_minus[ft] = ft_random
        s_plus = [int(i) for i in s_plus]
        s_minus = [int(i) for i in s_minus]
        return s_plus, s_minus

    def get_denorm_actions(self, actions):
        """Get actions denormalized values.

        Args:
            actions (list): List of actions.
        """
        denorm_actions = []
        if len(self.action) == 1:
            scalar = self.minmax_scalars[self.action[0]]
            for a in actions:
                # Divide dig actions by # bins of the action dimension
                # to get a value between 0 and 1
                denorm_a = scalar.inverse_transform(
                    a.reshape(-1, 1) / self.env.bins[-1])
                denorm_actions.append(denorm_a[0][0])

        else:
            for a in actions:
                denorm_a = self.action[a]
                denorm_actions.append(denorm_a)

        return denorm_actions

    def normalize_sample(self):
        """Normalize sample.
        """
        normalized_sample = []
        for idx, ft in enumerate(self.features):
            scalar = self.minmax_scalars[ft]
            idx_df = pd.DataFrame(np.array(self.sample[idx]).reshape(-1, 1), columns=[ft])
            norm_ft = scalar.transform(idx_df)
            normalized_sample.append(norm_ft[0][0])
        return normalized_sample

    def predict_action(self):
        """Predict action.
        """
        Q_state = np.zeros(self.env.bins[-1])
        for a in range(self.env.bins[-1]):
            index = tuple(list(self.binned_sample) + [a])
            current_q = self.Q[index]
            Q_state[a] = current_q
        binned_action = np.argmax(np.array(current_q))
        action = self.get_denorm_actions([binned_action])
        return action
