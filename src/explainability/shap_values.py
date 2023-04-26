import matplotlib.pyplot as plt

from library import *


class ShapValues:
    """SHAP Values class."""

    def __init__(self, engine):
        """Initialise the ShapValues class.

        Args:
            engine (Engine): Engine object.
        """
        self.sample = None
        self.features = engine.dh.state_labels
        self.env = engine.env
        self.Q = engine.agent.Q
        self.minmax_scalars = engine.dh.minmax_scalars
        self.action = engine.dh.action_labels
        self.number_of_samples = engine.hyperparameters["explainability"][
            "shap_num_samples"
        ]
        self.binned_sample = None
        self.verbose = engine.hyperparameters["program_flow"]["verbose"]

    def compute_shap_values(self, sample):
        """Compute the SHAP values for a given sample.

        Args:
            sample (list): List with the sample to compute the SHAP values.

        Returns:
            shap_values (dict): Dictionary with the shap values for each feature.
            predicted_action (int): Predicted action.
        """
        self.sample = sample

        # Verify if sample length is correct
        if self.verbose:
            print("Verify if sample length is correct")
        if not self.verify_sample_length():
            raise ValueError("The sample length is not correct.")

        # Normalize sample
        self.sample = self.normalize_sample()

        # Bin sample
        self.binned_sample = self.bin_sample()

        # Verify if sample is an outlier
        if self.verbose:
            print("Verify if sample is an outlier")
        if self.verify_outliers(self.binned_sample):
            raise ValueError("The sample is an outlier.")

        # Verify if cell has been visited
        if self.verbose:
            print("Verify if selected cell has been visited")
        if not self.verify_cell_availability(self.binned_sample):
            raise ValueError("The cell has not been visited by the agent.")

        # Predict action
        if self.verbose:
            print("Predict action")
        predicted_action = self.predict_action()

        # Loop over all state dimensions
        if self.verbose:
            print("Compute shap values")
        shap_values = {}
        for shap_ft in range(len(self.features)):
            if self.verbose:
                print("Compute shap values for feature: ", self.features[shap_ft])
            num_bins_per_shap_ft = self.env.bins[shap_ft]
            action_samples_plus = np.zeros(self.number_of_samples, dtype=int)
            action_samples_minus = np.zeros(self.number_of_samples, dtype=int)
            for sample in range(self.number_of_samples):
                verified_samples = False

                # Sample plus and minus samples
                while not verified_samples:
                    s_plus, s_minus = self.sample_plus_minus_samples(
                        shap_ft, num_bins_per_shap_ft
                    )
                    if not self.verify_cell_availability(
                        s_plus
                    ) or not self.verify_cell_availability(s_minus):
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
            difference = np.array(denorm_action_samples_plus) - np.array(
                denorm_action_samples_minus
            )

            # Compute mean
            mean_difference = round(np.mean(difference, axis=0), 4)

            # Append shap value for that feature
            shap_values.update({self.features[shap_ft]: mean_difference})

        self.shap_values = shap_values

        return shap_values, predicted_action

    def verify_sample_length(self):
        """Verify whether the sample length is correct.

        Returns:
            bool: True if the sample length is correct, False otherwise.
        """
        if len(self.sample) != len(self.features):
            return False
        return True

    def bin_sample(self):
        """Bin the samples.

        Returns:
            binned_sample (np.array): Binned sample.
        """
        state_dims = list(range(len(self.features)))
        binned_sample = self.env.bin_state(self.sample, idxs=state_dims)
        return binned_sample

    def verify_cell_availability(self, binned_sample):
        """Verify whether the cell has been visited.

        Args:
            binned_sample (np.array): Binned sample.
        Returns:
            bool: True if the cell has been visited, False otherwise.
        """
        num_actions = self.env.bins[-1]
        # The last element in the bins list is the number of actions
        for a in range(num_actions):
            index_current = tuple(list(binned_sample) + [a])
            if self.Q[index_current] != 0:
                # At least one action has been visited for this state has been visited
                return True
        return False

    def verify_outliers(self, binned_sample):
        """Verify whether the sample is an outlier.

        Args:
            binned_sample (np.array): Binned sample.

        Returns:
            bool: True if the sample is an outlier, False otherwise.
        """
        for ft in range(len(self.features)):
            if binned_sample[ft] >= self.env.bins[ft] or binned_sample[ft] < 0:
                return True
        return False

    def sample_plus_minus_samples(self, shap_ft, num_bins_per_shap_ft):
        """Sample the plus and minus samples.

        Args:
            shap_ft (int): Feature to explain.
            num_bins_per_shap_ft (int): Number of bins for the feature to explain.

        Returns:
            s_plus (np.array): Plus sample.
            s_minus (np.array): Minus sample.
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

        Returns:
            denorm_actions (list): List of denormalized actions.
        """
        denorm_actions = []
        if len(self.action) == 1:
            scalar = self.minmax_scalars[self.action[0]]
            for a in actions:
                # Divide dig actions by # bins of the action dimension to get a value between 0 and 1
                denorm_a = scalar.inverse_transform(
                    a.reshape(-1, 1) / self.env.bins[-1]
                )
                denorm_actions.append(denorm_a[0][0])

        else:
            for a in actions:
                denorm_a = self.action[a]
                denorm_actions.append(denorm_a)

        return denorm_actions

    def normalize_sample(self):
        """Normalize sample.

        Returns:
            normalized_sample (list): Normalized sample.
        """
        normalized_sample = []
        for idx, ft in enumerate(self.features):
            scalar = self.minmax_scalars[ft]
            idx_df = pd.DataFrame(
                np.array(self.sample[idx]).reshape(-1, 1), columns=[ft]
            )
            norm_ft = scalar.transform(idx_df)
            normalized_sample.append(norm_ft[0][0])
        return normalized_sample

    def predict_action(self):
        """Predict action.

        Returns:
            action (list): Predicted action.
        """
        Q_state = np.zeros(self.env.bins[-1])
        for a in range(self.env.bins[-1]):
            index = tuple(list(self.binned_sample) + [a])
            current_q = self.Q[index]
            Q_state[a] = current_q
        binned_action = np.argmax(np.array(current_q))
        action = self.get_denorm_actions([binned_action])
        return round(action[0], 4)

    def plot_shap_values(
        self, sample, shap_values, predicted_action, fig_name=None, savefig=False
    ):
        """Plot shap values.

        Args:
            sample (list): Sample.
            shap_values (dict): Shap values.
            predicted_action (float): Predicted action.
            fig_name (str): Figure name.
            savefig (bool): Whether to save the figure or not.
        """
        # Sort values
        sorted_shap_values = sorted(shap_values.items(), key=lambda x: x[1])

        # Get values
        features = [i[0] for i in sorted_shap_values]
        values = [i[1] for i in sorted_shap_values]
        colors = ["red" if i < 0 else "green" for i in values]

        # Plot values
        plt.grid(zorder=0)
        plt.barh(features, values, color=colors, zorder=3)
        plt.title(f"Shap values for {sample} - Action: {predicted_action}")
        plt.tight_layout()

        if savefig:
            plt.savefig(fig_name, dpi=600)

        plt.show()
