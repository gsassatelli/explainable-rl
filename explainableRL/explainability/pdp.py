from explainableRL.foundation.library import *


class PDP:
    """Partial Dependence Plots class."""

    def __init__(self, engine):
        """Initialise PDP class.

        Args:
            engine (Engine): Engine object containing the trained agent.
        """
        self.Q = engine.agent.Q
        self.Q_num_samples = engine.agent.Q_num_samples
        self._bins = engine.env.bins
        self._minmax_scalars = engine.dh.minmax_scalars
        self._action_labels = engine.dh.action_labels
        self._state_labels = engine.dh.state_labels
        self._dig_state_actions = []
        self._dig_state_actions_std = []
        self._dig_state_actions_samples = []
        self._denorm_actions = []
        self._denorm_states = []
        self._bins_per_dim = []
        self._Q_array = []

    def build_data_for_plots(self):
        """Prepare data to build PDP plots.
        """
        self._get_digitized_pdp()
        self._get_denorm_actions()
        self._get_denorm_states()

    def _get_digitized_pdp(self):
        """Compute average Q-value per each state-action pair.
        Marginal effect of the state-action pair averaging other state dimensions.
        """
        Q_array = self.Q.todense()
        self._Q_array = Q_array
        shape_Q = Q_array.shape
        num_dims = len(shape_Q)
        num_states = num_dims - 1  # last dimension is action
        self._bins_per_dim = [shape_Q[i] for i in range(num_dims)]
        set_states = set(list(range(num_states)))

        Q_num_samples_array = self.Q_num_samples.todense()

        # For each state dimension
        for dim in range(num_states):
            states_to_avg = tuple(set_states - set([dim]))
            Q_avg = np.mean(Q_array, axis=states_to_avg)
            Q_std = np.std(Q_array, axis=states_to_avg)
            Q_num_samples_sum = np.sum(Q_num_samples_array, axis=states_to_avg)
            # Select action with the highest avg Q value
            dig_actions = np.argmax(Q_avg, axis=-1)
            dig_actions_std = np.array(
                [Q_std[idx][action] for idx, action in enumerate(dig_actions)]
            )
            dig_actions_samples = np.array(
                [
                    Q_num_samples_sum[idx][action]
                    for idx, action in enumerate(dig_actions)
                ]
            )
            # Add the total number of samples per state bin
            dig_actions_samples = np.concatenate(
                [
                    np.expand_dims(dig_actions_samples, -1),
                    np.expand_dims(Q_num_samples_sum.sum(-1), -1),
                ],
                axis=-1,
            )

            self._dig_state_actions.append(dig_actions)
            self._dig_state_actions_std.append(dig_actions_std)
            self._dig_state_actions_samples.append(dig_actions_samples)

    def _get_denorm_actions(self):
        """Get actions denormalized values."""
        if len(self._action_labels) == 1:
            # The action column comes from the dataset
            scaler = self._minmax_scalars[self._action_labels[0]]
            for dig_actions in self._dig_state_actions:
                # Divide dig actions by # bins of the action dimension to get a value between 0 and 1
                denorm_action = scaler.inverse_transform(
                    dig_actions.reshape(-1, 1) / self._bins_per_dim[-1]
                )
                self._denorm_actions.append(denorm_action)

        else:
            # The action are imputed by the user
            for dim in self._dig_state_actions:
                denorm_action = [self._action_labels[i] for i in dim]
                self._denorm_actions.append(denorm_action)

    def _get_denorm_states(self):
        """Get states denormalized values."""
        num_states = len(self._state_labels)
        for i in range(num_states):
            n_bins = self._bins_per_dim[i]
            # Divide by number of bins to get a value between [0,1] which can then be inputted into the scaler
            dig_values = np.array(list(range(n_bins))) / n_bins
            scaler = self._minmax_scalars[self._state_labels[i]]
            denorm_state = scaler.inverse_transform(dig_values.reshape(-1, 1))
            self._denorm_states.append(denorm_state)

    def plot_pdp(self, feature, fig_name=None, savefig=True):
        """Build PDP plots.
        One marginalized plot per each state dimension.

        Args:
            feature (str): Feature to plot.
            fig_name (str): Name to save plot.
            savefig (bool): Whether to save the plot.
        """

        ft_index = self._state_labels.index(feature)

        # Plot action-state graph
        actions = self._denorm_actions[ft_index]
        states = [str(round(i[0], 2)) for i in self._denorm_states[ft_index]]
        samples = self._dig_state_actions_samples[ft_index]
        fig, ax1 = plt.subplots()
        plt.grid(zorder=0)
        ax1.plot(states, actions, marker="o", color="b", zorder=3)
        ax1.set_xlabel(f"State dimension {feature}")
        ax1.set_ylabel("Actions")

        # Super-impose number of samples plot
        ax2 = ax1.twinx()
        ax2.bar(
            x=states,
            height=samples[:, 1],
            zorder=3,
            alpha=0.25,
            color="b",
            label="total",
        )
        ax2.bar(
            x=states,
            height=samples[:, 0],
            zorder=3,
            alpha=0.5,
            color="b",
            label="greedy",
        )
        ax2.set_ylabel("Num. of samples")
        plt.legend()

        if savefig:
            plt.savefig(fig_name, dpi=600)

        plt.show()
