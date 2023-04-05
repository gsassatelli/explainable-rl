import numpy as np
import matplotlib.pyplot as plt
import sparse
import ipdb

class PDP:
    __slots__ = ["_bins", "_minmax_scalars", "_action_labels", "_state_labels",
                 "_dig_state_actions", "_denorm_actions", "_denorm_states", "_Q_array",
                 "_bins_per_dim", "_dig_state_actions_std", "_dig_state_actions_samples"]

    def __init__(self,
                 bins,
                 minmax_scalars,
                 action_labels,
                 state_labels):
        """Initialize PDP class.

        Args:
            bins (list): list of bins per each state and action pair.
            minmax_scalars (dict): list of scalars per each state dimension and actions.
            action_labels (list): list of action column names.
            state_labels (list): list of state dimensions column names.
            dig_state_actions (list[np.ndarray]): list of length = # state dims.
                Each element is a np.ndarray mapping the state bin to the index of the action
                that maximizes the (marginalized) Q-value wrt the other state dimensions.
            dig_state_std (list[np.ndarray]): list of length = # state dims.
                Each element is a np.ndarray mapping the state bin to the standard deviation
                of the (marginalized) Q-value corresponding to the greedy (argmax) action.
            dig_state_actions_samples (list[np.ndarray]): list of length = # state dims.
                Each element is a np.ndarray of shape (# bins, 2) mapping the state bin to:
                    1) # samples used to obtain the mean Q-value for the greedy (argmax) action.
                    2) total # samples explored for this bin

        """
        self._bins = bins
        self._minmax_scalars = minmax_scalars
        self._action_labels = action_labels
        self._state_labels = state_labels
        self._dig_state_actions = []
        self._dig_state_actions_std = []
        self._dig_state_actions_samples = []
        self._denorm_actions = []
        self._denorm_states = []
        self._bins_per_dim = []
        self._Q_array = []

    def build_data_for_plots(self,
                             Q,
                             Q_num_samples):
        """Prepare data to build PDP plots.

        Args:
            Q (sparse.DOK): Q-table to build plots.
            Q_num_samples (sparse.DOK): Q-table with number of samples per each state-action pair.
        """
        self._get_digitized_pdp(Q, Q_num_samples)
        self._get_denorm_actions()
        self._get_denorm_states()

    def _get_digitized_pdp(self,
                           Q,
                           Q_num_samples):
        """Compute average Q-value per each state-action pair.
        Marginal effect of the state-action pair averaging other state dimensions.

        Args:
            Q (sparse.DOK): Q-table to build plots.
            Q_num_samples (sparse.DOK): Q-table with number of samples per each state-action pair.
        """
        Q_array = Q.todense()
        self._Q_array = Q_array
        shape_Q = Q_array.shape
        num_dims = len(shape_Q)
        num_states = num_dims - 1  # last dimension is action
        self._bins_per_dim = [shape_Q[i] for i in range(num_dims)]
        set_states = set(list(range(num_states)))

        Q_num_samples_array = Q_num_samples.todense()

        # For each state dimension
        for dim in range(num_states):
            states_to_avg = tuple(set_states - set([dim]))
            Q_avg = np.mean(Q_array, axis=states_to_avg)
            Q_std = np.std(Q_array, axis=states_to_avg)
            Q_num_samples_sum = np.sum(Q_num_samples_array, axis=states_to_avg)
            # Select action with the highest avg Q value
            dig_actions = np.argmax(Q_avg, axis=-1)
            dig_actions_std = np.array([Q_std[idx][action] for idx, action in enumerate(dig_actions)])
            dig_actions_samples = np.array([Q_num_samples_sum[idx][action] for idx, action in enumerate(dig_actions)])
            # add the total number of samples per state bin
            dig_actions_samples = np.concatenate(
                [np.expand_dims(dig_actions_samples,-1), 
                np.expand_dims(Q_num_samples_sum.sum(-1),-1)],
                axis=-1
            )
            
            self._dig_state_actions.append(dig_actions)
            self._dig_state_actions_std.append(dig_actions_std)
            self._dig_state_actions_samples.append(dig_actions_samples)
            # TODO std no correct because it is not the std of denorm values

    def _get_denorm_actions(self):
        """Get actions denormalized values.
        """
        scaler = self._minmax_scalars[self._action_labels[0]]
        for dig_actions in self._dig_state_actions:
            # Divide dig actions by # bins of the action dimension
            # to get a value between 0 and 1
            denorm_action = scaler.inverse_transform(
                dig_actions.reshape(-1, 1) / self._bins_per_dim[-1])
            self._denorm_actions.append(denorm_action)

    def _get_denorm_states(self):
        """Get states denormalized values.
        """
        num_states = len(self._denorm_actions)
        for i in range(num_states):
            n_bins = self._bins_per_dim[i]
            # Divide by number of bins to get a value between [0,1]
            # which can then be inputted into the scaler
            dig_values = np.array(list(range(n_bins))) / n_bins
            scaler = self._minmax_scalars[self._state_labels[i]]
            denorm_state = scaler.inverse_transform(dig_values.reshape(-1, 1))
            self._denorm_states.append(denorm_state)

    def plot_pdp(self,
                 states_names,
                 fig_name,
                 type_features,
                 savefig=True,
                 all_states=True):
        """Build PDP plots.
        One marginalized plot per each state dimension.

        Args:
            states_names (list): list of state dimensions column names
            fig_name (str): figure name to save plot.
            type_features (dict): type of variable per each state dimension.
                Information used to choose how to plot each PDP.
            savefig (bool): bool to choose whether to save the plot.
            all_states (bool): bool to choose whether to plot the unvisited states.
        """
        rows = len(self._denorm_actions)
        cols = 1

        fig, ax = plt.subplots(rows, cols, sharex=False, sharey=True)

        for a in range(rows):
            state = states_names[a]
            # Plot action-state graph
            axis = [ax[a], ax[a].twinx()]
            actions = [i[0] for i in self._denorm_actions[a]]
            states = [str(round(i[0], 3)) for i in self._denorm_states[a]]
            samples = self._dig_state_actions_samples[a]
            if not all_states:
                total_samples = list(samples[:, 1])
                states = [states[idx] for idx, s in enumerate(total_samples) if s > 0]
                actions = [actions[idx] for idx, s in enumerate(total_samples) if s > 0]
                samples = np.array([samples[idx] for idx, s in enumerate(total_samples) if s > 0])

            axis[0].grid(zorder=0)
            #if type_features[state] == "continuous":
            if True:
                axis[0].plot(states, actions, marker="o", color='b', zorder=3)
            else:
                axis[0].bar(x=states, height=actions, color='b', zorder=3)
            axis[0].set(xlabel=f"State dimension {state}", ylabel="Actions")

            # Super-impose number of samples plot
            axis[1].bar(x=states, height=samples[:, 1], zorder=3, alpha=0.25, color='b',label='total')
            axis[1].bar(x=states, height=samples[:, 0], zorder=3, alpha=0.5, color='b', label='greedy')
            axis[1].set(ylabel='Num. of samples')
            axis[1].legend()
            # Super-impose the distribution over the actions
            """
            ipdb.set_trace()
            for i, txt in enumerate(len):
                ax.annotate(txt, (z[i], y[i]))
            """

        plt.subplots_adjust(top=0.99, bottom=0.1, hspace=0.5, wspace=0.4)

        if savefig:
            plt.savefig(fig_name, dpi=600)

        plt.show()