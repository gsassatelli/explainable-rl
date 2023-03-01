import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import sparse


class PDP:

    __slots__ = ["_bins", "_minmax_scalers", "_action_labels", "_state_labels",
                 "_dig_state_actions", "_denorm_actions", "_denorm_states", "_bins_per_dim"]

    def __init__(self,
                 bins,
                 minmax_scalers,
                 action_labels,
                 state_labels):
        """Initialize PDP class.

        Args:
            bins (list): list of bins per each state and action pair.
            minmax_scalers (dict): list of scalers per each state dimension and actions.
            action_labels (list): list of action column names.
            state_labels (list): list of state dimensions column names.
        """
        self._bins = bins
        self._minmax_scalers = minmax_scalers
        self._action_labels = action_labels
        self._state_labels = state_labels
        self._dig_state_actions = []
        self._denorm_actions = []
        self._denorm_states = []
        self._bins_per_dim = []

    def build_data_for_plots(self,
                             Q):
        """Prepare data to build PDP plots.

        Args:
            Q (sparse.DOK): Q-table to build plots.
        """
        self._get_digitized_pdp(Q)
        self._get_denorm_actions()
        self._get_denorm_states()
    
    def _get_digitized_pdp(self,
                          Q):
        """Compute average Q-value per each state-action pair.
        Marginal effect of the state-action pair averaging other state dimensions.

        Args:
            Q (sparse.DOK): Q-table to build plots.
        """
        Q_array = Q.todense()
        shape_Q = Q_array.shape
        num_dims = len(shape_Q)
        num_states = num_dims - 1 # last dimension is action
        self._bins_per_dim = [shape_Q[i] for i in range(num_dims)]
        set_states = set(list(range(num_states)))

        # For each state dimension
        for dim in range(num_states):
            states_to_avg = tuple(set_states - set([dim]))
            Q_avg = np.mean(Q_array, axis=states_to_avg)
            # Select action with highest avg Q value
            dig_actions = np.argmax(Q_avg, axis=-1)
            self._dig_state_actions.append(dig_actions)

    def _get_denorm_actions(self):
        """Get actions denormalized values.
        """
        scaler = self._minmax_scalers[self._action_labels[0]]
        for dig_actions in self._dig_state_actions:
            # Divide dig actions by # bins of the action dimension
            # to get a value between 0 and 1
            denorm_action = scaler.inverse_transform(
                            dig_actions.reshape(-1,1)/self._bins_per_dim[-1])
            self._denorm_actions.append(denorm_action)
    
    def _get_denorm_states(self):
        """Get states denormalized values.
        """
        num_states = len(self._denorm_actions)
        for i in range(num_states):
            n_bins = self._bins_per_dim[i]
            # Divide by number of bins to get a value between [0,1]
            # which can then be inputted into the scaler
            dig_values = np.array(list(range(n_bins)))/n_bins
            scaler = self._minmax_scalers[self._state_labels[i]]
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
            actions = [i[0] for i in self._denorm_actions[a]]
            states = [str(round(i[0], 2)) for i in self._denorm_states[a]]
            if not all_states:
                states = [states[idx] for idx, a in enumerate(actions) if a > 0.1]
                actions = [i for i in actions if i > 0.1]

            state = states_names[a]

            ax[a].grid(zorder=0)
            if type_features[state] == "continuous":
                ax[a].plot(states, actions, marker="o", zorder=3)
            else:
                ax[a].bar(x=states, height=actions, zorder=3)
            ax[a].set(xlabel=f"State dimension {state}", ylabel="Actions")

        plt.subplots_adjust(top=0.99, bottom=0.1, hspace=0.5, wspace=0.4)

        if savefig:
            plt.savefig(fig_name, dpi=600)

        plt.show()






