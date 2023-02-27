import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import sparse


class PDP():
    def __init__(self,
                 bins: List,
                 minmax_scalers: Dict,
                 action_labels: List,
                 state_labels: List):
        """Initialize PDP class.

        Args:
            bins: list of bins per each state and action pair.
            minmax_scalers: list of scalers per each state dimension and actions.
            action_labels: list of action column names.
            state_labels: list of state dimensions column names.
        """
        self.pdp = None
        self.bins = bins
        self.minmax_scalers = minmax_scalers
        self.action_labels = action_labels
        self.state_labels = state_labels
        self.dig_state_actions = []
        self.denorm_actions = []
        self.denorm_states = []
        self.bins_per_dim = []

    def build_data_for_plots(self,
                             Q: sparse.DOK):
        """Prepare data to build PDP plots.

        Args:
            Q: Q-table to build plots.
        """
        self.get_digitized_pdp(Q)
        self.get_denorm_actions()
        self.get_denorm_states()
    
    def get_digitized_pdp(self,
                          Q: sparse.DOK):
        """Compute average Q-value per each state-action pair.
        Marginal effect of the state-action pair averaging other state dimensions.

        Args:
            Q: Q-table to build plots.
        """
        Q_array = Q.todense()
        shape_Q = Q_array.shape
        num_dims = len(shape_Q)
        num_states = num_dims - 1 # last dimension is action
        self.bins_per_dim = [shape_Q[i] for i in range(num_dims)]
        set_states = set(list(range(num_states)))

        # For each state dimension
        for dim in range(num_states):
            states_to_avg = tuple(set_states - set([dim]))
            Q_avg = np.mean(Q_array, axis=states_to_avg)
            # Select action with highest avg Q value
            dig_actions = np.argmax(Q_avg, axis=-1)
            self.dig_state_actions.append(dig_actions)

    def get_denorm_actions(self):
        """Get actions denormalized values.
        """
        scaler = self.minmax_scalers[self.action_labels[0]]
        for dig_actions in self.dig_state_actions:
            # Divide dig actions by # bins of the action dimension
            # to get a value between 0 and 1
            denorm_action = scaler.inverse_transform(
                            dig_actions.reshape(-1,1)/self.bins_per_dim[-1])
            self.denorm_actions.append(denorm_action)
    
    def get_denorm_states(self):
        """Get states denormalized values.
        """
        num_states = len(self.denorm_actions)
        for i in range(num_states):
            n_bins = self.bins_per_dim[i]
            # Divide by number of bins to get a value between [0,1]
            # which can then be inputted into the scaler
            dig_values =  np.array(list(range(n_bins)))/n_bins
            scaler = self.minmax_scalers[self.state_labels[i]]
            denorm_state = scaler.inverse_transform(dig_values.reshape(-1,1))
            self.denorm_states.append(denorm_state)

    def plot_pdp(self,
                 states_names: List,
                 fig_name: str,
                 type_features: Dict,
                 savefig: bool = True,
                 all_states: bool = True):
        """Build PDP plots.
        One marginalized plot per each state dimension.

        Args:
            states_names: list of state dimensions column names
            fig_name: figure name to save plot.
            type_features: type of variable per each state dimension.
                Information used to choose how to plot each PDP.
            savefig: bool to choose whether to save the plot.
            all_states: bool to choose whether to plot the unvisited states.
        """
        rows = len(self.denorm_actions)
        cols = 1

        fig, ax = plt.subplots(rows, cols, sharex=False, sharey=True)

        for a in range(rows):
            actions = [i[0] for i in self.denorm_actions[a]]
            states = [str(round(i[0], 2)) for i in self.denorm_states[a]]
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






