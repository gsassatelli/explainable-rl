import numpy as np
import matplotlib.pyplot as plt


class PDP():
    def __init__(self, bins,
                 minmax_scalers,
                 action_labels,
                 state_labels):
        """ Initialize PDP class.
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

    def build_data_for_plots(self, Q):
        self.get_digitized_pdp(Q)
        self.get_denorm_actions()
        self.get_denorm_states()
    
    def get_digitized_pdp(self, Q):
        Q_array = Q.todense()
        shape_Q = Q_array.shape
        num_dims = len(shape_Q)
        num_states = num_dims - 1 # last dimension is action
        self.bins_per_dim = [shape_Q[i] for i in range(num_dims)]
        set_states = set(list(range(num_states)))

        # for each state dimension
        for dim in range(num_states):
            states_to_avg = tuple(set_states - set([dim]))
            Q_avg = np.mean(Q_array, axis=states_to_avg)
            # select action with highest avg Q value
            dig_actions = np.argmax(Q_avg, axis=-1)
            self.dig_state_actions.append(dig_actions)

    def get_denorm_actions(self):
        scaler = self.minmax_scalers[self.action_labels[0]]
        for dig_actions in self.dig_state_actions:
            # divide dig actions by # bins of the action dimension
            # to get a value between 0 and 1
            denorm_action = scaler.inverse_transform(
                            dig_actions.reshape(-1,1)/self.bins_per_dim[-1])
            self.denorm_actions.append(denorm_action)
    
    def get_denorm_states(self):
        num_states = len(self.denorm_actions)
        for i in range(num_states):
            n_bins = self.bins_per_dim[i]
            # divide by number of bins to get a value between [0,1]
            # which can then be inputted into the scaler
            dig_values =  np.array(list(range(n_bins)))/n_bins
            scaler = self.minmax_scalers[self.state_labels[i]]
            denorm_state = scaler.inverse_transform(dig_values.reshape(-1,1))
            self.denorm_states.append(denorm_state)

    def plot_pdp(self, states_names, fig_name, type_features, savefig=True, all_states=True):
        rows = len(self.denorm_actions)
        cols = 1

        fig, ax = plt.subplots(rows, cols, sharex=False, sharey=True)

        for r in range(rows):
            actions = [i[0] for i in self.denorm_actions[r]]
            states = [str(round(i[0], 2)) for i in self.denorm_states[r]]
            if not all_states:
                states = [states[idx] for idx, a in enumerate(actions) if a > 0.1]
                actions = [i for i in actions if i > 0.1]

            state = states_names[r]

            ax[r].grid(zorder=0)
            if type_features[state] == "continuous":
                ax[r].plot(states, actions, marker="o", zorder=3)
            else:
                ax[r].bar(x=states, height=actions, zorder=3)
            ax[r].set(xlabel=f"State dimension {state}", ylabel="Actions")

        plt.subplots_adjust(top=0.99, bottom=0.1, hspace=0.5, wspace=0.4)

        if savefig:
            plt.savefig(fig_name, dpi=600)

        plt.show()






