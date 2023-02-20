import numpy as np
import matplotlib.pyplot as plt


class PDP():
    def __init__(self, bins,
                 minmax_scalers,
                 action_labels,
                 state_labels):
        """ Initialize PDP class.
        
        FOR GIULIA:
        I put the "denormalized" state values in the denorm_states attribute,
        you can directly use it for the x-axis of the plots.
        You can also use self.state_labels and self.action_labels for the names.
        
        I also fixed the normalization issue and now we get prices between[0,350] for the prices
        <3
        """
        self.pdp = None
        self.bins = bins
        self.minmax_scalers = minmax_scalers
        self.action_labels = action_labels
        self.state_labels = state_labels
        self.dig_state_actions = []
        self.state_actions = []
        self.denorm_states = []
        self.bins_per_dim = []

    def build_pdp_plots(self,Q, states_names, savefig=False):
        self.get_digitized_pdp(Q)
        self.denormalize_pdp()
        self.get_denorm_states()
        self.plot_pdp(states_names, savefig=savefig)
    
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

    def denormalize_pdp(self):
        scaler = self.minmax_scalers[self.action_labels[0]]
        for dig_actions in self.dig_state_actions:
            # divide dig actions by # bins of the action dimension
            # to get a value between 0 and 1
            denorm_action = scaler.inverse_transform(
                            dig_actions.reshape(-1,1)/self.bins_per_dim[-1])
            self.state_actions.append(denorm_action)
    
    def get_denorm_states(self):
        num_states = len(self.state_actions)
        for i in range(num_states):
            n_bins = self.bins_per_dim[i]
            # divide by number of bins to get a value between [0,1]
            # which can then be inputted into the scaler
            dig_values =  np.array(list(range(n_bins)))/n_bins
            scaler = self.minmax_scalers[self.state_labels[i]]
            denorm_state = scaler.inverse_transform(dig_values.reshape(-1,1))
            self.denorm_states.append(denorm_state)

    def plot_pdp(self, states_names, savefig = False):
        rows = len(self.state_actions)
        cols = 1

        fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row')

        for r in range(rows):
            actions = self.state_actions[r]
            states = list(range(len(actions)))
            state = states_names[r]
            ax[r].plot(states, actions, marker="o")
            ax[r].set(xlabel=f"State dimension {state}",
                      ylabel="Actions")
                      # title=f"PDP of state dimension {state}"

        if savefig:
            plt.savefig("PDP plots", dpi=600)

        plt.show()






