import numpy as np

class PDP():
    def __init__(self, bins, minmax_scal):
        self.pdp = None
        self.bins = bins
        self.minmax_scal = minmax_scal
        self.dig_state_actions = []
        self.state_actions = []
    
    def get_digitized_pdp(self, Q):
        Q_array = Q.todense()
        shape_Q = Q_array.shape
        num_states = len(shape_Q)
        set_states = set(list(range(num_states-1)))

        # for each state dimension
        for dim in range(num_states-1):
            states_to_avg = tuple(set_states - set([dim]))
            Q_avg = np.mean(Q_array, axis=states_to_avg)
            # select action with highest avg Q value
            dig_actions = np.argmax(Q_avg, axis=-1)
            self.dig_state_actions.append(dig_actions)
            
    def denormalize_pdp(self):
        for dig_actions in self.dig_state_actions:
            unscaled = self.minmax_scal.inverse_transform(
                            dig_actions.reshape(-1,1))
            self.state_actions.append(unscaled)
            
    
    def plot_pdp(self, Q, savefig = False):
        pass


