
Q_array = self.Q.todense()
dim = 0
num_states = len(shape_Q)
set_states = set(list(range(num_states-1)))
states_to_avg = tuple(set_states - set([dim]))
values = np.mean(Q_array, axis=states_to_avg)
max_indeces = np.argmax(values, axis=-1)