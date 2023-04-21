from library import *

# Import functions
from src.foundation.utils import *
from src.foundation.agent import Agent


class TD(Agent):
    """Agent class to store and update q-table."""

    def __init__(self, env, gamma, verbose=False):
        super().__init__(env, gamma, verbose)
        """Initialise the agent class.

        Args:
            env (MDP): MDP object.
            gamma (float): Discount factor.
        """
        self.Q = None
        self.state_to_action = None
        self.Q_num_samples = None
        self.state = None

    def fit(self, n_episodes, n_steps, use_uncertainty, lr=0.1, lr_decay=0.05, lr_min=0.01,
            epsilon=0.1, epsilon_decay=0.05, epsilon_min=0.01, verbose=False):
        """Fit agent to the dataset.

        Args:
            n_episodes (int): Number of episodes.
            n_steps (int): Number of steps per episode.
            use_uncertainty (bool): Use uncertainty in q-table.
            lr (float): Learning rate.
            lr_decay (float): Learning rate decay.
            lr_min (float): Minimum learning rate.
            epsilon (float): Epsilon-greedy policy parameter.
            epsilon_decay (float): Epsilon decay.
            epsilon_min (float): Minimum epsilon.
            verbose (bool): Print training information.
        """
        if verbose:
            print("Apply q-learning and update q-table")

        for _ in range(n_episodes):

            self.state = self.env.reset()

            for i in range(n_steps):
                done = self._step(epsilon=epsilon, lr=lr, use_uncertainty=use_uncertainty)
                if done:
                    break
            lr = decay_param(lr, lr_decay, lr_min)
            epsilon = decay_param(epsilon, epsilon_decay, epsilon_min)

        if verbose:
            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"{timestamp}: Finished training :) ! \n"
                  f"Example Q-table for state "
                  f"{[1, 9, 0]}: {self.Q[1, 9, 0].todense()}\n"
                  f"Example Q-table for state "
                  f"{[1, 0, 0]}: {self.Q[1, 0, 0].todense()}")

    def create_tables(self, verbose=False):
        """Initialize the agent.

        This resets the environment, creates the q-table and the state to
        action mapping.

        Args:
            verbose (bool): Print information.
        """
        self.env.reset()
        if verbose:
            print("Create q-table")

        # Create q-table
        self._init_q_table()
        self.state_to_action = self.env.state_to_action

    def _epsilon_greedy_policy(self,
                               state=None,
                               epsilon=0.1,
                               Q=None):
        """Get the epsilon greedy action.

        Args:
            state (list): current state of the agent.
            epsilon (float): the exploration parameter.
            Q (np.array): q-table.

        Returns:
            action (int): Selected action.
        """
        if state is None:
            state = self.state

        state_str = self._convert_to_string(state)
        index = tuple(list(state))
        if Q is None:
            Q = self.Q
        q_values = Q[index].todense()
        r = random.random()
        if r > epsilon:
            action = np.argmax(q_values)
        else:
            try:
                action = random.choice(list(self.state_to_action[str(state_str)]))
            except KeyError:
                state = state[:-1]
                state_str = self._convert_to_string(state)
                action = random.choice(list(self.state_to_action[str(state_str)]))
        return action
    
    def uncertainty_informed_policy(self, state=None, epsilon=0.1, use_uncertainty=False, q_importance=0.7):
        """Get epsilon greedy policy that favours more densely populated state-action pairs. 

        Args:
            state (list): Current state of the agent.
            epsilon (float): The exploration parameter.
            use_uncertainty (bool): Whether to use uncertainty informed policy.
            q_importance (float): The importance of the q value in the policy.

        Returns:
            action (int): selected action.
        """
        if state is None:
            state = self.state
        try:
            state_str = self._convert_to_string(state)
            index_no_action = tuple(list(state))
            possible_actions = self.env.state_to_action[state_str]
        except KeyError:
            state_str = self._convert_to_string(state[:-1])
            index_no_action = tuple(list(state))
            possible_actions = self.env.state_to_action[state_str]

        if use_uncertainty:

            state_action_counts = {}
            q_values_weights = {}

            # Determine the sum of the q values for the possible actions 
            sum_possible_q = sum(self.Q[index_no_action].todense())
            
            if sum_possible_q == 0:
                return np.random.choice(list(possible_actions))

            for possible_action in possible_actions:
                possible_state_action_str = self._convert_to_string(state + [possible_action])
                counts = self.env.bins_dict[possible_state_action_str][0]
                # Count number of times a state-action pair occurred
                state_action_counts[str(possible_action)] = counts
                index_with_action = tuple(state + [possible_action])
                q_values_weights[possible_action] = self.Q[index_with_action] / sum_possible_q
            
            # Get weights given population for state-action space
            # N.B. A high value represents a well-known, certain state
            uncertainty_weights = {int(key): float(value)/sum(state_action_counts.values())
                                   for (key, value) in state_action_counts.items()}

            if random.random() > epsilon:  # Exploring
                action = np.random.choice(list(possible_actions))
            else:  # Exploiting
                for possible_action in possible_actions:
                    score = q_importance * q_values_weights[possible_action] + \
                            (1 - q_importance) * uncertainty_weights[possible_action]
                    action_scores = {possible_action: score}
                action = np.argmax(list(action_scores.values()))
        else:
            action = self._epsilon_greedy_policy(self.state, epsilon=epsilon)
        
        return action

    @staticmethod
    def _convert_to_string(state):
        """Convert a state to a string.

        Args:
            state (list): The state to convert.

        Returns:
            state_str (string): The state as a string.
        """
        return ",".join(str(s) for s in state)

    def _init_q_table(self):
        """Initialize the q-table with zeros."""
        self.Q = sparse.DOK(self.env.bins)
        self.Q_num_samples = sparse.DOK(self.env.bins)

    def _step(self, epsilon, lr, use_uncertainty):
        """Perform a step in the environment.

        Args:
            epsilon (float): Epsilon-greedy policy parameter.
            lr (float): Learning rate.
            use_uncertainty (bool): Whether to use uncertainty informed policy.

        Returns:
            bool: Defines whether the episode is finished.
        """
        action = self.uncertainty_informed_policy(self.state,
                                                  epsilon=epsilon,
                                                  use_uncertainty=False, # TODO: readd parameter
                                                  q_importance=0.7)
        
        state, next_state, reward, done = self.env.step(self.state, action)
        self._update_q_values(state=state,
                              action=action,
                              next_state=next_state,
                              reward=reward,
                              lr=lr,
                              epsilon=epsilon)
        self.state = next_state
        return done

    def _update_q_values(self, state, action, next_state, reward, epsilon, lr, **kwargs):
        """Update the Q table.

        Args:
            state (list): Current state of the agent.
            action (int): Selected action.
            next_state (list): Next state of the agent.
            reward (float): Reward for the selected action.
            epsilon (float): The exploration parameter.
            lr (float): Learning rate.
            **kwargs (dict): The keyword arguments.
        """
        raise NotImplementedError
