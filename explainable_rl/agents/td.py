# Import functions
from explainable_rl.foundation.utils import *
from explainable_rl.foundation.agent import Agent


class TD(Agent):
    """Agent class to store and update q-table."""

    def __init__(self, env, gamma, verbose=False):
        super().__init__(env, gamma, verbose)
        """Initialise the agent class.

        Args:
            env (MDP): MDP object.
            gamma (float): Discount factor.
            verbose (bool): Defines whether print statements should be called.
        """
        self.Q = None
        self.state_to_action = None
        self.Q_num_samples = None
        self.state = None

    def fit(self, agent_hyperparams, training_hyperparams, verbose=False, pbar=None):
        """Fit agent to the dataset.

        Args:
            agent_hyperparams (dict): Dictionary of agent hyperparameters.
            training_hyperparams (dict): Dictionary of training hyperparameters.
            verbose (bool): Print training information.
            pbar (tqdm): Progress bar.
        """
        if verbose:
            print("Apply q-learning and update q-table")
        lr = agent_hyperparams["learning_rate"]
        epsilon = agent_hyperparams["epsilon"]
        for _ in range(training_hyperparams["num_episodes"]):

            self.state = self.env.reset()

            for i in range(training_hyperparams["num_steps"]):
                done = self._step(
                    epsilon=epsilon,
                    lr=lr,
                    use_uncertainty=agent_hyperparams["use_uncertainty"],
                )
                if done:
                    break
            if pbar is not None:
                pbar.update(1)
            lr = decay_param(
                lr,
                agent_hyperparams["learning_rate_decay"],
                agent_hyperparams["learning_rate_minimum"],
            )
            epsilon = decay_param(
                epsilon,
                agent_hyperparams["epsilon_decay"],
                agent_hyperparams["epsilon_minimum"],
            )

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

    def _epsilon_greedy_policy(self, state=None, epsilon=0.1, Q=None):
        """Epsilon-greedy policy.

        Args:
            state (int): State.
            epsilon (float): Epsilon of epsilon-greedy policy.
                Defaults to 0 for pure exploitation.
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

    def uncertainty_informed_policy(
        self, state=None, epsilon=0.1, use_uncertainty=False, q_importance=0.7
    ):
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
        index_no_action = tuple(list(state))
        possible_actions = self._get_possible_actions(state)
        if use_uncertainty:
            sum_possible_q = sum(self.Q[index_no_action].todense())
            if sum_possible_q == 0:
                return np.random.choice(list(possible_actions))

            state_action_counts, q_values_weights = self._get_q_value_weights(
                sum_possible_q=sum_possible_q,
                state=state,
                possible_actions=possible_actions,
            )
            uncertainty_weights = self._get_uncertainty_weights(state_action_counts)
            if random.random() < epsilon:
                action = np.random.choice(list(possible_actions))
            else:
                action_scores = self._get_action_scores(
                    possible_actions=possible_actions,
                    q_importance=q_importance,
                    q_values_weights=q_values_weights,
                    uncertainty_weights=uncertainty_weights,
                )
                action = max(action_scores, key=action_scores.get)
        else:
            action = self._epsilon_greedy_policy(self.state, epsilon=epsilon)

        return action

    def _step(self, epsilon, lr, use_uncertainty):
        """Perform a step in the environment.

        Args:
            epsilon (float): Epsilon-greedy policy parameter.
            lr (float): Learning rate.
            use_uncertainty (bool): Whether to use uncertainty informed policy.

        Returns:
            bool: Defines whether the episode is finished.
        """
        action = self.uncertainty_informed_policy(
            self.state,
            epsilon=epsilon,
            use_uncertainty=use_uncertainty,
            q_importance=0.7,
        )

        state, next_state, reward, done = self.env.step(self.state, action)
        self._update_q_values(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            lr=lr,
            epsilon=epsilon,
        )
        self.state = next_state
        return done

    def _update_q_values(
        self, state, action, next_state, reward, epsilon, lr, **kwargs
    ):
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

    def _get_possible_actions(self, state):
        """Get the possible actions from a state.

        Args:
            state (list): current state of the agent.

        Returns:
            possible_actions (set): the possible actions that the agent can
                                    take from the state.
        """
        try:
            state_str = self._convert_to_string(state)
            possible_actions = self.env.state_to_action[state_str]
        except KeyError:
            state_str = self._convert_to_string(state[:-1])
            possible_actions = self.env.state_to_action[state_str]

        return possible_actions

    def _get_q_value_weights(self, sum_possible_q, state, possible_actions):
        """Get the q value of each action as a percentage of the total q value.

        Args:
            sum_possible_q (float): the sum of the q values for the state.
            state (list): the state of the agent.
            possible_actions (set): the possible actions that the agent can
                                    take from the state.

        Returns:
            state_action_counts (dict): count of how many times a state-action
            pair has appeared.
            q_values_weights (dict): the q-weight of each state-action pair.
        """
        state_action_counts = {}
        q_values_weights = {}

        for possible_action in possible_actions:
            possible_state_action_str = self._convert_to_string(
                state + [possible_action]
            )
            counts = self.env.bins_dict[possible_state_action_str][0]
            # Count number of times a state-action pair occurred
            state_action_counts[str(possible_action)] = counts
            index_with_action = tuple(state + [possible_action])
            q_values_weights[possible_action] = (
                self.Q[index_with_action] / sum_possible_q
            )

        return state_action_counts, q_values_weights

    def _get_action_scores(
        self, possible_actions, q_importance, q_values_weights, uncertainty_weights
    ):
        """Get the score for each action from a state.

        Args:
            possible_actions (set): the possible actions for an agent in a
                                    state.
            q_importance (float): the weighting of the q value vs the amount
                                  a state has been seen.
            q_values_weights (dict): the q-weight of each state-action pair.
            uncertainty_weights (dict): the count-weight of each state-action
                                        pair.

        Returns:
            action_scores (dict): the weighted score of each possible action
                                  from the state.

        """
        action_scores = {}
        for possible_action in possible_actions:
            score = (
                q_importance * q_values_weights[possible_action]
                + (1 - q_importance) * uncertainty_weights[possible_action]
            )
            action_scores[possible_action] = score
        return action_scores

    @staticmethod
    def _get_uncertainty_weights(state_action_counts):
        """Get uncertainty weight of an action from a state.

        This is defined as the proportion of times a state is visited in the
        historical data vs the total state visits of the possible next states.

        Args:
            state_action_counts (dict): the number of times a state has been
                                        visited in the historical data.

        Returns:
            dict: uncertainty weight of each possible state.
        """
        return {
            int(key): float(value) / sum(state_action_counts.values())
            for (key, value) in state_action_counts.items()
        }
