from library import *

# Import environment and agent
from src.agents.q_learner import QLearningAgent
from src.agents.sarsa import SarsaAgent
from src.agents.sarsa_lambda import SarsaLambdaAgent
from src.agents.double_q_learner import DoubleQLearner
from src.environments.strategic_pricing_suggestion import StrategicPricingSuggestionMDP
from src.environments.strategic_pricing_prediction import StrategicPricingPredictionMDP


# TODO: Ludo thinks we should just pass the Engine the whole hyperparam dictionary and that it should also create the data handler.
class Engine:
    """Define the Engine class which is responsible for creating the agent and
    environment instances and running the training loop.
    """

    __slots__ = ["dh", "agent_type", "env_type", "agent", "env", "gamma",
                 "episode_flag", "num_episodes", "num_steps", "policy", 
                 "q_table", "bins", "train_test_split", "agent_cumrewards",
                 "hist_cumrewards", "_eval_states", "_eval_actions","_eval_rewards",
                 "_eval_b_states","_eval_state_dims","_eval_action_dims", "verbose"]

    def __init__(self, 
                 dh,
                 agent_type,
                 env_type,
                 num_episodes,
                 num_steps,
                 bins,
                 train_test_split=0.2,
                 gamma=0.9,
                 verbose=False):
        """Initilize engine class.

        Args:
            dh (DataHandler): DataHandler to be given to the Environment
            agent_type (str): Type of agent to initialize
            env_type (str): Type of environment to initialize
            num_episodes (int): Number of episodes to train the agent for
            num_steps (int): Number of steps per episode
            bins (list): List of bins per state/action to discretize the state
                        space.
            gamma (float): Discount factor
            train_test_split (float): proportion of test data
        """
        # Save data handler
        self.dh = dh

        # Hyperparameters
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.gamma = gamma

        # Initialize agent
        self.agent_type = agent_type
        self.agent = None

        # Initialize environment
        self.env_type = env_type
        self.env = None
        self.verbose = verbose
        self.bins = bins

        # Parameters of the agent
        self.policy = None
        self.q_table = None

        # Parameters for evaluation
        self.agent_cumrewards = None
        self.hist_cumrewards = None
        self._eval_states = None
        self._eval_b_states = None
        self._eval_actions = None
        self._eval_rewards = None
        self._eval_state_dims = None
        self._eval_action_dims = None

    def create_world(self):
        """Create the Agent and MDP instances for the given task.
        """
        # Create chosen environment
        print("Initialize environment")
        self.create_env()
        
        # Create chosen agent
        print("Initialize agent")
        self.create_agent()

    def create_agent(self):
        """Create an agent and store it in Engine.
        """
        # Initialize agent
        if self.agent_type == "q_learner":
            self.agent = QLearningAgent(self.env,
                                        gamma=self.gamma,
                                        verbose=self.verbose)

        elif self.agent_type == "sarsa":
            self.agent = SarsaAgent(env=self.env,
                                    gamma=self.gamma,
                                    verbose=self.verbose)

        elif self.agent_type == "sarsa_lambda":
            self.agent = SarsaLambdaAgent(env=self.env,
                                          gamma=self.gamma,
                                          verbose=self.verbose,
                                          lambda_=0.9) # TODO make this a parameter passed by the dictionary.

        elif self.agent_type == "double_q_learner":
            self.agent = DoubleQLearner(env=self.env,
                                        gamma=self.gamma,
                                        verbose=self.verbose)

        else:
            raise NotImplementedError

        self.agent.create_tables()

    def create_env(self):
        """Create an env and store it in Engine.
        """
        # Initialize environment
        if self.env_type == "strategic_pricing_predict":
            self.env = StrategicPricingPredictionMDP(self.dh, self.bins)

        elif self.env_type == "strategic_pricing_suggest":
            self.env = StrategicPricingSuggestionMDP(self.dh, self.bins)

        else:
            raise NotImplementedError

    def train_agent(self,
                    evaluate=False,
                    n_eval_steps=10):
        """Train the agent for a chosen number of steps and episodes.

        Args:
            evaluate (bool): whether to evaluate agent
            n_eval_steps (int): number of evaluation steps
        """
        # Fit the agent
        if not evaluate:
            self.agent.fit(self.num_episodes, self.num_steps)
        
        self.agent_cumrewards = []
        if evaluate:
            self.build_evaluation()
            for i in range(int(self.num_episodes/n_eval_steps)):
                self.agent.fit(n_eval_steps, self.num_steps)
                self.agent_cumrewards.append(self._evaluate_total_agent_reward())
            self.hist_cumrewards = self._evaluate_total_hist_reward()

    def save_parameters(self):
        """Save the parameters learned during training.
        This could be e.g. the q-values, the policy, or any other learned parameters.

        TODO: Not sure this function is needed, can call directly agent
        TODO: Epsilon greedy policy already contains q-values, remove it?
        """
        # Save parameters of the trained agent to predict
        self.policy = self.agent.policy
        self.q_table = self.agent.q_table

    def _inverse_scale_feature(self,
                               values,
                               labels):
        """De-bin and de-normalize feature values.

        Args:
            labels (list): list of feature labels
            values (list): list of (scaled) feature values
        
        Returns:
        """
        i_values = []
        for i, label in enumerate(labels):
            try:
                scaler = self.dh.minmax_scalars[label]
            except:
                ipdb.set_trace()
            val = np.array([v[i] for v in values])
            val = scaler.inverse_transform(
                    val.reshape(-1, 1))
            i_values.append(val)
        # transpose and convert to list
        i_values = np.concatenate(
            [np.expand_dims(v,1) for v in i_values],
            1).squeeze(-1).tolist()
        return i_values

    # TODO: From Giulia, for when we do the clean up, do we need to keep this here or can we put it in the evaluator?
    # TODO: Since it's part of the evaluation is better maybe
    def build_evaluation(self):
        """ Save data for evaluation. """
        # Get test data from data handler
        self._eval_states = self.dh.get_states(split='test').to_numpy().tolist()
        self._eval_actions = self.dh.get_actions(split='test').to_numpy().tolist()
        self._eval_rewards = self.dh.get_rewards(split='test').to_numpy().tolist()

        # Get state and action indexes
        self._eval_state_dims = list(range(self.env.state_dim))
        self._eval_action_dims = list(range(self.env.state_dim,
                                            self.env.state_dim+self.env.action_dim))
        # Get the binned states
        self._eval_b_states = self.env.bin_states(self._eval_states, idxs=self._eval_state_dims)

    def _evaluate_total_agent_reward(self):
        """ Calculate the total reward obtained on the evaluation states using
            the agent's policy.
        
        Returns:
            cumreward (float): total (not scaled) cumulative reward
        """
        # Get actions corresponding to agent's learned policy
        b_actions_agent = self.agent.predict_actions(self._eval_b_states)

        # De-bin the recommended actions
        actions_agent = self.env.debin_states(b_actions_agent, idxs=self._eval_action_dims)

        # Get reward based on agent policy
        rewards_agent = self.agent.predict_rewards(self._eval_b_states, b_actions_agent)
        
        # Inverse scale agent rewards
        rewards_agent = self._inverse_scale_feature(rewards_agent,
                                                    self.dh._reward_labels)

        return np.sum(rewards_agent)
    
    def _evaluate_total_hist_reward(self):
        """ Calculate the total reward obtained on the evaluation states using
            the agent's policy.
        
        Returns:
            cumreward (float): total (not scaled) cumulative based on historical data
        """
        # Get the binned actions
        b_actions =  self.env.bin_states(self._eval_actions, idxs=self._eval_action_dims)

        # Get reward based on historical policy
        rewards_hist = self.agent.predict_rewards(self._eval_b_states, b_actions)

        # Inverse scale agent rewards
        rewards_hist = self._inverse_scale_feature(rewards_hist,
                                                    self.dh._reward_labels)

        return np.sum(rewards_hist)

    def _evaluate_total_agent_reward(self):
        """ Calculate the total reward obtained on the evaluation states using
            the agent's policy.
        
        Returns:
            cumreward (float): total (not scaled) cumulative reward
        """
        # Get actions corresponding to agent's learned policy
        b_actions_agent = self.agent.predict_actions(self._eval_b_states)

        # De-bin the recommended actions
        actions_agent = self.env.debin_states(b_actions_agent, idxs=self._eval_action_dims)

        # Get reward based on agent policy
        rewards_agent = self.agent.predict_rewards(self._eval_b_states, b_actions_agent)
        
        # Inverse scale agent rewards
        rewards_agent = self._inverse_scale_feature(rewards_agent,
                                                    self.dh._reward_labels)

        return np.sum(rewards_agent)
    
    def _evaluate_total_hist_reward(self):
        """ Calculate the total reward obtained on the evaluation states using
            the agent's policy.
        
        Returns:
            cumreward (float): total (not scaled) cumulative based on historical data
        """
        # Get the binned actions
        b_actions =  self.env.bin_states(self._eval_actions, idxs=self._eval_action_dims)

        # Get reward based on historical policy
        rewards_hist = self.agent.predict_rewards(self._eval_b_states, b_actions)

        # Inverse scale agent rewards
        rewards_hist = self._inverse_scale_feature(rewards_hist,
                                                    self.dh._reward_labels)

        return np.sum(rewards_hist)

    def evaluate_agent(self, epsilon=0):
        """Evaluate the learned policy for the test states.
        Rewards are calculated using the average reward matrix.

        Args:
            epsilon: value of epsilon in the epsilon-greedy policy
                (default= 0 corresponds to pure exploitation)
        Returns:
            states (list): list of test states
            actions (list): list of test actions (historical)
            rewards_hist (list): list of historical rewards (calculated)
            actions_agent (list): list of recommended actions
            rewards_agent (list): list of rewards obtained by agent (calculated)
        """
        # Define dictionary containing evaluation results
        eval_results = {}

        # Save training results
        eval_results['agent_cumrewards'] = self.agent_cumrewards
        eval_results['hist_cumrewards'] = self.hist_cumrewards
 
        # Get test data from data handler
        states = self.dh.get_states(split='test').to_numpy().tolist()
        actions = self.dh.get_actions(split='test').to_numpy().tolist()
        rewards = self.dh.get_rewards(split='test').to_numpy().tolist()

        # get state and action indexes
        state_dims = list(range(self.env.state_dim))
        action_dims = list(range(self.env.state_dim, 
                                 self.env.state_dim+self.env.action_dim))
        # Get the binned states
        b_states = self.env.bin_states(states, idxs=state_dims)
        # Inverse scaling
        states = self._inverse_scale_feature(states, self.dh._state_labels)

        # Get the binned actions
        b_actions = self.env.bin_states(actions, idxs=action_dims)

        # Get actions corresponding to agent's learned policy
        b_actions_agent = self.agent.predict_actions(b_states)

        # De-bin the recommended actions
        actions_agent = self.env.debin_states(b_actions_agent, idxs=action_dims)

        # Get reward based on agent policy
        rewards_agent = self.agent.predict_rewards(b_states, b_actions_agent)

        # Get reward based on historical policy
        rewards_hist = self.agent.predict_rewards(b_states, b_actions)

        #  Apply inverse scaling to actions, states, and rewards
        eval_results['states'] = self._inverse_scale_feature(states,
                                            self.dh._state_labels)
        eval_results['actions_hist'] = self._inverse_scale_feature(actions,
                                                    self.dh._action_labels)
        eval_results['actions_agent'] = self._inverse_scale_feature(actions_agent,
                                                    self.dh._action_labels)
        eval_results['rewards_hist'] = self._inverse_scale_feature(rewards_hist,
                                                    self.dh._reward_labels)
        eval_results['rewards_agent'] = self._inverse_scale_feature(rewards_agent,
                                                    self.dh._reward_labels)
        
        # Save additional arrays
        eval_results['b_actions'] = b_actions
        eval_results['b_actions_agent'] = b_actions_agent
        
        return eval_results
