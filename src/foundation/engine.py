from library import *

# Import environment and agent
from src.agents.q_learner import QLearningAgent
from src.agents.sarsa import SarsaAgent
from src.agents.sarsa_lambda import SarsaLambdaAgent
from src.agents.double_q_learner import DoubleQLearner
from src.environments.strategic_pricing_suggestion import StrategicPricingSuggestionMDP
from src.environments.strategic_pricing_prediction import StrategicPricingPredictionMDP


class Engine:
    """Responsible for creating the agent and environment instances and running the training loop."""

    def __init__(self, dh, hyperparam_dict, verbose=False):
        """Initialise engine class.

        Args:
            dh (DataHandler): DataHandler to be given to the Environment.
            hyperparam_dict (dict): Dictionary containing all hyperparameters.
            verbose (bool): Whether print statements about the program flow should be displayed.
        """
        # Save data handler
        self.dh = dh
        self.hyperparameters = hyperparam_dict

        # Hyperparameters
        self.num_episodes = hyperparam_dict['training']['num_episodes']
        self.num_steps = hyperparam_dict['training']['num_steps']
        self.gamma = hyperparam_dict['agent']['gamma']

        # Initialize agent
        self.agent_type = hyperparam_dict['agent']['agent_type']
        self.agent = None

        # Initialize environment
        self.env_type = hyperparam_dict['training']['env_type']
        self.env = None
        self.verbose = hyperparam_dict['program_flow']['verbose']
        self.bins = self._get_bins()

        # Parameters of the agent
        self.policy = None
        self.q_table = None

        # Parameters for evaluation        
        self.evaluate = hyperparam_dict['training']['evaluate']
        self.num_eval_steps = hyperparam_dict['training']['num_eval_steps']
        self.eval_agent_rewards = []
        self.eval_hist_rewards = None

    def create_world(self):
        """Create the Agent and MDP instances for the given task."""
        # Create chosen environment
        if self.verbose:
            print("Initialize environment")
        self.create_env()
        
        # Create chosen agent
        if self.verbose:
            print("Initialize agent")
        self.create_agent()

    def create_agent(self):
        """Create an agent and store it in Engine."""
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
                                          lambda_=self.hyperparameters['agent']['lambda'])

        elif self.agent_type == "double_q_learner":
            self.agent = DoubleQLearner(env=self.env,
                                        gamma=self.gamma,
                                        verbose=self.verbose)

        else:
            raise NotImplementedError

        self.agent.create_tables()

    def create_env(self):
        """Create an env and store it in Engine."""
        # Initialize environment
        if self.env_type == "strategic_pricing_predict":
            self.env = StrategicPricingPredictionMDP(self.dh, self.bins)

        elif self.env_type == "strategic_pricing_suggest":
            self.env = StrategicPricingSuggestionMDP(self.dh, self.bins)

        else:
            raise NotImplementedError

    def train_agent(self):
        """Train the agent for a chosen number of steps and episodes.
        """
        # Fit the agent
        if not self.evaluate:
            with tqdm(total=self.num_episodes) as pbar:
                self.agent.fit(agent_hyperparams=self.hyperparameters['agent'],
                               training_hyperparams=self.hyperparameters['training'],
                               verbose=self.verbose,
                               pbar=pbar)
        
        self.agent_cumrewards = []
        if self.evaluate:
            self.build_evaluation()
            self.hyperparameters['training']['num_episodes'] = self.num_eval_steps
            with tqdm(total=self.num_episodes) as pbar:
                for i in range(int(self.num_episodes/self.num_eval_steps)):
                    self.agent.fit(agent_hyperparams=self.hyperparameters['agent'],
                                   training_hyperparams=self.hyperparameters['training'],
                                   verbose=self.verbose,
                                   pbar=pbar)
                    self.eval_agent_rewards.append(self._evaluate_total_agent_reward())
                self.eval_hist_rewards = self._evaluate_total_hist_reward()

    def inverse_scale_feature(self,
                              values,
                              labels):
        """De-bin and de-normalize feature values.

        Args:
            labels (list): list of feature labels.
            values (list): list of (scaled) feature values.
        
        Returns:
            list: Inverse transformation coefficient for all feature labels.
        """
        i_values = []
        for i, label in enumerate(labels):
            scaler = self.dh.minmax_scalars[label]
            val = np.array([v[i] for v in values])
            val = scaler.inverse_transform(val.reshape(-1, 1))
            i_values.append(val)
        # Transpose and convert to list
        i_values = np.concatenate(
            [np.expand_dims(v,1) for v in i_values],
            1).squeeze(-1).tolist()
        return i_values

    # TODO: From Giulia, for when we do the clean up, do we need to keep this here or can we put it in the evaluator?
    # TODO: Since it's part of the evaluation is better maybe
    def build_evaluation(self):
        """Save data for evaluation."""
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
        """Calculate the total reward obtained on the evaluation states using the agent's policy.
        
        Returns:
            float: Total (not scaled) cumulative reward.
        """
        # Get actions corresponding to agent's learned policy
        b_actions_agent = self.agent.predict_actions(self._eval_b_states)

        # De-bin the recommended actions
        actions_agent = self.env.debin_states(b_actions_agent, idxs=self._eval_action_dims)

        # Get reward based on agent policy
        rewards_agent = self.agent.predict_rewards(self._eval_b_states, b_actions_agent)
        
        # Inverse scale agent rewards
        rewards_agent = self.inverse_scale_feature(rewards_agent,
                                                   self.dh.reward_labels)

        return np.sum(rewards_agent)
    
    def _evaluate_total_hist_reward(self):
        """Calculate the total reward obtained on the evaluation states using the agent's policy.
        
        Returns:
            float: Total (not scaled) cumulative based on historical data.
        """
        # Get the binned actions
        b_actions = self.env.bin_states(self._eval_actions, idxs=self._eval_action_dims)

        # Get reward based on historical policy
        rewards_hist = self.agent.predict_rewards(self._eval_b_states, b_actions)

        # Inverse scale agent rewards
        rewards_hist = self.inverse_scale_feature(rewards_hist,
                                                  self.dh.reward_labels)

        return np.sum(rewards_hist)

    def _get_bins(self):
        """Get the bins for the states and actions.
        """
        state_labels = self.dh.state_labels
        action_labels = self.dh.action_labels

        bins = []
        for label in state_labels:
            bins.append(self.hyperparameters['dimensions']['states'][label])
        for label in action_labels:
            bins.append(self.hyperparameters['dimensions']['actions'][label])
        return bins
