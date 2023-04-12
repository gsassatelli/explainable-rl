# Import environment and agent
from src.foundation.agent import QLearningAgent
from src.foundation.environment import StrategicPricingMDP
import ipdb 

class Engine:

    __slots__ = ["dh", "agent_type", "env_type", "agent", "env", "gamma",
                 "episode_flag", "num_episodes", "num_steps", "policy", 
                 "q_table", "bins", "train_test_split"]

    def __init__(self, 
                 dh,
                 agent_type,
                 env_type,
                 num_episodes,
                 num_steps,
                 bins,
                 train_test_split,
                 gamma=0.9):
        """Initilize engine class.

        Args:
            dh (DataHandler): DataHandler to be given to the Environment
            agent_type (str): Type of agent to initialize
            env_type (int): Type of environment to initialize
            num_episodes (int): Number of episodes to train the agent for
            num_steps (int): Number of steps per episode
            bins (int): List of bins per state/action to discretize the state
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

        # Parameters of the agent
        self.policy = None
        self.q_table = None

        self.bins = bins

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
            self.agent = QLearningAgent(self.env, gamma=0.9)
            self.agent.create_tables()

    def create_env(self):
        """Create an env and store it in Engine.

        """
        # Initialize environment
        if self.env_type == "strategic_pricing":
            self.env = StrategicPricingMDP(self.dh, self.bins)
            self.env.initialise_env()

    def train_agent(self):
        """Train the agent for a chosen number of steps and episodes.

        """
        # Fit the agent
        self.agent.fit(self.num_episodes, self.num_episodes)

    def get_results(self):
        """Get the results of training.

        TODO: Next sprint to compare 2 agents
              This could be the average return after convergence.
        """
        pass

    def save_parameters(self):
        """Save the parameters learned during training.

        This could be e.g. the q-values, the policy, or any other learned parameters.

        TODO: Not sure this function is needed, can call directly agent
        TODO: Epsilon greedy policy already contains q-values, remove it?
        """
        # Save parameters of the trained agent to predict
        self.policy = self.agent.policy
        self.q_table = self.agent.q_table

    def _denorm_feature(self,
                        label,
                        bins,
                        values):
        """De-bin and de-normalize feature values.

        Args:
            label (list): name of the feature (state or action)
            values (list): normalized feature values
        
        Returns:
        """
        scaler = self._minmax_scalars[label]
        values = np.array(values) # convert to array
        d_values = scaler.inverse_transform(
                values.reshape(-1, 1) / bins)
        return d_values

    def evaluate_agent(self,
                 epsilon=0):
        """Evaluate the learned policy for the test states

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

        # Get the binned actions
        b_actions =  self.env.bin_states(actions, idxs=action_dims)

        # Get actions corresponding to agent's learned policy
        b_actions_agent = self.agent.predict_actions(b_states)

        # Get reward based on agent policy
        rewards_agent = self.agent.predict_rewards(b_states, b_actions_agent)

        # Get reward based on historical policy
        rewards_hist = self.agent.predict_rewards(b_states, b_actions)

        # De-bin the recommended actions
        actions_agent = self.env.debin_states(b_actions_agent, idxs=action_dims)

        # TODO: De-norm actions, states, and rewards

        return states, actions, rewards_hist, actions_agent, rewards_agent
