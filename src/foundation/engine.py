# Import environment and agent
from src.foundation.agent import QLearningAgent
from src.foundation.environment import StrategicPricingMDP

class Engine:

    __slots__ = ["dh", "agent_type", "env_type", "agent", "env", "gamma",
                 "episode_flag", "num_episodes", "num_steps", "policy", "q_table", "bins"]

    def __init__(self, 
                 dh,
                 agent_type,
                 env_type,
                 num_episodes,
                 num_steps,
                 bins,
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
        self.agent.fit(self.num_episodes, self.num_steps)

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


    def evaluate(self,
                 state):
        """Evaluate the learned policy at a particular state.

        Args:
            state: state for which an action needs to be predicted.
        Returns:
            action_reward: action and reward for a given state

        TODO: ensure that here output is action with max q values (NO exploration)
        """
        # Get both action & reward
        action_reward = self.agent._epsilon_greedy_policy(state)
        return action_reward

