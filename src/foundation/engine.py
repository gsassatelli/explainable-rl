# Import environment and agent
from src.agents.q_learner import QLearningAgent
from src.agents.sarsa import SarsaAgent
from src.agents.sarsa_lambda import SarsaLambdaAgent
from src.agents.double_q_learner import DoubleQLearner
from src.environments.strategic_pricing import StrategicPricingMDP



# TODO: Ludo thinks we should just pass the Engine the whole hyperparam dictionary and that it should also create the data handler.
class Engine:

    __slots__ = ["dh", "agent_type", "env_type", "agent", "env", "gamma",
                 "episode_flag", "num_episodes", "num_steps", "policy", "q_table", "bins", "verbose"]

    def __init__(self, 
                 dh,
                 agent_type,
                 env_type,
                 num_episodes,
                 num_steps,
                 bins,
                 gamma=0.9,
                 verbose=False):
        """Initialise engine class.

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

        self.verbose = verbose

    def create_world(self):
        """Create the Agent and MDP instances for the given task.

        """
        # Create chosen environment
        if self.verbose:
            print("Initialize environment")
        self.create_env()
        
        # Create chosen agent
        if self.verbose:
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
        if self.env_type == "strategic_pricing":
            self.env = StrategicPricingMDP(self.dh, self.bins, verbose=self.verbose)

        else:
            raise NotImplementedError

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

