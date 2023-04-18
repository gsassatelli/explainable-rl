# Import packages
import numpy as np
import sparse
import random
from datetime import datetime
import ipdb
from tqdm import tqdm

from src.foundation.utils import *
from src.foundation.super_classes import Agent


class QLearningAgent(Agent):
    """Agent class to store and update q-table.
    """

    __slots__ = ['Q', 'state_to_action', 'state', 'Q_num_samples']

    def __init__(self, env, gamma):
        super().__init__(env, gamma)
        """Initialise the agent class.
        
        Args:
            env (MDP): MDP object.
            gamma (float): Discount factor.
        """
        self.Q = None
        self.state_to_action = None
        self.Q_num_samples = None
        self.state = None

    def fit(self, n_episodes, n_steps, lr=0.1, lr_decay=0.05, lr_min=0.01,
            epsilon=0.1, epsilon_decay=0.05, epsilon_min=0.01, verbose=False):

        """Fit agent to the dataset.

        Args:
            n_episodes (int): number of episodes.
            n_steps (int): number of steps per episode.
            lr (float): learning rate.
            lr_decay (float): learning rate decay.
            lr_min (float): minimum learning rate.
            epsilon (float): epsilon-greedy policy parameter.
            epsilon_decay (float): epsilon decay.
            epsilon_min (float): minimum epsilon.
            verbose (bool): print training information.
        """
        if verbose:
            print("Apply q-learning and update q-table")

        for i in tqdm(range(n_episodes)):

            self.state = self.env.reset()

            for i in range(n_steps):
                done = self._step(epsilon=epsilon, lr=lr)
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

    def create_tables(self,
                      verbose=False):
        """Initialize the agent.

        This resets the environment, creates the q-table and the state to
        action mapping.

        Args:
            verbose (bool): print information.
        """
        self.env.reset()
        if verbose:
            print("Create q-table")
        # create q-table
        self._init_q_table()
        self.state_to_action = self.env.state_to_action
    
    def uncertainty_informed_policy(self, state=None, epsilon=0.1, alpha=0.9):
        """Get epsilon greedy policy that favours more densely populated state-action pairs. 

        Args:
            state (list): current state of the agent.
            epsilon (float): the exploration parameter.
            alpha (float): the exploitation parameter.

        Returns:
            action (int): selected action.
        """
        if state is None:
            state = self.state

        state_str = self._convert_to_string(state)
        index = tuple(list(state))
        possible_actions = self.env.state_to_action[state_str]
        state_action_counts = {}
        for possible_action in possible_actions:
            possible_state_action_str = self._convert_to_string(state + [possible_action])
            counts = self.env.bins_dict[possible_state_action_str][0]
            # Count number of times an state-action pair occurred 
            state_action_counts[str(possible_action)] = counts

        # Get weights for counts
        action_weights = {key:float(value)/sum(state_action_counts.values()) for (key, value) in state_action_counts.items()}

        # q_value for each action
        q_values = self.Q[index].todense()

        if random.random() > epsilon:
            action = np.argmax(q_values)
        else:
            action = int(np.random.choice(list(action_weights.keys()), 1, list(action_weights.values())))
        
        return action
        
    
    def _epsilon_greedy_policy(self,
                               state=None,
                               epsilon=0.1):
        """Get the epsilon greedy action.

        Args:
            state (list): current state of the agent.
            epsilon (float): the exploration parameter.

        Returns:
            action (int): selected action.
        """
        if state is None:
            state = self.state

        state_str = self._convert_to_string(state)

        index = tuple(list(state))

        q_values = self.Q[index].todense()
            
        if random.random() > epsilon:
            action = np.argmax(q_values)
        else:
            action = random.choice(list(self.state_to_action[state_str]))
        
        return action
       

    @staticmethod
    def _convert_to_string(state):
        return ",".join(str(s) for s in state)

    def _init_q_table(self):
        """Initialize the q-table with zeros.
        """
        self.Q = sparse.DOK(self.env.bins)
        self.Q_num_samples = sparse.DOK(self.env.bins)

    def _step(self, epsilon, lr):
        """Perform a step in the environment.

        Args:
            epsilon (float): epsilon-greedy policy parameter.
            lr (float): learning rate.

        Returns:
            done: boolean indicating whether the episode is finished.
        """
        # action = self._epsilon_greedy_policy(self.state,
        #                                      epsilon=epsilon)
        action = self.uncertainty_informed_policy(self.state,
                                             epsilon=epsilon,
                                             alpha=0.9)
        state, next_state, reward, done = self.env.step(self.state,
                                                        action)

        self._update_q_values(state, action, next_state, reward, lr, done)
        self.state = next_state
        return done

    def _update_q_values(self,
                         state,
                         action,
                         next_state,
                         reward,
                         lr,
                         done):
        """Update the Q table using the Bellman equation.

        Args:
            state (list): current state of the agent.
            action (int): selected action.
            next_state (list): next state of the agent.
            reward (float): reward for the selected action.
            lr (float): learning rate.
        """

        index_current = tuple(list(state) + [action])
        
        q_current = self.Q[index_current]
        
        if done == True:
            next_state = state

        index_next_state = tuple(next_state)

        q_next = np.max(self.Q[index_next_state].todense())    
        
        self.Q[index_current] = \
            q_current + lr * (reward + self.gamma * q_next - q_current)

        self.Q_num_samples[index_current] += 1
    
    def predict_actions(self,
                states, 
                epsilon=0):
        """ Predict action for a list of states using epislon-greedy policy.
        
        Args:
            states (list): States (binned).
            epislon (float): Epislon of epislon-greedy policy.
                Defaults to 0 for pure exploitation.
        
        Returns:
            actions (list): List of recommended actions
        """
        actions = []
        for state in states:
            action = self._epsilon_greedy_policy(state, epsilon)
            actions.append([action])

        return actions
    
    def predict_rewards(self,
                    states,
                    actions):
        """ Predict reward for a list of state-actions.
         
        This function uses the avg reward matrix (which simulates a real-life scenario)
        
        Args:
            states (list): States (binned).
            actions (list): Actions (binned).
        
        Returns:
            rewards (list): List of recommended actions
        """

        rewards = []
        for state, action in zip(states, actions):
            _, _, reward, _ = self.env.step(state, action)
            rewards.append([reward[0]])
        
        return rewards

