import numpy as np
import sparse
import random

class Agent():
    def __init__(self, env):
        """Initialise the agent superclass."""
        self.env = env
        self.Q = None
        self.state_to_action = None
        self.gamma = 0.9
        self.initialize_agent()
        #self.state = None
        #self.q = ...
        self.rewards_per_episode = []
        # self.total_episode_reward = 0
        pass

    def fit(self, n_episodes, n_steps, lr=1):
        """Fit agent to dataset.

        TODO: decay. """
        
        
        for _ in range(n_episodes):
            #self.state = self.env.reset()
            # env reset is not working so implementing it here:
            self.state_str = random.choice(list(self.state_to_action.keys()))
            self.state = [int(s) for s in self.state_str.split(",")]
            for i in range(n_steps):
                action = self.epsilon_greedy_policy(self.state_str)
                state, next_state, reward, done = self.env.step(self.state,
                                                                action)

                self.update_q_values(state, action, next_state, reward, lr)

                self.state = next_state
                if done:
                    break
        print("Finished training :) !")
        print(f"Example Q-table for state {[0,11,0]}: {self.Q[0,11,0].todense()}")

    def update_q_values(self, state, action, next_state, reward, lr):
        """Update the q table of the agent using Bellman.
        
        TODO: implement part related to gamma (not necessary for now
        because we have a myopic env)
        """
        self.Q[state[0],state[1],state[2],action] = \
            (1-lr)*self.Q[state[0],state[1],state[2],action] + lr*(reward)

    def initialize_agent(self):
        """Reset the environment. Called by agent when the episode starts."""
        # reset environment
        self.env.reset()
        
        # create q-table
        coords = []
        for state_str, actions in self.env.state_to_action.items():
            state = [int(s) for s in state_str.split(",")]
            actions = list(actions)
            for action in actions:
                coords.append(state+[action])
        q_values = np.zeros(len(coords))
        coords = np.array(coords).T.tolist()
        
        # create COO (read only) matrix
        self.Q = sparse.COO(coords, q_values)
        
        # convert to DOK
        self.Q = sparse.DOK.from_numpy(self.Q.todense())
        
        # create state to action mapping
        self.state_to_action = self.env.state_to_action


    def epsilon_greedy_policy(self, state):
        """Returns the epsilon greedy action.
        
        TODO: implement the "epsilon part"
        """
        return random.choice(list(self.state_to_action[state]))
        
