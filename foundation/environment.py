class MDP:
    def _init_(self, state_state, action_space, reward_function, discount_factor):
        self.state_state = state_state
        self.action_space = action_space
        self.reward_function = reward_function
        self.discount_factor = discount_factor
    
    def create_env(self):
        pass

    def step(self):
        pass
