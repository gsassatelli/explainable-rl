# TODO: build Environment super class
# TODO: build Agent super class
# TODO: build Environment super class

class Agent:
    __slots__ = ['env', 'gamma']

    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
