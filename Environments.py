import gym

SEED = 42
class FrozenLakeGame(object):
    def __init__(self, mode="Stochastic"):
        # create the environment
        if mode.startswith("Stochastic"):
            is_slippery=True
        elif mode.startswith("Deterministic"):
            is_slippery=False

        self.env = gym.make('FrozenLake-v1', render_mode="rgb_array", is_slippery=is_slippery)
        
        # define actions parameters
        self.action_space = self.env.action_space
        self.action_space.seed(SEED)
        self.actions = ["LEFT", "DOWN", "RIGHT", "UP"]
        self.state_space = self.env.observation_space