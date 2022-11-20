from matplotlib import pyplot as plt
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

class CartPole(object):
    """Cart Pole environment, detailed in:
    https://www.gymlibrary.dev/environments/classic_control/cart_pole/
    """    
    def __init__(self, mode="Stochastic"):
        # create the environment
        if mode.startswith("Stochastic"):
            is_slippery=True
        elif mode.startswith("Deterministic"):
            is_slippery=False
            
        self.env = gym.make('CartPole-v1', render_mode="rgb_array")
        
        # define actions parameters
        self.action_space = self.env.action_space
        self.action_space.seed(SEED)
        self.actions = ["LEFT", "RIGHT"]
        self.state_space = self.env.observation_space
        
if __name__ == "__main__":
    cartpole_env = CartPole()
    cartpole_env.env.reset()
    cartpole_env.state_space.shape
    plt.imshow(cartpole_env.env.render())
    