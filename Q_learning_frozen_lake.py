from matplotlib import pyplot as plt
import numpy as np
import gym

class FrozenLakeGame(object):
    def __init__(self) -> None:
        # create the environment
        self.env = gym.make('FrozenLake-v1')
        
        # define actions parameters
        self.action_space = self.env.action_space
        self.actions = ["LEFT", "DOWN", "RIGHT", "UP"]
        self.state_space = self.env.observation_space

class QLearning(object):
    def __init__(self, environment, learning_rate:float,
                 discount_factor:float, decaying_rate:float,
                 epsilon:float) -> None:
        self.env = environment
        self.q_values = self.initialize_q()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.decaying_rate = decaying_rate

    def initialize_q(self):
        dim = (self.env.state_space.n, self.env.action_space.n)
        return np.zeros(dim)

    def generate_lookup_table(self):
        fig, ax = plt.subplots() 
        ax.set_axis_off()
        q_lookup_table = ax.table(cellText=self.q_values,
                                    rowLabels=[i for i in range(self.env.state_space.n)],
                                    colLabels=self.env.actions,
                                    rowColours =["yellow"] * self.env.state_space.n,  
                                    colColours =["yellow"] * self.env.action_space.n, 
                                    cellLoc ='center',  
                                    loc ='upper left')
        ax.set_title('Q-value approximation lookup table', fontweight ="bold")
    
    def epsilon_greedy_action(self, current_state):
            odd = np.random.choice(1, 1, p=[self.epsilon, 1 - self.epsilon])
            if odd == 1: return np.argmax(self.q_values[current_state, :])
            else: return self.env.action_space.sample()
    
    def get_target(self, current_action:ActType, current_state:ObsType)-> tuple(double, bool):
        next_state, reward, terminated, truncated, _ = self.env.step(current_action)
        done = terminated and truncated
        if not(done):
            target = reward + self.discount_factor * np.max(self.q_values[next_state, :])
        else:
            target = reward
        return next_state, target, done
            
    def q_update(self, current_state, current_action, target) -> float:
        q = (1 - self.learning_rate) * self.q_values[current_state, current_action] + self.learning_rate * target
        err = self.q_values[current_state, current_action] - q
        self.q_values[current_state, current_action] = q
        return err

        
if __name__ == "__main__":        
    frozen_lake = FrozenLakeGame()
    q_algo = QLearning(frozen_lake)
    q_algo.generate_lookup_table()
    plt.show()
    
    num_of_episodes = 1000
    for episode in range(num_of_episodes):
        current_state = q_algo.env.reset()
        done = False
        while (not(done)):
            current_action = q_algo.epsilon_greedy_action(current_state)
            next_state, target, done = q_algo.get_target(current_state, current_action)
            err = q_algo.q_update(current_state, current_action, target)
            current_state = next_state
        
        


# # action spaces parameters
# # action_space = env.action_space
# action_space_length = action_space.n
# initial_action = action_space.start

# # state spaces parameters
# state_space = env.observation_space
# state_space_length = state_space.n
# initial_state = state_space.start

# # Generate Q-values matrix
# Q = np.zeros((state_space_length, ))


# state_space = env.observation_spac