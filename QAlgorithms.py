from matplotlib import pyplot as plt
import numpy as np
import gym

SEED = 42
np.random.seed(SEED)

class QLearning(object):
    def __init__(self, environment, learning_rate:float,
                 discount_factor:float, decaying_rate:float,
                 epsilon:float):
        self.environment = environment
        self.q_values = self.initialize_q()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.decaying_rate = decaying_rate

    def initialize_q(self):
        dim = (self.environment.state_space.n, self.environment.action_space.n)
        return np.zeros(dim)

    def generate_lookup_table(self, episode):
        fig, ax = plt.subplots() 
        ax.set_axis_off()
        q_lookup_table = ax.table(cellText=np.round(self.q_values, decimals= 4),
                                    rowLabels=[i for i in range(self.environment.state_space.n)],
                                    colLabels=self.environment.actions,
                                    rowColours =["yellow"] * self.environment.state_space.n,  
                                    colColours =["yellow"] * self.environment.action_space.n, 
                                    cellLoc ='center',  
                                    loc ='upper left')
        ax.set_title('Q-value lookup table after {} episodes'.format(episode), fontweight ="bold")
    
    def epsilon_greedy_action(self, current_state):
            odd = np.random.choice(2, 1, p=[self.epsilon, 1 - self.epsilon])
            if odd == 1: return np.argmax(self.q_values[current_state, :])
            else: return self.environment.action_space.sample()
    
    def update_epsilon(self):
        """
            Updates epsilon value by multiplying it by decaying rate
        """
        self.epsilon *= self.decaying_rate
            
    def env_step(self, current_action):
        next_state, reward, terminated, truncated, _ = self.environment.env.step(current_action)
        done = terminated or truncated
        return next_state, reward, done
    
    def get_target(self, next_state, reward, done):      
        if not(done):
            target = reward + self.discount_factor * np.max(self.q_values[next_state, :])
        else:
            target = reward
        return target
            
    def q_update(self, current_state, current_action, target):
        q = (1 - self.learning_rate) * self.q_values[current_state, current_action] + self.learning_rate * target
        err = self.q_values[current_state, current_action] - q
        self.q_values[current_state, current_action] = q
        return err