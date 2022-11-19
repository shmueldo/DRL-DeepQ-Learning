from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import numpy as np
import gym

SEED = 42
np.random.seed(SEED)

class QLearning(object):
    def __init__(self, environment, q_learning_rate:float,
                 discount_factor:float, decaying_rate:float,
                 epsilon:float):
        self.environment = environment
        self.q_values = self.initialize_q()
        self.q_learning_rate = q_learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.decaying_rate = decaying_rate
        self.dim = (self.environment.state_space.n, self.environment.action_space.n)

    def initialize_q(self):
        return np.zeros(self.dim)

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
    
    def sample_action(self, current_state):
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
        q = (1 - self.q_learning_rate) * self.q_values[current_state, current_action] + self.q_learning_rate * target
        err = self.q_values[current_state, current_action] - q
        self.q_values[current_state, current_action] = q
        return err


class ThreeLayersModel(tf.keras.Model):
    
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.input_layer    = Dense(32, input_shape=input_dim, activation='relu')
    self.hidden         = Dense(16, activation='relu')
    self.output_layer   = Dense(output_dim, activation='linear')
    self.dropout        = tf.keras.layers.Dropout(0.5)

  def call(self, inputs):
    x = self.input_layer(inputs)
    x = self.hidden(x)
    return self.output_layer(x)

class DeepQLearning(QLearning):
    def __init__(self, environment, q_learning_rate:float, discount_factor:float,
                 decaying_rate:float, epsilon:float, model_type:tf.keras.Model,
                 criterion, optimizer, net_learning_rate):
        super().__init__(environment, q_learning_rate, discount_factor,
                         decaying_rate, epsilon)
        self.exp_reply_deque    = deque(maxlen=50000)
        self.main_model         = self.model_init(model_type, criterion, optimizer, net_learning_rate)
        self.target_model       = self.model_init(model_type, criterion, optimizer, net_learning_rate)
        
    def input_reshape(self, input_state):
        """ reshape single state input as a model input

        Args:
            input_state (state): the input state
        """        
        return input_state.reshape([1, input_state.shape[0]])
    
    def model_init(self, model_type, criterion, optimizer, net_learning_rate):
        model = model_type(self.dim[0], self.dim[1])
        model.compile(loss= criterion, optimizer=optimizer(net_learning_rate), metrics=['mse'])
        return model
    
    def sample_action(self, current_state):
        odd = np.random.choice(2, 1, p=[self.epsilon, 1 - self.epsilon])
        if odd == 1:
            return np.argmax(self.main_model.predict(self.input_reshape(current_state)).flatten()) 
        else:
            return self.environment.action_space.sample()

    def sample_batch(self, batch_size):
        return np.random.choice(self.exp_reply_deque, batch_size)
    
    def store_past_exp(self, current_state, current_action, reward, done, next_state):
        self.exp_reply_deque.append([current_state, current_action, reward, done, next_state])
    
    def get_target(self, next_state, reward, done):
        if not(done):
            target = reward + self.discount_factor * np.max(self.target_model.predict(next_state))
        else:
            target = reward
        return target
    
    def train_agent(self, batch_size):
        # consider to apply condition on the size of the experience reply deque
      