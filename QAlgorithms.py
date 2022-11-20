from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import numpy as np
import gym
from Models import *
from Environments import *

SEED = 42
np.random.seed(SEED)

class QLearning(object):
    def __init__(self, environment, q_learning_rate:float,
                 discount_factor:float, decaying_rate:float,
                 epsilon:float):
        self.environment = environment
        self.q_learning_rate = q_learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.decaying_rate = decaying_rate
        try:
            self.dim = (self.environment.state_space.n, self.environment.action_space.n)
        except AttributeError:
            self.dim = (4, self.environment.action_space.n)
        self.q_values = self.initialize_q()

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
class DeepQLearning(QLearning):
    def __init__(self, environment, q_learning_rate:float, discount_factor:float,
                 decaying_rate:float, epsilon:float, model_type:tf.keras.Model,
                 criterion, optimizer, net_learning_rate):
        super().__init__(environment, q_learning_rate, discount_factor,
                         decaying_rate, epsilon)
        
        # define the experience reply deque
        self.maximal_reply_size = 10000
        self.minimal_reply_size = 1000
        self.exp_reply_deque    = deque(maxlen=self.maximal_reply_size)
        
        # initialize models and equalize weights
        self.main_model         = self.model_init(model_type, criterion, optimizer, net_learning_rate)
        self.target_model       = self.model_init(model_type, criterion, optimizer, net_learning_rate)
        self.assign_weights()
        
    def assign_weights(self):
        main_model_weights = self.get_weights()
        self.target_model.set_weights(main_model_weights)
    
    def model_input_reshape(self, input_state):
        """ reshape single state input as a model input

        Args:
            input_state (state): the input state
        """
        return input_state.reshape([1, input_state.shape[0]])
    
    def model_init(self, model_type, criterion, optimizer, net_learning_rate):
        model = model_type(self.dim[0], self.dim[1])
        model.compile(optimizer=optimizer(net_learning_rate),
                      loss= criterion, 
                      metrics=[tf.keras.metrics.MeanSquaredError])
                    #   metrics=['mse'])
        return model
    
    def sample_action(self, current_state):
        odd = np.random.choice(2, 1, p=[self.epsilon, 1 - self.epsilon])
        if odd == 1:
            return np.argmax(self.main_model.predict(self.model_input_reshape(current_state)).flatten()) 
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
    
    def q_update(self, current_state, current_action, target):
        # calculate current model "q" function
        model_q_value = self.main_model.predict(self.model_input_reshape(current_state)).flatten()
        
        # calculate the modified Bellman equation given main model output
        q = (1 - self.q_learning_rate) * model_q_value[current_action] + self.q_learning_rate * target
        
        # update the model current action output
        model_q_value[current_action] = q
        return model_q_value[current_action]
    
    def generate_training_database(self, batch_size):
        """generates datasets for the main model training \
           where the input data is the current states and the "labels" \
           are updated modified q values, based on Bellman equation

        Args:
            batch_size (int): size of sampled mini-batch (from experience reply deque)

        Returns:
            observations (NDArray): observations dataset
            q_values     (NDArray): modified q values dataset
        """        
        observations = []
        q_values     = []
        
        mini_batch = self.sample_batch(batch_size)
        
        for exp_sample in mini_batch:
            # assign variables for iteration and model
            current_state, current_action, reward, done, next_state = exp_sample
            observations.append(current_state)
            
            # get target based on target model
            target = self.get_target(next_state, reward, done)
            
            # update the model current action i.e. modified q-value and append to q_values
            q_values.append(self.q_update(current_state, current_action, target))
        
        return np.array(observations), np.array(q_values) 
        
    def train(self, batch_size, epochs):
        if len(self.exp_reply_deque) >= self.minimal_reply_size:
            # generate datasets for training
            observations, q_values = self.generate_training_database(batch_size)
            # train model
            self.main_model.fit(x=observations, y=q_values,
                                batch_size=batch_size,
                                epochs=epochs,
                                shuffle=True)
    
    def test_agent(self, state):
        action = np.argmax(self.main_model.predict(self.model_input_reshape(state)).flatten())
        next_state, reward, done = self.env_step(action)
    
    def train_agent(self, num_of_episodes, weights_assign_num, training_num,
                    batch_size = 64, epochs=10):
        rewards          = []
        averaged_steps   = []
        averaged_rewards = []
        for episode in range(num_of_episodes):
        
            # reset environment 
            current_state = self.environment.env.reset()[0]
            
            # initialize episode parameters
            done = False
            step = 0
            update_counter = 0
            
            # update epsilon value following decaying epsilon greedy method 
            self.update_epsilon()
            
            while True:
                # increment steps
                step += 1
                update_counter += 1
                
                # sample action based on epsilon-greedy method
                current_action = self.sample_action(current_state)
                
                # apply environment step
                next_state, reward, done = self.env_step(current_action)
                
                # enrich the experience reply deque
                self.store_past_exp(current_state=current_state,
                                    current_action= current_action,
                                    reward=reward,
                                    done=done,
                                    next_state=next_state)
                
                # train the main model
                if update_counter % training_num == 0:
                    self.train(batch_size=batch_size, epochs=epochs)

                # move on to the next step 
                current_state = next_state
                rewards.append(reward)
                
                # check if game terminated
                if done:
                    steps_of_hundred_episodes.append(step)
                    if update_counter >= weights_assign_num:
                        self.assign_weights()
                        update_counter = 0
                    break
            
            if (episode % 100 == 0) and (episode != 0):
                # calculate average number of steps every 100 episodes: 
                averaged_steps.append(np.mean(np.array(steps_of_hundred_episodes)))
                steps_of_hundred_episodes = []

                # calculate average reward every 100 episodes: 
                rewards_of_hundred_episodes = rewards[-100:]
                averaged_rewards.append(np.mean(np.array(rewards_of_hundred_episodes)))
        
        # end environment activity
        self.environment.env.close()

    def test_agent(self):
        # reset environment
        rewards = [] 
        step = 0
        current_state = self.environment.env.reset()[0]
        done = False
        plt.imshow(self.environment.env.render())
        
        while True:
            # increment steps
            step += 1
            update_counter += 1
            
            # perform action based on policy
            current_action = np.argmax(self.main_model.predict(self.model_input_reshape(current_state)).flatten())
            
            plt.imshow(self.environment.env.render())
            
            # apply environment step
            next_state, reward, done = self.env_step(current_action)

            # move on to the next step 
            current_state = next_state
            rewards.append(reward)
            
            # check if game terminated
            if done:
                print("game ended after {} steps, overall reward is {}".format(step, np.sum(rewards)))
                break
        
        # end environment activity
        self.environment.env.close()

if __name__ == "__main__":
    cart_pole_env = CartPole()
    dqn = DeepQLearning(environment=cart_pole_env,
                        q_learning_rate=0.0005,
                        discount_factor=0.95,
                        decaying_rate=0.999,
                        epsilon=1,
                        model_type=ThreeLayersModel,
                        optimizer=tf.keras.optimizers.Adam,
                        criterion=tf.keras.losses.MSE,
                        net_learning_rate=0.0005)
