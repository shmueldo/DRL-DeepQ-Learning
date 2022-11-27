from matplotlib import pyplot as plt
import tensorflow as tf
from collections import deque
import numpy as np
import gym
from Models import *
from Environments import *
import random
import time
from tqdm import tqdm
import os
from ModifiedTensorBoard import *
from datetime import datetime

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(42)

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
            self.dim = (self.environment.state_space.shape[0], self.environment.action_space.n)
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
        # do not nullify epsilon for keep exploring
        if self.epsilon < 0.05:
            self.epsilon = 0.05

            
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
                 criterion, optimizer, net_learning_rate, model_name):
        super().__init__(environment, q_learning_rate, discount_factor,
                         decaying_rate, epsilon)
        
        
        # define the experience reply deque
        self.maximal_reply_size = 5000
        self.minimal_reply_size = 150
        self.exp_reply_deque    = deque(maxlen=self.maximal_reply_size)
        
        # initialize models and equalize weights
        self.main_model         = self.model_init(model_type, criterion, optimizer, net_learning_rate)
        self.target_model       = self.model_init(model_type, criterion, optimizer, net_learning_rate)
        self.assign_weights()
        self.model_name = model_name
        
        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(self.model_name,
                                               log_dir="{}logs/{}-{}".format(os.getcwd() +r"/",
                                                self.model_name, datetime.now().strftime("%d%m-%H%M")))

    def assign_weights(self):
        if len(self.exp_reply_deque) < self.minimal_reply_size:
            return
        try:
            main_model_weights = self.main_model.get_weights()
            self.target_model.set_weights(main_model_weights)
        except ValueError:
            print("Value error")
            pass

    def env_step(self, current_action):
        next_state, reward, terminated, truncated, _ = self.environment.env.step(current_action)
        done = terminated
        return next_state, reward, done
    
    def model_input_reshape(self, input_state):
        """ reshape single state input as a model input

        Args:
            input_state (state): the input state
        """
        return input_state.reshape([1, input_state.shape[0]])
    
    def model_init(self, model_type, criterion, optimizer, net_learning_rate):
        model = model_type((self.dim[0],), self.dim[1])
        model.compile(optimizer=optimizer(learning_rate = net_learning_rate),
                      loss= criterion, metrics=['mse'])
        return model
    
    def sample_action(self, current_state):
        odd = np.random.choice(2, 1, p=[self.epsilon, 1 - self.epsilon])
        if odd == 1:
            return np.argmax(self.main_model.predict(self.model_input_reshape(current_state), verbose=0).flatten())
        else:
            return self.environment.action_space.sample()

    def sample_batch(self, batch_size):
        return random.sample(self.exp_reply_deque, batch_size)
    
    def store_past_exp(self, current_state, current_action, reward, done, next_state):
        if len(self.exp_reply_deque) > self.maximal_reply_size:
            self.exp_reply_deque.popleft()
        self.exp_reply_deque.append([current_state, current_action, reward, done, next_state])
    
    def get_target(self, next_q_val, reward, done):
        if not(done):
            target = reward + self.discount_factor * np.max(next_q_val)
        else:
            target = reward
        return target
    
    def q_update(self, current_q_val, current_action, target):
        
        # # update the model current action output
        current_q_val[current_action] = target
        return current_q_val
    
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
        
        current_states  = np.array([batch_data[0] for batch_data in mini_batch])
        current_q_val   = self.main_model.predict(current_states, verbose = 0)
        
        next_states     = np.array([batch_data[4] for batch_data in mini_batch])
        next_q_val      = self.target_model.predict(next_states, verbose = 0)
        
        for i,exp_sample in enumerate(mini_batch):
            # assign variables for iteration and model
            current_state, current_action, reward, done, next_state = exp_sample
            observations.append(current_state)
            
            # get target based on target model
            target = self.get_target(next_q_val[i], reward, done)
            
            # update the model current action i.e. modified q-value and append to q_values
            q_values.append(self.q_update(current_q_val[i], current_action, target))
        
        return np.array(observations), np.array(q_values) 
        
    def train(self, batch_size, step):
        observations, q_values = self.generate_training_database(batch_size)
        # train model
        self.main_model.fit(x=observations, y=q_values,
                                   batch_size=batch_size,
                                   verbose=0,
                                   callbacks=[self.tensorboard])
        
    
    def train_agent(self, num_of_episodes, weights_assign_num, training_num,
                    batch_size = 64, epochs=10):
        rewards          = []
        losses             = []
        averaged_steps   = []
        averaged_rewards = []
        steps_of_hundred_episodes = []
        beat_mean_reward = 0
        update_counter = 0
        overall_steps = 0
        for episode in tqdm(range(num_of_episodes)):
            # Reset environment 
            current_state = self.environment.env.reset()[0]
            
            # initialize episode parameters
            done = False
            step = 0
            overall_reward = 0
            
            # Update epsilon value following decaying epsilon greedy method 
            self.update_epsilon()
            
            while True:
                #update tensor board step
                self.tensorboard.step = overall_steps
                
                # Increment steps
                step += 1
                overall_steps +=1
                update_counter += 1
                
                # Sample action based on epsilon-greedy method
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
                    if len(self.exp_reply_deque) >= self.minimal_reply_size and len(self.exp_reply_deque) >= batch_size:
                        self.train(batch_size=batch_size, step=overall_steps)

                # move on to the next step 
                current_state = next_state
                overall_reward += reward
                
                # truncate is reach 1000 steps
                if step >= 1000:
                    done = True
                
                # check if game terminated
                if done:
                    rewards.append(overall_reward)
                    print("episode:{}, steps:{}".format(episode, step))
                    steps_of_hundred_episodes.append(step)
                    if update_counter >= weights_assign_num:
                        self.assign_weights()
                        # print("model loaded")
                        update_counter = 0
                    break

            if (episode <= 100):
                mean_reward = np.mean(np.array(rewards))
            
            else:
                # calculate average reward every 100 episodes: 
                mean_reward = np.mean(np.array(rewards[-100:]))
            
            ## save model weights for better validation performences
            if mean_reward > beat_mean_reward:
                print("mean reward increased {}--->{}, saving model".format(beat_mean_reward, mean_reward))
                beat_mean_reward = mean_reward
                best_episode = episode
                
                ## Saving State Dict
                self.main_model.save_weights(os.getcwd() + r"\Model_weights\{}_weights".format(self.model_name))
            
                if mean_reward > 476:
                    break
            averaged_rewards.append(mean_reward)
            self.tensorboard.update_stats(Rewards_per_episode = overall_reward, mean_reward= mean_reward, step=episode)
        
        # end environment activity
        self.environment.env.close()
        return rewards, averaged_steps, averaged_rewards, losses

    def test_agent(self, num_of_episodes):
        rewards          = []
        averaged_steps   = []
        averaged_rewards = []
        steps_of_hundred_episodes = []
        for episode in range(num_of_episodes):
        
            # reset environment 
            current_state = self.environment.env.reset()[0]
            
            # initialize episode parameters
            done = False
            step = 0
            update_counter = 0
            overall_reward = 0
            # plt.imshow(self.environment.env.render())
            # plt.show()
            while True:
                # increment steps
                step += 1
                
                # Perform action based on policy
                current_action = np.argmax(self.main_model.predict(self.model_input_reshape(current_state), verbose=0).flatten())
                # print("current_action", current_action)
            
                # Apply environment step
                next_state, reward, done = self.env_step(current_action)
                # plt.imshow(self.environment.env.render())

                # Move on to the next step 
                current_state = next_state
                overall_reward += reward
               
                # Check if game terminated
                if done:
                    rewards.append(overall_reward)
                    # plt.imshow(self.environment.env.render())
                    steps_of_hundred_episodes.append(step)
                    print("game ended after {} steps, overall reward is {}".format(step, overall_reward))
                    break
            
            if (episode % 100 == 0) and (episode != 0):
                # calculate average number of steps every 100 episodes: 
                averaged_steps.append(np.mean(np.array(steps_of_hundred_episodes)))
                steps_of_hundred_episodes = []

                # calculate average reward every 100 episodes: 
                rewards_of_hundred_episodes = rewards[-100:]
                averaged_rewards.append(np.mean(np.array(rewards_of_hundred_episodes)))
        
        # end environment activity
        plt.plot(range(num_of_episodes), rewards)
        plt.plot(range(num_of_episodes), steps_of_hundred_episodes)
        plt.show()
        self.environment.env.close()

class DoubleDeepQLearning(DeepQLearning):
    def __int__(self, environment, q_learning_rate, discount_factor, decaying_rate, epsilon, epsilon_min,
                model_type:tf.keras.Model, criterion, optimizer, net_learning_rate):
        super().__init__(environment, q_learning_rate, discount_factor, decaying_rate, epsilon)

        self.nS = self.environment.observation_space.shape[0]
        self.nA = self.environment.action_space.n
        # define the experience reply deque
        self.maximal_reply_size = 50000
        self.minimal_reply_size = 256
        self.exp_reply_deque = deque(maxlen=self.maximal_reply_size)
        self.epsilon_min = epsilon_min

        self.model = self.model_init(model_type=model_type, criterion=criterion, optimizer=optimizer, net_learning_rate=net_learning_rate)
        self.model_target = self.model_init(model_type=model_type, criterion=criterion, optimizer=optimizer, net_learning_rate=net_learning_rate)
        self.assign_weights()
        self.loss = []
        self.train_end = 0

    def generate_training_database(self, batch_size):
        # Execute the experience replay
        minibatch = random.sample(self.exp_reply_deque, batch_size)  # Randomly sample from memory

        # Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = np.array(minibatch)
        st = np.zeros((0, self.nS))  # States
        nst = np.zeros((0, self.nS))  # Next States
        for i in range(len(np_array)):  # Creating the state and next state np arrays
            st = np.append(st, np_array[i, 0], axis=0)
            nst = np.append(nst, np_array[i, 3], axis=0)
        st_predict = self.model.predict(st)  # Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        nst_predict_target = self.model_target.predict(nst)  # Predict from the TARGET
        index = 0
        for current_state, current_action, reward, done, next_state in minibatch:
            x.append(current_state)
            # Predict from state
            nst_action_predict_target = nst_predict_target[index]
            nst_action_predict_model = nst_predict[index]
            target = self.get_target(done=done)
            target_f = st_predict[index]
            target_f[current_action] = target
            y.append(target_f)
            index += 1
        # Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size, self.nS)
        y_reshape = np.array(y)
        epoch_count = 1
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        # Graph Losses
        for i in range(epoch_count):
            self.loss.append(hist.history['loss'][i])
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.update_epsilon()

    def train_agent(self, num_of_episodes, weights_assign_num, training_num, batch_size=64, epochs=10):
        rewards = []
        epsilons = []
        test_episodes = 0
        for episode in range(num_of_episodes):
            state = self.environment.reset()
            state = np.reshape(state, [1, self.nS])  # Resize to store in memory to pass to .predict
            tot_rewards = 0
            for time in range(210):  # 200 is when you "solve" the game. This can continue forever as far as I know
                action = self.sample_action(state)
                next_state, reward, done, _ = self.environment.step(action)
                next_state = np.reshape(next_state, [1, self.nS])
                tot_rewards += reward
                self.store_past_exp(state, action, reward, done, next_state)  # Resize to store in memory to pass to .predict
                state = next_state
                # done: CartPole fell.
                # time == 209: CartPole stayed upright
                if done or time == 209:
                    rewards.append(tot_rewards)
                    epsilons.append(self.epsilon)
                    print("episode: {}/{}, score: {}, e: {}".format(episode, num_of_episodes, tot_rewards, dqn.epsilon))
                    break
                # Experience Replay
                if len(self.exp_reply_deque) > batch_size:
                    self.train(batch_size)
            # Update the weights after each episode (You can configure this for x steps as well
            self.assign_weights()
            # If our current NN passes we are done
            # I am going to use the last 5 runs
            if len(rewards) > 5 and np.average(rewards[-5:]) > 195:
                # Set the rest of the EPISODES for testing
                test_episodes = num_of_episodes - episode
                self.train_end = episode
                break


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
    dqn.train_agent(num_of_episodes=100, weights_assign_num= 4, training_num=1, batch_size= 1, epochs=10)
    print("ënd")