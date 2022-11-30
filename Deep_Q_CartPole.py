from matplotlib import pyplot as plt
import numpy as np
import os
from Models import *
from Environments import *
from QAlgorithms import *
from datetime import datetime
import tensorflow as tf
import h5py
    
    
if __name__ == "__main__":
    TEST = False 
    cart_pole_env = CartPole()
    print("DeepQLearning")
    dqn = DeepQLearning(environment=cart_pole_env,
                            q_learning_rate=0.0001,
                            discount_factor=0.95,
                            decaying_rate=0.95,
                            epsilon=0.9,
                            model_type=ThreeLayersModel,
                            optimizer=tf.keras.optimizers.Adam,
                            criterion=tf.keras.losses.MSE,
                            net_learning_rate=0.002,
                            model_name="FiveLayersModel")
    
    rewards, averaged_steps, averaged_rewards, losses = dqn.train_agent(num_of_episodes=3,
                                                                weights_assign_num=5,
                                                                training_num=1,
                                                                batch_size=128,
                                                                epochs=1)
    if TEST:
        dqn.main_model.load_weights(os.getcwd() + r"\Model_weights\optimal_checkpoint\DDQN_weights")
        dqn.test_agent(150)
    
    with open(os.getcwd() + r"\lists\rewards_50weights_assign_num.npy", 'wb') as f:
        np.save(f, np.array(rewards))

    with open(os.getcwd() + r"\lists\averaged_rewards_50weights_assign_num.npy", 'wb') as f:
        np.save(f, np.array(averaged_rewards))
    
    fig = plt.figure()
    ax = fig.subplots(1,2)
    ax[0].set_title("mean reward over 100 episodes")
    ax[0].set_xlabel("episodes")
    ax[0].set_ylabel("averaged reward")
    ax[1].set_title("overall rewards")
    ax[1].set_xlabel("episodes")
    ax[1].set_ylabel("reward")
        
    ax[0].plot(averaged_rewards)
    ax[1].plot(rewards)
    
    ax[0].legend()
    ax[1].legend()
    plt.show()

    print("ënd")