from matplotlib import pyplot as plt
import numpy as np
import gym
from Models import *
from Environments import *
from QAlgorithms import *
from datetime import datetime
import tensorflow as tf
    
    
if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.subplots(1,2)
    ax[0].set_title("mean reward over 100 episodes")
    ax[0].set_xlabel("episodes")
    ax[0].set_ylabel("averaged reward")
    ax[1].set_title("overall rewards")
    ax[1].set_xlabel("episodes")
    ax[1].set_ylabel("reward")
    for lr in [0.01, 0.005, 0.001, 0.0005]:
        cart_pole_env = CartPole()
        lr_steps = []
        lr_rewards = []
        dqn = DeepQLearning(environment=cart_pole_env,
                            q_learning_rate=0.0001,
                            discount_factor=0.95,
                            decaying_rate=0.95,
                            epsilon=0.9,
                            model_type=ThreeLayersModel,
                            optimizer=tf.keras.optimizers.Adam,
                            criterion=tf.keras.losses.MSE,
                            net_learning_rate=lr,
                            model_name="lr_sim")
        rewards, averaged_steps, averaged_rewards, losses = dqn.train_agent(num_of_episodes=350,
                                                                    weights_assign_num=10,
                                                                    training_num=4,
                                                                    batch_size=128,
                                                                    epochs=1)

        ax[0].plot(averaged_rewards, label=lr)
        ax[1].plot(rewards, label=lr)
    ax[0].legend()
    ax[1].legend()
    plt.show()

    print("ënd")