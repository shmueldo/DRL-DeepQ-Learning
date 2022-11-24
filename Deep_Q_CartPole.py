from matplotlib import pyplot as plt
import numpy as np
import gym
from Models import *
from Environments import *
from QAlgorithms import *

    
if __name__ == "__main__":
    
    cart_pole_env = CartPole()
    lr_steps = []
    lr_rewards = []

    dqn = DeepQLearning(environment=cart_pole_env,
                        q_learning_rate=0.0001,
                        discount_factor=0.95,
                        decaying_rate=0.95,
                        epsilon=0.25,
                        model_type=ThreeLayersModel,
                        optimizer=tf.keras.optimizers.Adam,
                        criterion=tf.keras.losses.MSE,
                        net_learning_rate=0.002)
    rewards, averaged_steps, averaged_rewards = dqn.train_agent(num_of_episodes=500,
                                                                weights_assign_num=10,
                                                                training_num=1,
                                                                batch_size=128,
                                                                epochs=1)

            
    fig = plt.figure()
    fig.suptitle("rewards and steps")
    ax = fig.subplots(1,3)

    ax[0].plot(averaged_steps)
    ax[0].set_xlabel("episodes [*100]")
    ax[0].set_ylabel("averaged steps")
    
    # for i,rewards in enumerate(lr_rewards):
    ax[1].plot(rewards)
    ax[1].set_xlabel("episodes")
    ax[1].set_ylabel("reward")

    ax[2].plot(averaged_rewards)
    ax[2].set_xlabel("episodes [*100]")
    ax[2].set_ylabel("averaged reward")
    plt.show()

    print("ënd")