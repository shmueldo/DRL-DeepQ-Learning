from matplotlib import pyplot as plt
import numpy as np
import gym
from Environments import *
from QAlgorithms import *

SEED = 42
np.random.seed(SEED)
    
if __name__ == "__main__":

    frozen_lake = FrozenLakeGame()
    lr_steps = []
    lr_rewards = []
    lr_list = [0.001, 0.01, 0.05, 0.1, 0.5, 0.95]
    for lr in lr_list:
        q_algo = QLearning(environment=frozen_lake, learning_rate= lr,
                        discount_factor=0.9, decaying_rate=0.999,
                        epsilon= 0.9) 
                            
        num_of_episodes = 5001
        rewards = []
        averaged_rewards = []
        rewards_of_hundred_episodes = []
        averaged_steps = []
        steps_of_hundred_episodes = []
        
        for episode in range(num_of_episodes):
            print("episode: {}".format(episode))
            
            ## print Q lookup table
            # if episode in [499, 1999, 4999]:
            #     q_algo.generate_lookup_table(episode + 1)

            # reset environment 
            current_state = q_algo.environment.env.reset()[0]
            
            # initialize episode parameters
            done = False
            step = 0
            
            # update epsilon value following decaying epsilon greedy method 
            q_algo.update_epsilon()
            
            while True:
                # increment step
                step += 1
                
                # sample action based on epsilon-greedy method
                current_action = q_algo.epsilon_greedy_action(current_state)
                
                # apply environment step
                next_state, reward, done = q_algo.env_step(current_action)
                
                # calculate target
                target = q_algo.get_target(next_state, reward, done)
                
                # update Q value
                err = q_algo.q_update(current_state, current_action, target)

                # move on to the next step 
                current_state = next_state
                
                # check if game terminated
                if done:
                    # append target to overall reward
                    rewards.append(target)
                    
                    # if the agent reached destination:
                    if target == 1: steps_of_hundred_episodes.append(step)
                    
                    # if agent failed:
                    else: steps_of_hundred_episodes.append(100)
                    break
            
            if (episode % 100 == 0) and (episode != 0):
                # calculate average number of steps every 100 episodes: 
                averaged_steps.append(np.mean(np.array(steps_of_hundred_episodes)))
                steps_of_hundred_episodes = []

                # calculate average reward every 100 episodes: 
                rewards_of_hundred_episodes = rewards[-100:]
                averaged_rewards.append(np.mean(np.array(rewards_of_hundred_episodes)))
        
        # end environment activity
        q_algo.environment.env.close()
        lr_steps.append(averaged_steps)        
        lr_rewards.append(averaged_rewards)
            
    fig = plt.figure()
    ax = fig.subplots(1,2)
    for i, steps in enumerate(lr_steps):
        ax[0].plot(list(range(len(steps))), steps, label = "lr={}".format(lr_list[i]))
    plt.legend()
    
    for i,rewards in enumerate(lr_rewards):
        ax[1].plot(list(range(len(rewards))), rewards, label = "lr={}".format(lr_list[i]))
    plt.legend()
    plt.show()
    #     ax[1].scatter(range(len(averaged_rewards)), averaged_rewards)
    #     ax[2].plot(rewards)
    # plt.show()
