
import gym
import numpy as np
import time, pickle, os

env = gym.make('FrozenLake-v1', render_mode="human")

# epsilon = 0.9
# total_episodes = 10000
# max_steps = 100

# lr_rate = 0.81
# gamma = 0.96

# Q = np.zeros((env.observation_space.n, env.action_space.n))
    
# def choose_action(state):
#     action=0
#     if np.random.uniform(0, 1) < epsilon:
#         action = env.action_space.sample()
#     else:
#         action = np.argmax(Q[state, :])
#     return action

# def learn(state, state2, reward, action):
#     predict = Q[state, action]
#     target = reward + gamma * np.max(Q[state2, :])
#     Q[state, action] = Q[state, action] + lr_rate * (target - predict)

# # Start
# for episode in range(total_episodes):
#     state = env.reset()
#     t = 0
    
#     while t < max_steps:
#         env.render()

#         action = choose_action(state)  

#         state2, reward, done, info = env.step(action)  

#         learn(state, state2, reward, action)

#         state = state2

#         t += 1
       
#         if done:
#             break

#         time.sleep(0.1)

# print(Q)

# with open("frozenLake_qTable.pkl", 'wb') as f:
#     pickle.dump(Q, f)


import time 

# Number of steps you run the agent for 
num_steps = 1500

obs = env.reset()

for step in range(num_steps):
    # take random action, but you can also do something more intelligent
    # action = my_intelligent_agent_fn(obs) 
    action = env.action_space.sample()
    
    # apply the action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render the env
    env.render()

    # Wait a bit before the next frame unless you want to see a crazy fast video
    time.sleep(0.001)
    
    # If the epsiode is up, then start another one
    if terminated:
        print("terminated = {}".format(terminated))
        env.reset()
        
    if truncated:
        print("truncated = {}".format(truncated))
        env.reset()


# Close the env
env.close()