from matplotlib import pyplot as plt
import numpy as np
import gym

SEED = 42
np.random.seed(SEED)
    
class FrozenLakeGame(object):
    def __init__(self):
        # create the environment
        self.env = gym.make('FrozenLake-v1', render_mode="rgb_array")
        
        # define actions parameters
        self.action_space = self.env.action_space
        self.action_space.seed(SEED)
        self.actions = ["LEFT", "DOWN", "RIGHT", "UP"]
        self.state_space = self.env.observation_space

class QLearning(object):
    def __init__(self, environment:FrozenLakeGame, learning_rate:float,
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

    def generate_lookup_table(self):
        fig, ax = plt.subplots() 
        ax.set_axis_off()
        q_lookup_table = ax.table(cellText=np.round(self.q_values, decimals= 4),
                                    rowLabels=[i for i in range(self.environment.state_space.n)],
                                    colLabels=self.environment.actions,
                                    rowColours =["yellow"] * self.environment.state_space.n,  
                                    colColours =["yellow"] * self.environment.action_space.n, 
                                    cellLoc ='center',  
                                    loc ='upper left')
        ax.set_title('Q-value approximation lookup table', fontweight ="bold")
    
    def epsilon_greedy_action(self, current_state):
            odd = np.random.choice(2, 1, p=[self.epsilon, 1 - self.epsilon])
            if odd == 1: return np.argmax(self.q_values[current_state, :])
            else: return self.environment.action_space.sample()
    
    def update_epsilon(self):
        """
            Updates epsilon value by multiplying it by decaying rate
        """
        self.epsilon *= self.decaying_rate
            
    def get_target(self, current_action):
        """_summary_

        Args:
            self (_type_): _description_
            bool (_type_): _description_

        Returns:
            _type_: _description_
        """        
        next_state, reward, terminated, truncated, _ = self.environment.env.step(current_action)
        done = terminated or truncated
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
    q_algo = QLearning(environment=frozen_lake, learning_rate= 0.1,
                       discount_factor=0.9, decaying_rate=0.999,
                       epsilon= 0.9) 
                       
    # q_algo.generate_lookup_table()
    plt.show()
    
    num_of_episodes = 5000
    rewards = []
    averaged_rewards = []
    rewards_of_hundred_episodes = []
    averaged_steps = []
    steps_of_hundred_episodes = []
    for episode in range(num_of_episodes):
        print(episode)
        if episode == 199 or episode == 4999:
            q_algo.generate_lookup_table()
        current_state = q_algo.environment.env.reset()[0]
        q_algo.update_epsilon()
        done = False
        steps = 0
        # plt.imshow(q_algo.environment.env.render())
        while True:
            current_action = q_algo.epsilon_greedy_action(current_state)
            steps += 1
            next_state, target, done = q_algo.get_target(current_action)
            # plt.imshow(q_algo.environment.env.render())
            err = q_algo.q_update(current_state, current_action, target)
            current_state = next_state
            if done:
                rewards.append(target)
                if target == 1: steps_of_hundred_episodes.append(steps)
                else: steps_of_hundred_episodes.append(100)
                break
        
        if (episode % 100 == 0) and (episode != 0):
            rewards_of_hundred_episodes = rewards[-100:]
            averaged_steps.append(np.mean(np.array(steps_of_hundred_episodes)))
            averaged_rewards.append(np.mean(np.array(rewards_of_hundred_episodes)))
            steps_of_hundred_episodes = []
    
    q_algo.environment.env.close()
    fig = plt.figure()
    ax = fig.subplots(1,3)
    ax[0].scatter(range(len(averaged_steps)),averaged_steps)
    ax[1].scatter(range(len(averaged_rewards)), averaged_rewards)
    ax[2].plot(rewards)
plt.show()


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