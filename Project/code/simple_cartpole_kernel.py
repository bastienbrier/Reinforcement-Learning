# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#importing the packages
from __future__ import division
import numpy as np
import gym
import matplotlib.pyplot as plt

#Environment investigation
env = gym.make('CartPole-v0')
nb_actions =env.action_space.n
nb_dimensions = env.observation_space.shape[0]
bounds = list(zip(env.observation_space.low, env.observation_space.high))
bounds[1] = [-0.5, 0.5]
bounds[3] = [-np.radians(50), np.radians(50)]

#Setting the hyperparameters
debug_mode = False
l = 0.1
o = 25
p = 12
gamma=0.995
sigma_1 = 0.05
sigma_2 = 0.5
max_episodes = 1000
max_steps = 200
desired_mean = 195
gamma = 0.99  
min_epsilon = 0.001
min_learning_rate = 0.001

#Initialise the grid and weights
theta= np.zeros(((p+1)*(o+1),2)) # initialize weights
grid = np.zeros(((p+1)*(o+1),2)) # grid initialization           
count = 0
for k in range(o+1):
            for h in range(p+1):
                grid[count][0] = bounds[2][0] + k * (bounds[2][1] - bounds[2][0] ) / o
                grid[count][1] = bounds[3][0] + h * (bounds[3][1] - bounds[3][0] ) / p
                count += 1

#Epsilon greedy strategy
def take_action(old_state, epsilon):      
    if np.random.random() < epsilon: 
            action = np.random.randint(0, 2) 
            return action
    else:
            distance = np.exp(-np.square((old_state[2]-grid[:,0]))/sigma_1)* np.exp(-np.square((old_state[3]-grid[:,1]))/sigma_2) # new distance with new state
            Q_approx = np.dot(np.transpose(theta), distance)    
            action = np.argmax(Q_approx) 
            return action

#update the network
def update(old_state, state, action, reward, done, learning_rate)  :
    
    distance = np.exp(-np.square((old_state[2]-grid[:,0]))/sigma_1)* np.exp(-np.square((old_state[3]-grid[:,1])/sigma_2)) #  distance with old state
    distance_next = np.exp(-np.square((state[2]-grid[:,0]))/sigma_1)* np.exp(-np.square((state[3]-grid[:,1])/sigma_2)) #  distance with new state

    Q_approx = np.dot(np.transpose(theta), distance) #Q approx of old state
    Q_approx_next = np.dot(np.transpose(theta), distance_next) #Q approx of old state
    delta = reward - Q_approx[action] + gamma * max(Q_approx_next) # calculate TD error      

    # update the weights matrix, only for the action that was selected
    theta[:, action] +=  learning_rate * delta * distance 
    #to check than the rewards are not exploding
    if Q_approx[0] == np.nan:
        print "ALERT!!"

#update epsilon
def get_epsilon(t):
    return max(min_epsilon, min(1, 1.0 - np.log10((t+1)/25)))
#update learning rate
def get_learning_rate(t):    
    return max(min_learning_rate, min(0.01, 1/(t+100)))

#initialise everything
learning_rate = get_learning_rate(0)
epsilon = get_epsilon(0)
mean_t = np.zeros(100)
episode = 0

#run the agent in the environment
while(np.mean(mean_t[-99:])<195):
        episode+=1
        learning_rate = get_learning_rate(episode)
        epsilon = get_epsilon(episode)
        reward_cumulated = ()
        all_actions = ()
        state = env.reset()
        old_state = state  
        
        for t in range(200):
            #env.render()     
            action = take_action(old_state, epsilon)
            state, reward, done, info = env.step(action)

            update(old_state, state, action, reward, done, learning_rate)                
            old_state = state  

            all_actions = np.append(all_actions, action)
            reward_cumulated = np.append(reward_cumulated, reward)

            if done:
                mean_t = np.append(mean_t,sum(reward_cumulated))
                            
                if debug_mode:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Action: %d" % np.mean(all_actions))
                    print("Reward: %f" % reward)
                    print("Explore rate: %f" % epsilon)
                    print("Learning rate: %f" % learning_rate)
                    print("Episode finished with a reward of {}".format(sum(reward_cumulated)))
                else :
                    print("Episode finished with a reward of {}".format(sum(reward_cumulated)))
                    print(np.mean(mean_t[-99:]))
                    print(episode)
                    print(learning_rate)
                break
 
#Plot the result           
x = range(len(mean_t)-100)
mvg_average = np.zeros(len(mean_t)-100)
for i in range(len(mean_t)-100):
    mvg_average[i] = np.mean(mean_t[i:i+100])
fig = plt.figure()
plt.plot(x,mvg_average,'r')
plt.plot(x,mean_t[100:],'b')
fig.savefig('train_perf_cartpolekernel.png')

