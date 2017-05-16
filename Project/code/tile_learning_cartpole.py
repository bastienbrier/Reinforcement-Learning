#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 09:17:09 2017

@author: andrei & bastien
"""

#importing the packages
from __future__ import division
import gym
import numpy as np
import matplotlib.pyplot as plt

#Environment investigation

env = gym.make('CartPole-v0')
nb_actions =env.action_space.n
nb_dimensions = env.observation_space.shape[0]
bounds = list(zip(env.observation_space.low, env.observation_space.high))
bounds[1] = [-0.5, 0.5]
bounds[3] = [-np.radians(50), np.radians(50)]

#Hyperparameters
debug_mode = False
bucket_sizes = [1,1,10,5]
max_episodes = 1000
max_steps = 200
desired_mean = 195
gamma = 0.99  
min_epsilon = 0.01
min_learning_rate = 0.1


#initialise table of discretised states
tile_table = dict()

#function to increment the table if new discretization of state is encountered
def increment_table(old_tile):
    if str(old_tile) not in tile_table.keys():
        tile_table[str(old_tile)] = np.zeros(2)
    
#function to discretize the state
def state_to_tile(state):
    tile = ()
    for i in range(nb_dimensions):
        span_bound = bounds[i][1]-bounds[i][0]
        if state[i] <= bounds[i][0]:
            j = 0
        elif state[i] >= bounds[i][1]:
            j = bucket_sizes[i] - 1
        else :
            for j in range(bucket_sizes[i]):
                if state[i] < (bounds[i][0] + j * span_bound / bucket_sizes[i]):
                    break
                
        tile = np.append(tile,j)
    return tile

#update the table with improved reward estimates
def update_table(old_tile, tile, action, reward, done, learning_rate):
    tile_table[str(old_tile)][np.int(action)] += learning_rate* (reward + gamma * np.max(tile_table[str(tile)]) -tile_table[str(old_tile)][np.int(action)])

#act according to an epsilon greedy policy
def take_action(tile, epsilon):
    if np.random.uniform() < epsilon:
        action = np.random.choice(range(nb_actions))
    else :
        action = np.argmax(tile_table[str(tile)])
    return action
    
#function to decay epsilon
def get_epsilon(t):
    return max(min_epsilon, min(1, 1.0 - np.log10((t+1)/25)))

#function to decay learning rate
def get_learning_rate(t):    
    return max(min_learning_rate, min(0.5, 1.0 - np.log10((t+1)/25)))


 
#initialise everything
learning_rate = get_learning_rate(0)
epsilon = get_epsilon(0)
mean_t = np.zeros(100)
episode = 0

#run the agent in the environment
while(np.mean(mean_t[-99:])<195):
        learning_rate = get_learning_rate(episode)
        epsilon = get_epsilon(episode)
        reward_cumulated = ()
        all_actions = ()
        state = env.reset()
        old_tile = state_to_tile(state)
        
        for t in range(200):
            increment_table(old_tile)
            #env.render()     
            action = take_action(old_tile, epsilon)
            state, reward, done, info = env.step(action)
    
            tile = state_to_tile(state)
            increment_table(tile)

            update_table(old_tile, tile, action, reward, done, learning_rate)                
            old_tile = tile
            
            all_actions = np.append(all_actions, action)
            reward_cumulated = np.append(reward_cumulated, reward)

            if done:
                episode+=1
                mean_t = np.append(mean_t,sum(reward_cumulated))
                            
                if debug_mode:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Action: %d" % np.mean(all_actions))
                    print tile
                    print("Reward: %f" % reward)
                    print tile_table[str(tile)]
                    print("Explore rate: %f" % epsilon)
                    print("Learning rate: %f" % learning_rate)
                    print("Episode finished with a reward of {}".format(sum(reward_cumulated)))
                else :
                    print("Episode finished with a reward of {}".format(sum(reward_cumulated)))
                    print(np.mean(mean_t[-99:]))
                    print(episode)
                break


#Plot the result           
x = range(len(mean_t)-100)
mvg_average = np.zeros(len(mean_t)-100)
for i in range(len(mean_t)-100):
    mvg_average[i] = np.mean(mean_t[i:i+100])
fig = plt.figure()
plt.plot(x,mvg_average,'r')
plt.plot(x,mean_t[100:],'b')
fig.savefig('train_perf_cartpoletile.png')
