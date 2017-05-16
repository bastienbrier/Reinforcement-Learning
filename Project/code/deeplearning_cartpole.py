#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:27:29 2017

@author: andrei & bastien
"""

#Import packages
from __future__ import division
import numpy as np
import gym
import time

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
import matplotlib.pyplot as plt


#Environment investigation
env = gym.make('CartPole-v0')

action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]


#Set hyperparameters
debug_mode = False
epsilon = 1
decay_rate = 0.99
min_epsilon = 0
taget_update_step = 200
sample_size = 32
memory_size = 50000
min_samples = 100
gamma = 0.99
max_episodes = 600



#Architecture of the models

# Q-Model
model = Sequential()
model.add(Dense(40, input_dim=state_dim, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(2))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer=Adam(lr=0.0005,  decay=1e-4))
model.summary()

# Q-Model for target
target_model = Sequential()
target_model.add(Dense(40, input_dim=state_dim, activation='relu'))
target_model.add(Dense(40, activation='relu'))
target_model.add(Dense(2))
target_model.add(Activation('linear'))
target_model.compile(loss='mse', optimizer=Adam(lr=0.0005,  decay=1e-4))



#Agent
class QLearningAgent():
    
    #initialization
    def __init__(self, gamma):
        """
        Initialize your internal state
        """
        self.gamma = gamma # importance of future rewards
        self.action = 0
        self.memory = np.zeros((memory_size,11)) 
        self.memory[:,4] = 2
        self.action_matrix = np.append(np.zeros((32,1)), np.ones((32,1)), axis = 1)
        self.count = 0
    
    #define acting in a epsilon greedy way
    def act(self,observation, epsilon):
        """
        Choose action depending on your internal state
        """    
        if (np.random.random() < epsilon) or (self.count < min_samples): # epsilon-greedy
            self.action = np.random.randint(0, 2) # return a random action
            return self.action
        else:
            Q_approx = model.predict(observation) # Q approximation  
            self.action = np.argmax(Q_approx) 
            return self.action 


    #storing each observation to be used later for network update and updating once in a while the target network
    def store(self, prev_state, next_state, action, reward, done):
        """
        Storing the states
        """
        if done==True:
            is_done=1
        else: 
            is_done=0
        to_store = np.append(np.append(np.ravel(prev_state), action),np.append(reward, np.ravel(next_state)))
        to_store = np.append(to_store, is_done)
        to_store = np.reshape(to_store, ((1,11)))
        self.memory = np.append(self.memory,to_store, axis = 0)
        self.memory = np.delete(self.memory,0, axis = 0)
        self.count +=1

        if (self.count % taget_update_step == 0) and (self.count > min_samples):
            print "target network update"
            target_model.set_weights(model.get_weights())
        
    def network_update(self):
        """
        Updating the network
        """
        #Start only after a while
        if self.count == min_samples:
                print "Let the learning begin"
        if self.count > min_samples:
            
            #Sampling uniformly at random
            indexes = np.random.choice(np.ravel(np.where(self.memory[:, 0]!=0)), sample_size)
            
            #Selecting the sampled tuples of state, action reward, next state, done
            phi = np.array([self.memory[i][0:4] for i in indexes])
            phi_next = np.array([self.memory[i][6:10] for i in indexes])
            reward_j = np.array([self.memory[i,5] for i in indexes])
            action_j = np.array([self.memory[i,4] for i in indexes])
            is_done = np.array([self.memory[i,10] for i in indexes])
            target = model.predict(phi)
            target_next  = target_model.predict(phi_next)
            
            #transforming the action array to make it usable in matrix operation
            action_chosen = np.stack((action_j,action_j), axis=1)
            full_action = np.double(action_chosen==self.action_matrix)
            max_next = np.max(target_next, axis=1)
            
            #Computed the targets in a matrix multiplication fashion for faster computations
            y = np.zeros((sample_size, action_dim))
            y= (1-full_action) * target + full_action * np.stack((reward_j,reward_j), axis=1)+ full_action * np.stack((max_next,max_next),axis=1) * gamma *(1-np.stack((is_done,is_done), axis=1))     

            #Update the model by feeding it the batch
            model.fit(phi, y, batch_size=sample_size, nb_epoch=1, verbose=False)
        

#function to set the epsilon
def get_epsilon(t):
    return max(min_epsilon, decay_rate ** t)

#initialising everything
agent = QLearningAgent(gamma)
mean_t = np.zeros(100)
epsilon = get_epsilon(0)
episode = 0
mean_t = np.zeros(100)


#running the agent in the environment
while(np.mean(mean_t[-99:])<196) and (episode < max_episodes):
        episode+=1
        epsilon = get_epsilon(episode)
        reward_cumulated = ()
        all_actions = ()
        state = env.reset()
        old_state = np.array(state)
        
        for t in range(200):
            #env.render()     
            action = agent.act(np.reshape(old_state,[1,4]), epsilon)
            state, reward, done, info = env.step(action)
            agent.store(old_state, state, action, reward, done)   
            agent.network_update()             
            old_state = np.array(state)  

            reward_cumulated = np.append(reward_cumulated, reward)

            if done:
                mean_t = np.append(mean_t,sum(reward_cumulated))
                            
                if debug_mode:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Reward: %f" % reward)
                    print("Explore rate: %f" % epsilon)
                    print("Episode finished with a reward of {}".format(sum(reward_cumulated)))
                else :
                    print("Episode finished with a reward of {}".format(sum(reward_cumulated)))
                    print(np.mean(mean_t[-99:]))
                    print(episode)
                break

#plotting the results
mvg_average = np.zeros(len(mean_t)-100)
for i in range(len(mean_t)-100):
    mvg_average[i] = np.mean(mean_t[i:i+100])
    

x = range(len(mean_t)-100)
plt.plot(x,mvg_average,'r')
plt.plot(x,mean_t[100:],'b')