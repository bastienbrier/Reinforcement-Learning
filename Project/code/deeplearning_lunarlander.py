#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:44:19 2017

@author: andrei & bastien
"""

#Import packages
from __future__ import division
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import model_from_json

#Environment investigation
env = gym.make('LunarLander-v2')
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]


#Set hyperparameters
debug_mode = False
epsilon = 1
decay_rate = 0.975
taget_update_step = 500
sample_size = 32
memory_size = 50000
min_samples = 1000
gamma = 0.99
max_episodes = 2000
                
#Architecture of the models
                
#loss function
def hubert_loss(y_true, y_pred):
    err = y_pred - y_true
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )
                
# Q-Model
model = Sequential()
model.add(Dense(50, input_dim=state_dim, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))

model.add(Dense(4))
model.add(Activation('linear'))
model.compile(loss=hubert_loss, optimizer=Adam(lr=0.00025))
model.summary()

# Q-Model for target
target_model = Sequential()
target_model.add(Dense(50, input_dim=state_dim, activation='relu'))
target_model.add(Dense(40, activation='relu'))
target_model.add(Dense(40, activation='relu'))
target_model.add(Dense(4))
target_model.add(Activation('linear'))
target_model.compile(loss=hubert_loss, optimizer=Adam(lr=0.00025))

                
#Agent
class QLearningAgent():
    
    #initialization
    def __init__(self, gamma):
        """
        Initialize your internal state
        """
        self.gamma = gamma # importance of future rewards
        self.action = 0
        self.action_matrix = np.append(np.zeros((32,1)), np.ones((32,1)),axis = 1)
        self.action_matrix = np.append(self.action_matrix, np.ones((32,1))*2,axis = 1)
        self.action_matrix = np.append(self.action_matrix, np.ones((32,1))*3,axis = 1)

        self.memory_size = memory_size
        self.phi = []
        self.action_j  = []
        self.reward_j  = []
        self.phi_next = []
        self.is_done = []
        self.count = 0
        pass

    #define acting in a epsilon greedy way
    def act(self,observation, epsilon):
        """
        Choose action depending on your internal state
        """    
        if (np.random.random() < epsilon) or (self.count < min_samples): # epsilon-greedy
            self.action = np.random.randint(0, 4) # return a random action
            return self.action
        else:
            Q_approx = model.predict(observation) # Q approximation  
            self.action = np.argmax(Q_approx[0]) 
            return self.action 

    #storing each observation to be used later for network update and updating once in a while the target network
    def update(self, prev_state, next_state, action, reward, done):
        """
        Update your internal state
        """
        self.phi.append(prev_state)
        self.action_j.append(action)
        self.reward_j.append(reward)
        self.phi_next.append(next_state)
        self.is_done.append(np.double(done))
        # If the buffer is overflowing, remove the oldest added observation
        if len(self.phi) > self.memory_size:
            self.phi.pop(0)
            self.action_j.pop(0)
            self.reward_j.pop(0)
            self.phi_next.pop(0)
            self.is_done.pop(0)

        if self.count % taget_update_step == 0:
            print "target network update"
            target_model.set_weights(model.get_weights())
        self.count +=1

    #updating the network      
    def network_update(self):
        #Start only after a while
        if self.count == min_samples:
                print "Let the learning begin"
        if self.count > min_samples:
            #Sampling uniformly at random
            indexes = np.random.choice(len(self.phi), sample_size)
                            
            #Selecting the sampled tuples of state, action reward, next state, done
            phi = np.array([self.phi[i] for i in indexes])
            phi_next = np.array([self.phi_next[i] for i in indexes])
            reward_j = np.array([self.reward_j[i] for i in indexes])
            action_j = np.array([self.action_j[i] for i in indexes])
            is_done = np.array([self.is_done[i] for i in indexes])
            target = model.predict(phi)
            target_next  = target_model.predict(phi_next)
            
            #transforming the action array to make it usable in matrix operation
            action_chosen = np.stack((action_j,action_j,action_j,action_j), axis=1)
            full_action = np.double(action_chosen==self.action_matrix)
            
            #Computed the targets in a matrix multiplication fashion for faster computations
            y = np.zeros((sample_size, action_dim))
            max_next = np.max(target_next, axis=1)
            y= (1-full_action) * target + full_action * np.stack((reward_j,reward_j,reward_j,reward_j), axis=1)+ full_action * np.stack((max_next,max_next,max_next,max_next),axis=1) * gamma *(1-np.stack((is_done,is_done,is_done,is_done), axis=1))     
                
            #Update the model by feeding it the batch
            model.fit(phi, y, batch_size=sample_size, nb_epoch=1, verbose=False)
                
                
#function to set the epsilon
def get_epsilon(epsilon):
    return decay_rate * epsilon
                
#initialising everything
agent = QLearningAgent(gamma)
mean_t = np.zeros(100)
episode = 0
epsilon = get_epsilon(0)
                
#running the agent in the environment
while(np.mean(mean_t[-99:])<200) and (episode<max_episodes):
        episode+=1
        epsilon = get_epsilon(epsilon)
        reward_cumulated = ()
        all_actions = ()
        state = env.reset()
        old_state = np.array(state)
        t = 0
        done = False
        while not done:
            #env.render()   
            t+=1
            action = agent.act(np.reshape(old_state,[1,8]), epsilon)
            state, reward, done, info = env.step(action)

            agent.update(old_state, state, action, reward, done)   
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
                    print("episode finished after %d steps" % t)
                    print(np.mean(mean_t[-99:]))
                    print(episode)
                break
                
                #plotting the results
mvg_average = np.zeros(len(mean_t)-100)
for i in range(len(mean_t)-100):
    mvg_average[i] = np.mean(mean_t[i:i+100])
                
fig = plt.figure()
x = range(len(mean_t)-100)
plt.plot(x,mean_t[100:],'b')
plt.plot(x,mvg_average,'r')
file_name = "train_perf_final.png"
fig.savefig(file_name)