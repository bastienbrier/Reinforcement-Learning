#### Bandit exercise ####

# README if libraries need to be installed 
# You need to install numpy and matplotlib using pip (or pip3).
# This can be done with the following commands (in a terminal in a Linux OS):
#   If running Python 2:
#    pip install numpy
#    pip install matplotlib
#   If running Python 3:
#    pip3 install numpy
#    pip3 install cairocffi
#    pip3 install matplotlib
# For this, 'pip' (or pip3) needs to be installed (which is usually already the case). If not, you can do it by installing classically python-dev (for pip) or python3-pip (for pip3), with your usual OS library management tool (yum, aptitude, apt-get, synaptic, ...). If using Python 3, you might need to install libffi-dev as well.
# This version was tested under Python 2.7.6 and under Python 3.4.3.

import numpy as np
import matplotlib.pyplot as plt
import sys


class Banditos:
    def __init__(self, N, k):
        self.cur = 0
        self.q_stars = np.random.randn(N, k)

    def select(self, n):
        self.cur = n

    def act(self, a):
        mean = self.q_stars[self.cur, a]
        reward = mean + np.random.randn()
        return reward

# Random Agent Class
class randomAgent:
    def __init__(self, A):
        self.A = A

    def interact(self):
        return np.random.randint(0, self.A)

    def update(self, a, r):
        pass
    
# Epsilon-Greedy Agent Class
class epsilonGreedyAgent:
    def __init__(self, A, epsilon):
        self.A = A
        self.meanQ = np.zeros(A) # vector of mean performance of each arm
        self.K = np.zeros(A) # vector of number of times each arm was chosen
        self.epsilon = epsilon

    def interact(self):
        if np.random.random() < self.epsilon: # if inferior to epsilon, take a random arm
            return np.random.randint(0, self.A)
        else: # else take the best one
            return np.argmax(self.meanQ)

    def update(self, a, r):
        self.meanQ[a] = (self.meanQ[a] * self.K[a] + r) / (self.K[a] + 1) # update mean with new value
        self.K[a] += 1
   
# Optimistic Epsilon-Greedy Agent Class     
class optimisticEpsGreedyAgent:
    def __init__(self, A, epsilon):
        self.A = A
        self.meanQ = np.ones(A) * 3 # higher mean for optimistic (final mean = 1.5)
        self.K = np.zeros(A)
        self.epsilon = epsilon

    def interact(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.A)
        else:
            return np.argmax(self.meanQ) 

    def update(self, a, r):
        self.meanQ[a] = (self.meanQ[a] * self.K[a] + r) / (self.K[a] + 1)
        self.K[a] += 1
       
# Softmax Agent Class   
class softMaxAgent:
    def __init__(self, A, alpha):
        self.A = A
        self.alpha = alpha
        self.meanQ = np.zeros(A)
        self.K = np.zeros(A)
        self.Prob = np.zeros(A) # vector of probability
        self.Z = 0
        self.cumProb = 0 # cumulative probability

    def interact(self):
        x = np.random.random()
        self.Z = sum(np.exp(self.alpha*self.meanQ)) # denominator
        for i in range(self.A): # compute each probability
            self.Prob[i] = np.exp(self.alpha*self.meanQ[i]) / self.Z # softmax formula
            self.cumProb += self.Prob[i]
            if x < self.cumProb: # select the arm we're at when x becomes inferior to cumProb
                return i # this way, each arm j has Prob[j] of being chosen

    def update(self, a, r):
        self.meanQ[a] = (self.meanQ[a] * self.K[a] + r) / (self.K[a] + 1)
        self.K[a] += 1
        self.cumProb = 0

# UCB Agent Class    
class UCBAgent:
    def __init__(self, A):
        self.A = A
        self.meanQ = np.zeros(A)
        self.K = np.zeros(A)
        self.t = 0 # time variable
    
    def interact(self):
        for i in range(self.A):
            if self.K[i] == 0: # first, test each arm one time: handles the case K=0 (for division issue)
                return i
        else:
            return np.argmax(self.meanQ + np.sqrt(2*np.log(self.t))/self.K) # UCB formula
    
    def update(self, a, r):
        self.meanQ[a] = (self.meanQ[a] * self.K[a] + r) / (self.K[a] + 1)
        self.K[a] += 1
        self.t += 1 # update the time


"""
Create your own agent classes implementing
first the epsilon greedy agent, and then the
Softmax agent, optimisticEpsGreedyAgent
and UCBAgent.
To make your classes compatible with the
tester, they must exhibit a constructor of
the form
def __init__(self, A, ...):

with ... being other parameters of your choice,
a function
def interact(self):

that returns an action given the current state
of the bandit, and a function
def update(self, a, r):

that takes the action that was performed, the
reward that was obtained, and updates the state
of the bandit. The epsGreedyAgent is here to
help you get an idea on how to implement these
methods.

Once your implementation of an agent is complete,
you can test it by replacing randomAgent in
the AgentTester parameters in the main script below
by your own class, and give a table containing the
parameters you want to use as a dictionnary (e.g.
{'epsilon': 0.1}) as an argument.

The AgentTester will automatically test the performance
of your agent, will give you both the epochwise mean
reward and percentage of optimal action, an will
plot your results.

You may want to start by testing the epsilon greedy
policy with various values of epsilon, to get a grasp
of the results you are supposed to obtain.
"""

# Do not modify this class.
class AgentTester:
    def __init__(self, agentClass, N, k, iterations, params):
        self.iterations = iterations
        self.N = N
        self.agentClass = agentClass
        self.agentTable = []
        params['A'] = k
        for i in range(N):
            self.agentTable[len(self.agentTable):] = [agentClass(**params)]
        self.bandits = Banditos(self.N, k)
        self.optimal = np.argmax(self.bandits.q_stars, axis=1)

    def oneStep(self):
        rewards = np.zeros(self.N)
        optimals = np.zeros(self.N)
        for i in range(self.N):
            self.bandits.select(i)
            action = self.agentTable[i].interact()
            optimals[i] = (action == self.optimal[i]) and 1 or 0
            rewards[i] = self.bandits.act(action)
            self.agentTable[i].update(action, rewards[i])
        return rewards.mean(), optimals.mean() * 100

    def test(self):
        meanrewards = np.zeros(self.iterations)
        meanoptimals = np.zeros(self.iterations)
        for i in range(self.iterations):
            meanrewards[i], meanoptimals[i] = self.oneStep()
            display = '\repoch: {0} -- mean reward: {1} -- percent optimal: {2}'
            sys.stdout.write(display.format(i, meanrewards[i], meanoptimals[i]))
            sys.stdout.flush()
        return meanrewards, meanoptimals

# Modify only the agent class and the parameter dictionnary.

if __name__ == '__main__':
    tester = AgentTester(UCBAgent, 2000, 10, 1000, {})

    # Do not modify.
    meanrewards, meanoptimals = tester.test()
    plt.figure(1)
    plt.plot(meanrewards)
    plt.xlabel('Epoch')
    plt.ylabel('Average reward')
    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Percent optimal')
    plt.plot(meanoptimals)
    plt.show()
