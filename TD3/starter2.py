import sys

import pylab as plb
import numpy as np
import mountaincar

episode_number = 20 # number of episodes
episode_max = 1000 # maximum number of episodes
steps_trial = 500 # number of steps for the trial
p_number = 150 # number of points horizontal grid
k_number = 40 # number of points vertical grid

class RandomAgent():
    def __init__(self):
        """
        Initialize your internal state
        """
        pass

    def act(self):
        """
        Choose action depending on your internal state
        """
        return np.random.randint(-1, 2)

    def update(self, next_state, reward):
        """
        Update your internal state
        """
        pass

# implement your own agent here
class QLearningAgent():
    def __init__(self, l, p, k, epsilon, alpha, gamma):
        """
        Initialize your internal state
        """
        self.l = l # lambda
        self.p = p # horizontal grid
        self.k = k # vertical grid
        self.theta= np.ones(((self.p+1)*(self.k+1),3)) * 0.5 # initialize weights
        self.e = np.zeros(((self.p+1)*(self.k+1))) # eligibility traces
        self.distance = np.zeros(((self.p+1)*(self.k+1))) # distance initialization
        
        self.grid = np.zeros(((self.p+1)*(self.k+1),2)) # grid initialization
        count = 0
        for i in range(self.p+1):
            for j in range(self.k+1):
                self.grid[count][0] = -150 + i * 150 / self.p
                self.grid[count][1] = -20 + j * 40 / self.k
                count += 1
        count = None
        
        self.epsilon = epsilon # epsilon-greedy
        self.alpha = alpha # learning rate
        self.gamma = gamma # importance of future rewards
        self.delta = 0 # TD error
        self.action = 0 # initialize first action
        pass

    def act(self):
        """
        Choose action depending on your internal state
        """        
        if np.random.random() < self.epsilon: # epsilon-greedy
            self.action = np.random.randint(-1, 2) # return a random action
            return self.action
        else:
            Q_approx = np.dot(np.transpose(self.theta), self.distance) # Q approximation    
            self.action = np.argmax(Q_approx) - 1 # return the argmax of approximation minus 1 (values are -1, 0, or 1)
            return self.action

        return self.state_

    def update(self, next_state, reward):
        """
        Update your internal state
        """
        Q_approx = np.dot(np.transpose(self.theta), self.distance)
        self.delta = reward - Q_approx[self.action + 1] # calculate TD error
        self.e = self.e + self.distance # update the trace with the distance of the current state
        
        x = [next_state[0]] * ((self.p+1) * (self.k+1)) # get position
        vx = [next_state[1]] * ((self.p+1) * (self.k+1)) # get velocity
        self.distance = np.exp(-np.square(x-self.grid[:,0])) * np.exp(-np.square(vx-self.grid[:,1])) # new distance with new state
        Q_approx = np.dot(np.transpose(self.theta), self.distance) # new Q-Learning approximation
        self.delta = self.delta + self.gamma * max(Q_approx) # update delta
        
        # update the weights matrix, only for the action that was selected
        self.theta[:, self.action + 1] = self.theta[:, self.action + 1] + self.alpha * np.mean(self.delta) * self.e 
        self.e = self.gamma * self.l * self.e # update the trace with decay
        pass
        
# test class, you do not need to modify this class
class Tester:

    def __init__(self, agent):
        self.mountain_car = mountaincar.MountainCar()
        self.agent = agent

    def visualize_trial(self, n_steps=steps_trial):
        """
        Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """

        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()

        # make sure the mountain-car is reset
        self.mountain_car.reset()

        for n in range(n_steps):
            print('\rt =', self.mountain_car.t)
            print("Enter to continue...")
            raw_input()

            sys.stdout.flush()

            self.agent.state = [self.mountain_car.x, self.mountain_car.vx]
            reward = self.mountain_car.act(self.agent.act())

            # update the visualization
            mv.update_figure()
            plb.draw()

            # check for rewards
            if reward > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break

    def learn(self, n_episodes, max_episode):
        """
        params:
            n_episodes: number of episodes to perform
            max_episode: maximum number of steps on one episode, 0 if unbounded
        """

        rewards = np.zeros(n_episodes)
        for c_episodes in range(1, n_episodes):
            self.mountain_car.reset()
            step = 1
            while step <= max_episode or max_episode <= 0:
                reward = self.mountain_car.act(self.agent.act())
                self.agent.update([self.mountain_car.x, self.mountain_car.vx],
                                  reward)
                rewards[c_episodes] += reward
                if reward > 0.:
                    break
                step += 1
            formating = "end of episode after {0:3.0f} steps,\
                           cumulative reward obtained: {1:1.2f}"
            print(formating.format(step-1, rewards[c_episodes]))
            sys.stdout.flush()
        return rewards


if __name__ == "__main__":
    # modify RandomAgent by your own agent with the parameters you want
    #agent = RandomAgent()
    agent = QLearningAgent(0.9, p_number, k_number, 0.1, 0.5, 0.9)
    test = Tester(agent)
    # you can (and probably will) change these values, to make your system
    # learn longer
    test.learn(episode_number, episode_max) # default = 10, 100

    print("End of learning, press Enter to visualize...")
    raw_input()
    test.visualize_trial()
    plb.show()
    