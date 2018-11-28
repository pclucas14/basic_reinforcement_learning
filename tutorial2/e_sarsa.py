import random
import numpy as np
np.set_printoptions(precision=4) 

# Sarsa with Eligibility Traces, Sarsa(\lambda)
# I'm moving away from the dictionary setup, as it's suboptimal for traces.

class Sarsa:
    def __init__(self, actions, world, epsilon=0.1, alpha=0.2, gamma=0.9, trace_coef=0.1, init_value=-1, algo='e_trace'):
        self.q = np.zeros((world.width, world.height, len(actions))) + init_value
        self.trace = np.zeros((world.width, world.height, len(actions)))
        
        print('using eligibility traces : {}'.format('trace' in algo))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.trace_coef = trace_coef # lambda or beta, depending
        self.algo = algo
        self.init_value = init_value

    def getQ(self, state, action):
        state = (state[0] - 1, state[1] - 1)
        out = self.q[state][action] 
        return out

    def setQ(self, state, action, value):
        state = (state[0] - 1, state[1] - 1)
        self.q[state][action] = value

    def getE(self, state, action):
        state = (state[0] - 1, state[1] - 1)
        return self.trace[state][action]

    def update_trace(self, state, action):
        state = (state[0] - 1, state[1] - 1)
        
        if self.algo == 'e_trace':
            lambda_ = self.trace_coef
            self.trace = lambda_ * self.gamma * self.trace
            self.trace[state, action] += 1
        elif self.algo == 'h_trace':
            beta = self.trace_coef
            self.trace = (1 - beta) * self.gamma * self.trace
            self.trace[state, action] += beta

    def learnQ(self, state, action, reward, target):
        # SARSA(lambda) or SARSA(beta)
        if 'trace' in self.algo:  
            self.update_trace(state, action)
            oldv = self.getQ(state, action)
            delta = target - oldv
            self.setQ(state, action, oldv + self.alpha * delta * self.getE(state, action))
        
        # regular setup
        else: 
            oldv = self.getQ(state, action)
            if oldv == self.init_value: # original value
                self.setQ(state, action, reward)
            else:
                self.setQ(state, action, oldv + self.alpha * (target - oldv))

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action

    def learn(self, state1, action1, reward, state2, action2):
        qnext = self.getQ(state2, action2)
        self.learnQ(state1, action1, reward, reward + self.gamma * qnext)
