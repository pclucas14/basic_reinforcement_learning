import random
import numpy as np
np.set_printoptions(precision=4) 

# Sarsa with Eligibility Traces, Sarsa(\lambda)
# I'm moving away from the dictionary setup, as it's suboptimal for traces.

class Sarsa:
    def __init__(self, actions, world, epsilon=0.1, alpha=0.1, gamma=0.9, trace_coef=0.7, init_value=0, algo='e_trace'):
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
        
        # I keep two previous values. One for learning Q, and one for choosing actions
        # they will be equivalent when performing online training, however will differ when evaluating
        self.prev_value = None
        self.prev_value_for_choosing_a = None

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
            self.trace[state][action] += 1
        elif self.algo == 'h_trace':
            beta = self.trace_coef
            self.trace = (1 - beta) * self.gamma * self.trace
            self.trace[state][action] += beta

    def learnQ(self, state, action, reward, target):
        # SARSA(lambda) or SARSA(beta)
        if self.algo == 'e_trace':
            self.update_trace(state, action)
            delta = target - self.getQ(state, action)
            self.q += self.alpha * delta * self.trace

        elif self.algo == 'h_trace':  
            self.update_trace(state, action)
            q = self.getQ(state, action)
            if self.prev_value is None:
                q_tilde = q 
            else:
                beta = self.trace_coef
                q_tilde = (1 - beta) * self.prev_value + beta * q

            delta = target - q_tilde
            self.q += self.alpha * delta * self.trace

            self.prev_value = q_tilde

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
            if self.algo == 'h_trace':
                if self.prev_value_for_choosing_a is not None:
                    beta = self.trace_coef
                    q = [(1 - beta) * self.prev_value_for_choosing_a + beta * q_sa for q_sa in q]

            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]

            # log for next iteration
            self.prev_value_for_choosing_a = maxQ

        return action

    def learn(self, state1, action1, reward, state2, action2):
        qnext = self.getQ(state2, action2)
        self.learnQ(state1, action1, reward, reward + self.gamma * qnext)
