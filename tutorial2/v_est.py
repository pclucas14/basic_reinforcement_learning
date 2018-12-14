import random
import numpy as np
np.set_printoptions(precision=2, suppress=True) 

# Sarsa with Eligibility Traces, Sarsa(\lambda)
# I'm moving away from the dictionary setup, as it's suboptimal for traces.

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

class V_est:
    def __init__(self, actions, world, epsilon=0.4, alpha=0.1, gamma=0.9, init_value=0, vc=0):
        self.v             = np.zeros((world.width, world.height)) + init_value
        self.trace         = np.zeros((world.width, world.height))
        self.state_history = np.zeros((world.width, world.height))
        self.betas         = np.zeros((world.width, world.height)) 

        self.epsilon = epsilon
        self.alpha   = alpha
        self.gamma   = gamma
        self.actions = actions
        self.vc      = vc

        # I keep two previous values. One for learning Q, and one for choosing actions
        # they will be equivalent when performing online training, however will differ when evaluating
        self.prev_value = None
        self.prev_value_for_choosing_a = None

    def log_state(self, state):
        state = (state[0] - 1, state[1] - 1)
        self.state_history[state] += 1

    def getV(self, state):
        state = (state[0] - 1, state[1] - 1)
        out = self.v[state]
        return out

    def setV(self, state, value):
        state = (state[0] - 1, state[1] - 1)
        self.v[state] = value

    def getE(self, state, action):
        state = (state[0] - 1, state[1] - 1)
        return self.trace[state][action]

    def update_trace(self, state):
        state = (state[0] - 1, state[1] - 1)
        sig_betas = sigmoid(self.betas)
        self.trace = (1 - sig_betas) * self.trace  #* self.gamma
        self.trace[state] += sig_betas[state]

    def update_betas(self, state, v_hat, v_tilde, target):
        state = (state[0] - 1, state[1] - 1)
        
        if self.prev_value is None:
            return 

        sig_betas = sigmoid(self.betas)
        state_beta = sig_betas[state]
        prev_value = self.prev_value

        CLIP = 500 # 0.05
        derivative = (state_beta)*(1 - state_beta)*( (v_tilde - target) * (v_hat - prev_value) + self.vc)
        derivative = max(-CLIP, min(CLIP, derivative))
       
        # self.betas[state] = 0.0005 * -derivative + (1 - self.alpha) * self.betas[state]
        self.betas[state] -= 1e-4 * derivative 
        
    
    def learnV(self, state, reward, target):
        self.update_trace(state)
        v = self.getV(state)
        beta = sigmoid(self.betas[state])

        assert 0. <= beta <= 1.

        if self.prev_value is None:
            v_tilde = v
        else:
            v_tilde = (1 - beta) * self.prev_value + beta * v

        CLIP = 500
        delta = (target - v_tilde)
        delta = max(-CLIP, min(CLIP, delta))

        self.v += self.alpha * delta * self.trace

        # we update the betas at the end 
        self.update_betas(state, v, v_tilde, target)
        self.prev_value = v_tilde - reward
        

    def chooseAction(self, state):
        return random.choice(self.actions)


    def learn(self, state1, action1, reward, state2, action2):
        vnext = self.getV(state2)
        self.learnV(state1, reward, reward + self.gamma * vnext)
        self.log_state(state1)
