import random
import numpy as np
np.set_printoptions(precision=2, suppress=True) 

# Sarsa with Eligibility Traces, Sarsa(\lambda)
# I'm moving away from the dictionary setup, as it's suboptimal for traces.

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

class V_est:
    def __init__(self, actions, world, epsilon=0.4, alpha=0.1, gamma=0.9, init_value=2.5, vc=0):
        self.v             = np.zeros((world.width, world.height)) + init_value
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

    def update_betas(self, state, v_hat, v_tilde, target):
        state = (state[0] - 1, state[1] - 1)
        
        if self.prev_value is None:
            return 

        sig_betas = sigmoid(self.betas)
        state_beta = sig_betas[state]
        prev_value = self.prev_value

        CLIP = 3 # 0.05
        derivative = (state_beta)*(1 - state_beta)*( (v_tilde - target) * (v_hat - prev_value) + self.vc)
        # FLIP
        # derivative = (state_beta)*(1 - state_beta)*( (v_tilde - target) * -1 * (v_hat - prev_value) + self.vc)
       
        '''
        if derivative < 0:	
            self.betas[state] +=0.005
            if self.betas[state] > CLIP:
                self.betas[state] = CLIP
        else:
            self.betas[state] -= 0.005
            if self.betas[state] < -CLIP:
                self.betas[state] = -CLIP
	'''

        # self.betas[state] = 0.0005 * -derivative + (1 - self.alpha) * self.betas[state]
        self.betas[state] -= 0.1 * derivative 
        
    
    def learnV(self, state, reward, target):
        v = self.getV(state)
        beta = sigmoid(self.betas[state]) #state[0] - 1, state[1] - 1])

        assert 0. <= beta <= 1.

        if self.prev_value is None:
            v_tilde = v
        else:
            v_tilde = (1 - beta) * self.prev_value + beta * v

        CLIP  = 500
        delta = (target - v_tilde)
        delta = max(-CLIP, min(CLIP, delta))

        # update values
        new_v = v + self.alpha * delta
        self.setV(state, new_v)

        # we update the betas at the end 
        # print('v, v_tilde, target')
        # print(v, v_tilde, target)
        self.update_betas(state, v, v_tilde, target)
        self.prev_value = v_tilde - reward
        

    def chooseAction(self, state):
        if np.random.rand() > self.epsilon:
            # Hardcoding optimal policy
            if state == (1, 1):
                return 2
            elif state[0] != 12: 
                return 1
            else:
                return 0
            pass
        else:
            action = random.choice(self.actions)
        
        return action


    def learn(self, state1, action1, reward, state2, action2):
        vnext = self.getV(state2)
        self.learnV(state1, reward, reward + self.gamma * vnext)
        self.log_state(state1)
