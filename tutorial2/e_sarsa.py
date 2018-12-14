import random
import numpy as np
np.set_printoptions(precision=2, suppress=True) 


# Sarsa with Eligibility Traces, Sarsa(\lambda)
# I'm moving away from the dictionary setup, as it's suboptimal for traces.

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

class Sarsa:
    def __init__(self, actions, world, epsilon=0.4, alpha=0.1, gamma=0.9, trace_coef=0.7, init_value=0, algo='e_trace', learn_beta=True, vc=1e-5):
        self.q = np.zeros((world.width, world.height, len(actions))) + init_value
        self.trace = np.zeros((world.width, world.height, len(actions)))

        self.state_history = np.zeros((world.width, world.height))

        if algo == 'h_trace' and learn_beta:
            self.betas = np.zeros((world.width, world.height)) #* 10000

        print('using eligibility traces : {}'.format('trace' in algo))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.trace_coef = trace_coef # lambda or beta, depending
        self.algo = algo
        self.init_value = init_value
        self.learn_beta = learn_beta       
        self.vc  = vc

        # I keep two previous values. One for learning Q, and one for choosing actions
        # they will be equivalent when performing online training, however will differ when evaluating
        self.prev_value = None
        self.prev_value_for_choosing_a = None

    def log_state(self, state):
        state = (state[0] - 1, state[1] - 1)
        self.state_history[state] += 1

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
            if self.learn_beta: 
                self.trace = (1 - np.expand_dims(sigmoid(self.betas), -1)) * self.trace  #* self.gamma
                self.trace[state][action] += sigmoid(self.betas[state])
            else:
                beta = self.trace_coef
                self.trace = (1 - beta) * self.trace  #* self.gamma
                self.trace[state][action] += beta

    def update_betas(self, state, q_hat, q_tilde, target):
        state = (state[0] - 1, state[1] - 1)
        #mse = (self.prev_value - target)**2 - (q_hat -target)**2
        #mse = max(0,mse)

        # reverting back
        #mse = (q_tilde - target) ** 2

        #target = mse / (mse + self.vc)
        #self.betas[state] = self.alpha * target + (1 - self.alpha) * self.betas[state]
        # beta = alpha*(MSE(V_tilde) / (MSE(V_tilde)  + VAR)) + (1-alpha)*beta

        # useful quantities
        if self.prev_value is None:
            return 

        state_beta = sigmoid(self.betas[state])
        sig_state_beta = sigmoid(state_beta)
        prev_value = self.prev_value

        CLIP = 0.05
        derivative = (sig_state_beta)*(1 - sig_state_beta)*( (q_tilde - target) * (q_hat - prev_value) + self.vc)

        derivative = max(-CLIP, min(CLIP, derivative))
       
        # self.betas[state] = self.alpha * -derivative + (1 - self.alpha) * self.betas[state]
        self.betas[state] -= self.alpha * derivative 
        
    
    def learnQ(self, state, action, reward, target):
        # SARSA(lambda) or SARSA(beta)
        if self.algo == 'e_trace':
            self.update_trace(state, action)
            delta = target - self.getQ(state, action)
            self.q += self.alpha * delta * self.trace

        elif self.algo == 'h_trace':  
            self.update_trace(state, action)
            q = self.getQ(state, action)
            beta = sigmoid(self.betas[state])

            assert 0. <= beta <= 1.

            if self.prev_value is None:
                q_tilde = q 
            else:
                q_tilde = (1 - beta) * self.prev_value + beta * q

            delta = (target - q_tilde)

            self.q += self.alpha * delta * self.trace


            # we update the betas at the end 
            if self.learn_beta: 
                self.update_betas(state, q, q_tilde, target)
            
            self.prev_value = q_tilde - reward

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
                    beta = sigmoid(self.betas[state])
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

        self.log_state(state1)
