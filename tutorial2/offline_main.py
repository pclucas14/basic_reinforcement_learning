import cellular
import offline_v_est
import time
import sys
import numpy as np

startCell = None

def ppp(arr):
    betas = np.array_repr(arr).split('\n') 
    out = []
    for i in range(len(betas)):
        if i % 2 == 1 : 
            curr += ' ' + betas[i].strip()
        else:
            if i > 0 : out += [curr]
            curr = betas[i].strip()
    
    for piece in out:
        print(piece.replace('array', '').replace('(', '').replace(')', '').replace('[[', '['))


class Cell(cellular.Cell):
    def __init__(self):
        self.cliff = False
        self.goal = False
        self.wall = False

    def colour(self):
        if self.cliff:
            return 'red'
        if self.goal:
            return 'green'
        if self.wall:
            return 'black'
        else:
            return 'white'

    def load(self, data):
        global startCell
        if data == 'S':
            startCell = self
        if data == '.':
            self.wall = True
        if data == 'X':
            self.cliff = True
        if data == 'G':
            self.goal = True


class Agent(cellular.Agent):
    def __init__(self, world, vc):
        self.ai = offline_v_est.V_est(
            range(directions), world, epsilon=0.01, alpha=0.1, gamma=0.5, vc=vc)
        self.lastAction = None
        self.score = 0
        self.deads = 0
        self.episode_buffer = []
        self.n     = 1 # n-step return

    def colour(self):
        return 'blue'

    def update(self):
        reward = self.calcReward()
        state = self.calcState()
        action = self.ai.chooseAction(state)

        #if self.lastAction is not None:
        #    self.ai.learn(
        #        self.lastState, self.lastAction, reward, state, action)
        
        self.lastState = state
        self.lastAction = action
        here = self.cell

        # add to buffer
        self.episode_buffer += [(state, action, reward)]

        if here.goal or here.cliff:
            # train on episode
            self.train()
            self.episode_buffer = []
            self.cell = startCell
            self.lastAction = None
        else:
            self.goInDirection(action)

    def train(self):
        states, actions, rewards = [], [], []
        
        for (s,a,r) in self.episode_buffer:
            states += [s]; actions += [a]; rewards += [r]
            agent.ai.log_state(s)

        estimates = [self.ai.getV(s) for s in states]
        returns = []
        
        ep_T = len(rewards)

        # accumulate rewards
        for i in reversed(range(ep_T)):
            if len(returns) == 0:
                returns += [rewards[i]]
            else:
                return_tm1 = returns[0]
                returns = [self.ai.gamma * return_tm1 + rewards[i]] + returns
        
        # update every state in the trajectory in chronological order
        for i in range(ep_T-1):
            self.ai.learnV(states[i], rewards[i], returns[i])
    

    def calcState(self):
        return self.cell.x, self.cell.y

    def calcReward(self):
        here = self.cell
        if here.cliff:
            self.deads += 1
            return cliffReward
        elif here.goal:
            self.score += 1
            return goalReward
        else:
            return normalReward


for vc in [0]:
    normalReward = -1
    cliffReward = -100
    goalReward = 50

    directions = 4
    world = cellular.World(Cell, directions=directions, filename='../worlds/cliff.txt')

    if startCell is None:
        print "You must indicate where the agent starts by putting a 'S' in the map file"
        sys.exit()
    agent = Agent(world, vc)
    world.addAgent(agent, cell=startCell)

    pretraining = 500001
    for i in range(pretraining):
        if i % 10000 == 0 and i > 0:
            print i, agent.score, agent.deads
            ppp(offline_v_est.sigmoid(agent.ai.betas.squeeze().T))
            print(agent.ai.state_history.T)
            agent.score = 0
            agent.deads = 0
        world.update()
        #if i > 100 : break

    print('VARIANCE COST {}'.format(vc))
    ppp(offline_v_est.sigmoid(agent.ai.betas.squeeze().T))
    print(agent.ai.v.T)

    # print(agent.ai.state_history.T)
    np.save('offline_beta_maps/%d' % vc, offline_v_est.sigmoid(agent.ai.betas.squeeze().T))
