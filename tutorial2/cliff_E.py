import cellular
import e_sarsa
import time
import sys
import numpy as np

startCell = None

def pp(arr):
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
        self.ai = e_sarsa.Sarsa(
            range(directions), world, epsilon=0.2, alpha=0.1, gamma=0.9, algo='h_trace', vc=vc)
        self.lastAction = None
        self.score = 0
        self.deads = 0

    def colour(self):
        return 'blue'

    def update(self):
        reward = self.calcReward()
        state = self.calcState()
        action = self.ai.chooseAction(state)
        if self.lastAction is not None:
            self.ai.learn(
                self.lastState, self.lastAction, reward, state, action)
        self.lastState = state
        self.lastAction = action

        here = self.cell
        if here.goal or here.cliff:
            self.cell = startCell
            self.lastAction = None
        else:
            self.goInDirection(action)

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


for vc in [0, 100, 10000]:
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


    pretraining = 1000000
    for i in range(pretraining):
        if i % 100000 == 0:
            print i, agent.score, agent.deads
            pp(e_sarsa.sigmoid(agent.ai.betas.squeeze().T))
            pp(agent.ai.q.mean(axis=-1).squeeze().T)
            agent.score = 0
            agent.deads = 0
        world.update()

    # print(agent.ai.state_history.T)
    np.save('beta_maps/%d' % vc, agent.ai.betas.squeeze().T)
