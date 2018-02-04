from torch.autograd import Variable

from itertools import count
import random
import math
from collections import namedtuple

from environment.connect4 import Connect4State
from model.connect4_model import Connect4Model

BATCH_SIZE = 32
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
model = Connect4Model()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

def select_action(env):
    actions_mask = env.gen_actions()
    actions = actions_mask.nonzero()

    sample = random.random()
    eps_threshold = EPS_START
    if sample > eps_threshold:
        inputs = Variable(model.represent(env), volatile=True)
        Q_max, a = model.net(inputs)[actions_mask].max(0)
        a = actions[a.data][0,0]
        return a
    else:
        return random.choice(actions)[0]

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

def run_episode():
    env = Connect4State()
    inputs = Variable(model.represent(env), volatile=True)

    for t in count():
        a = select_action(env)
        env.do_action(a)
        score = env.get_score()
        r = -1 if score == 0 else 0
        next_inputs = Variable(model.represent(env), volatile=True)
        memory.push(inputs, a, next_inputs, r)

        inputs = next_inputs
        if score != -1:
            break
    
    print(env)

memory = ReplayMemory(1000)
run_episode()