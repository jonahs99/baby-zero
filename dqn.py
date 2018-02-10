import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from itertools import count
import random
import math
from collections import namedtuple

from environment.connect4 import Connect4State
from model.connect4_model import Connect4Model

INPUT_SHAPE = (2, 6, 7)

BATCH_SIZE = 32
EPS_START = 0.1#0.9
EPS_END = 0.05
EPS_DECAY = 200
GAMMA = 0.99

model = Connect4Model()
optimizer = optim.RMSprop(model.net.parameters())

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
        a = actions[a.data]
        return a.view(-1)
    else:
        return random.choice(actions).view(-1)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    batch_state = torch.zeros(BATCH_SIZE, *INPUT_SHAPE)
    batch_action = torch.LongTensor(BATCH_SIZE, 1)
    batch_next_state = torch.zeros(BATCH_SIZE, *INPUT_SHAPE)
    batch_reward = torch.zeros(BATCH_SIZE, 1)

    for i in range(BATCH_SIZE):
        batch_state[i] = batch.state[i]
        batch_action[i] = batch.action[i]
        batch_next_state[i] = batch.next_state[i]
        batch_reward[i] = batch.reward[i]
    
    batch_state = Variable(batch_state)
    batch_action = Variable(batch_action)
    batch_next_state = Variable(batch_next_state, volatile=True)
    batch_reward = Variable(batch_reward)

    state_action_values = model.net(batch_state).gather(1, batch_action)
    next_state_values = model.net(batch_next_state).max(1)[0].view(-1, 1)

    expected_state_action_values = (next_state_values * GAMMA) + batch_reward
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def run_episode(disp=False):
    env = Connect4State()
    inputs = model.represent(env)

    for t in count():
        a = select_action(env)
        env.do_action(a[0])
        score = env.get_score()
        r = -1 if score == 0 else 0
        r = torch.Tensor([r])
        next_inputs = model.represent(env)
        memory.push(inputs, a, next_inputs, r)

        inputs = next_inputs

        if disp:
            print(env)
        if score != -1:
            break

memory = ReplayMemory(1000)
for i in range(10000):
    if i%100 == 99:
        print(i+1)
    run_episode()
    optimize_model()
run_episode(True)