import torch
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from search import MCTS
import examples.connect4 as connect4

def self_play(index, model, eps, disp=False):
    mcts = MCTS(connect4.State(), model)
    mcts.eps = eps
    while not mcts.state.score()[1]:
        for _ in range(100):
            mcts.iterate()
        mcts.do_action(mcts.best_action())
        if disp:
            print(mcts.state)
    print('game %d done.' % index)
    return collect_data(mcts)

def evaluate(index, model):
    pass

def collect_data(mcts):
    gamma = 0.99

    visit_thresh = 6
    nodes = [node for node in mcts.root.flat() if node.n > 6]

    input_tensor = torch.FloatTensor(len(nodes) * mcts.state.SYMMETRIES, *mcts.state.INPUT_SHAPE)
    target_tensor = torch.FloatTensor(len(nodes) * mcts.state.SYMMETRIES, 1)
    
    for i, node in enumerate(nodes):
        if node.children:
            if node.turn == 0:
                Q = gamma * max(-child.w / child.n for child in node.children if child.n)
            else:
                Q = gamma * -max(child.w / child.n for child in node.children if child.n)
        else:
            Q = node.w / node.n

        input_tensor[i*4:i*4 + 4] = node.inputs[0]
        target_tensor[i*4:i*4 + 4] = Q * node.inputs[1]

    return data_utils.TensorDataset(input_tensor, target_tensor)

def train(model, optimizer, datasets):
    data = data_utils.ConcatDataset(datasets)
    dataloader = data_utils.DataLoader(data, batch_size=16, shuffle=True)
    print(len(dataloader), 'batches to train...')

    n_epochs = 1
    batch_count = 0
    running_loss = 0

    for epoch in range(n_epochs):
        for inputs, targets in dataloader:
            inputs, targets = Variable(inputs), Variable(targets)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = F.smooth_l1_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            batch_count += 1

    return running_loss / batch_count

import matplotlib.pyplot as plt
from multiprocessing import Pool
import math

def plot_window():
    plt.figure()
    plt.ion()
    plt.xlabel("batch #")
    plt.ylabel("average huber loss of batch")

def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.autoscale(enable=True, axis='y')
    plt.ylim(ymin = 0)
    plt.draw()
    plt.pause(0.001)

def worker(args):
    model, eps = args
    data = play_game(model, eps)
    print('game done.')
    return data

def start():
    model = connect4.Model()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    loss_history = []
    n_game_batch = 3
    n_batches = 2

    eps_start = 1.0
    eps_end = 1.0#0.05
    eps_decay = -math.log(eps_end / eps_start)

    for i in range(n_batches):
        print('batch:', i+1, '/', n_batches)

        eps = eps_start * math.exp(-eps_decay * i / (n_batches - 1))
        print('eps=', eps)

        pool = Pool(processes=2)
        datasets = pool.starmap(self_play, [(i, model, eps) for i in range(n_game_batch)])

        loss = train(model, optimizer, datasets)
        torch.save(model.state_dict(), './examples/data/model.pt')

        loss_history.append(loss)
        torch.save(loss_history, './examples/data/loss.pt')
        plot_loss(loss_history)
        print('LOSS:', loss)

start()
plt.show()