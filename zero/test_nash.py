import numpy as np
import visualize
import matplotlib.pyplot as plt

import time

from example.nash.nash_game import NashGame
from zero import MCTS
from baseline import random_player
from evaluate import Arena
from nn import NN

from trainer import Trainer

n = 6
n2 = n**2

epochs = 200
games = 40
sims = 200
gens = 2

game = NashGame(n)
net = NN(game, 'example/nash/config-nash')

baseline_player = random_player(game)

scores = []

ax = plt.figure()
plt.ion()

plt.xlabel('Epoch')
plt.ylabel('Score vs Random Baseline (20 games)')

def callback(epoch):
    ellapsed = time.time() - time_start
    print('Epoch {}/{}'.format(epoch, epochs))
    print('{:.0f} sec so far. ~{:.0f} to go.'.format(ellapsed, ellapsed * epochs/(epoch+1)))

    bmcts = MCTS(game, None)

    mcts = MCTS(game, net.predictor())
    arena = Arena(game, lambda state: np.argmax(mcts.simulate(state, n=sims)),
                        lambda state: np.argmax(bmcts.simulate(state, n=sims, first=True)))
    score, n_games = arena.play_games(half_n=10, verbose=1)

    scores.append(score/n_games)
    plt.plot(scores)
    plt.draw()
    plt.pause(0.001)

    print('DONE')

trainer = Trainer(game, net)
time_start = time.time()
trainer.train(epochs=epochs, games=games, gens=gens, sims=sims, callback=callback)

node_names = {-1-i: 'X' + str(i) for i in range(n2)}
node_names.update({-1-i-n2: 'O' + str(i) for i in range(n2)})
node_names.update({i: str(i) for i in range(n2)})
node_names.update({n2: 'V'})
visualize.draw_net(net.pop.config, net.winner_genome, True, node_names=node_names)

print('Waiting to exit...')
input()