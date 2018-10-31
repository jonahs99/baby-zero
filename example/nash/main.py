import zero.zero as zero
import zero.nn as nn
import zero.trainer as trainer
import zero.evaluate as evaluate

from example.nash.nash_game import NashGame

game = NashGame(4)

net = nn.NN(game, 'config-nash')
train = trainer.Trainer(game, net)