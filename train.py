#from connect4 import Connect4State
from learned_mcts import LearnedMCTS
#from connect4_model import Model
from model_interface import ModelInterface

from nash import NashState
from nash_model import Model

model = ModelInterface(Model())
model.load_parameters('nash')

mcts = LearnedMCTS(NashState, model)

for game in range(2000):
    print('Starting game', game)

    mcts.reset()
    model.clear_training()

    while mcts.state.get_score() == -1:
        mcts.think(max_time=5, max_its=10000)
        mcts.do_action(mcts.select_action())
        print(mcts.state)

    model.train()
    model.plot_loss()

    model.save_parameters()

input()