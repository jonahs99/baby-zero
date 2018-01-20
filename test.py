from connect4 import Connect4State
from mcts import MCTS

from connect4_model import Model
from model_interface import ModelInterface
from learned_mcts import LearnedMCTS

from nash import NashState
from nash_model import Model

from nash import NashState

p1 = MCTS(NashState())

model = ModelInterface(Model())
model.load_parameters('nash')
p2 = LearnedMCTS(NashState, model)

def play_against(a, b):
    while a.state.get_score() == -1:
        p = a if a.state.turn == 1 else b
        p.think(max_time=5)
        action = p.select_action()
        a.do_action(action)
        b.do_action(action)
        print(a.state)

play_against(p1, p2)