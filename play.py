from examples.connect_fast import State
from mcts import MCTS

state = State()
mcts = MCTS(state)

while True:
	terminal, score = state.score()
	if terminal:
		print(score)
		break

	mcts.think()
	move = mcts.select_action()
	mcts.do_action(move)

	print(mcts.state)